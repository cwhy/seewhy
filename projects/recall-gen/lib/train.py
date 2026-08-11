"""The shared train/eval driver. Experiment files set constants and call `run`.

Every experiment here is the same model on the same task with one thing changed,
so the driver is the control: if it lives in one place, two runs differ only by
what their `Run` says they differ by.
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

from shared_lib.datasets import load_supervised_image

from .core import Cfg, PIX, init_params, n_params, predict, masked_mse, row_mask
from . import evalsets
from . import viz

PROJECT = Path(__file__).parent.parent
JSONL = PROJECT / "results.jsonl"


@dataclass
class Run:
    exp_name: str
    name: str
    # task
    M: int = 16                  # context images per episode
    Q: int = 4                   # queries per episode
    mask_rows: int = 14          # bottom rows hidden (14 = bottom half)
    # training
    batch: int = 256      # the token scan is launch-bound; 4x the batch costs ~1.2x
    steps: int = 12000
    lr: float = 3e-4
    seed: int = 0
    train_mode: str = "recall"   # "recall" | "gen" | "mix"
    p_gen: float = 0.5           # only for train_mode="mix"
    init_from: str | None = None  # exp_name whose params_*.pkl to start from
    # data
    train_digits: tuple | None = None   # None = all ten
    held_digits: tuple | None = None    # pool for the "novel" conditions
    # model
    cfg: Cfg = field(default_factory=Cfg)
    # eval
    n_eval: int = 512
    eval_every: int = 500
    conditions: dict | None = None


# ── data ──────────────────────────────────────────────────────────────────────

def build_pools(rn: Run):
    """Three pools.

    train      episodes are drawn from here during training
    held       the "novel" pool — MNIST's test split, optionally restricted to
               `held_digits`. When `held_digits` is disjoint from `train_digits`
               this is novel CLASSES, not merely novel images.
    held_same  MNIST test split restricted to the TRAINING digits. Identical to
               `held` unless a digit split is in play, in which case it is the
               control that separates "image never seen" from "class never seen".
    """
    ds = load_supervised_image("mnist")
    Xtr = np.asarray(ds.X).reshape(-1, PIX).astype(np.float32) / 255.0
    ytr = np.asarray(ds.y)
    Xte = np.asarray(ds.X_test).reshape(-1, PIX).astype(np.float32) / 255.0
    yte = np.asarray(ds.y_test)
    if rn.train_digits is not None:
        keep = np.isin(ytr, rn.train_digits)
        Xtr, ytr = Xtr[keep], ytr[keep]
    same = np.isin(yte, rn.train_digits) if rn.train_digits is not None \
        else np.ones(len(yte), bool)
    Xsame = Xte[same]
    if rn.held_digits is not None:
        keep = np.isin(yte, rn.held_digits)
        Xte, yte = Xte[keep], yte[keep]
    return ({"train": Xtr, "held": Xte, "held_same": Xsame},
            {"train": ytr, "held": yte})


# ── training ──────────────────────────────────────────────────────────────────

def make_block(rn: Run, opt, mask):
    """One jitted block of `eval_every` optimiser steps, scanned."""
    cfg, M, Q, B = rn.cfg, rn.M, rn.Q, rn.batch
    gen_frac = {"recall": 0.0, "gen": 1.0, "mix": rn.p_gen}[rn.train_mode]

    def block(p, st, pool, key):
        n = pool.shape[0]

        def sample(k):
            kc, kq, kf, kg = jax.random.split(k, 4)
            ctx = pool[jax.random.randint(kc, (B, M), 0, n)]              # (B,M,784)
            sel = jax.random.randint(kq, (B, Q), 0, M)
            from_ctx = jnp.take_along_axis(ctx, sel[..., None], axis=1)   # target present
            fresh = pool[jax.random.randint(kf, (B, Q), 0, n)]            # target absent
            if gen_frac == 0.0:
                return ctx, from_ctx
            if gen_frac == 1.0:
                return ctx, fresh
            use = (jax.random.uniform(kg, (B, 1, 1)) < gen_frac)
            return ctx, jnp.where(use, fresh, from_ctx)

        def step(carry, k):
            p, st = carry
            ctx, qry = sample(k)
            loss, g = jax.value_and_grad(
                lambda pp: masked_mse(predict(pp, ctx, qry, mask, cfg), qry, mask))(p)
            up, st = opt.update(g, st, p)
            return (optax.apply_updates(p, up), st), loss

        (p, st), losses = jax.lax.scan(step, (p, st), jax.random.split(key, rn.eval_every))
        return p, st, losses.mean()

    return jax.jit(block)


# ── evaluation ────────────────────────────────────────────────────────────────

def make_eval(rn: Run, mask):
    cfg = rn.cfg

    @jax.jit
    def fn(p, ctx, qry):
        pred = predict(p, ctx, qry, mask, cfg)
        # distance from the model output to each context image, HIDDEN pixels only:
        # a model that merely copies the visible half cannot score on this.
        d = (((pred[:, :, None, :] - ctx[:, None, :, :]) ** 2) * mask).sum(-1)
        return pred, jnp.argmin(d, axis=-1)

    return fn


def evaluate(eval_fn, p, ev: dict, mask, mean_img, chunk=128):
    mask_j, mean_j = jnp.array(mask), jnp.array(mean_img)
    out = {}
    for cond, es in ev.items():
        E = es.ctx.shape[0]
        se = sn = sm = smn = 0.0
        hits = nnhits = 0.0
        preds = None
        for i in range(0, E, chunk):
            c, q = es.ctx[i:i + chunk], es.qry[i:i + chunk]
            pred, argmin = eval_fn(p, c, q)
            if preds is None:
                preds = np.asarray(pred[:8])
            w = c.shape[0] / E
            se += w * float(masked_mse(pred, q, mask_j))
            nn_pick = jnp.take_along_axis(c, es.nn_idx[i:i + chunk][..., None], axis=1)
            smn += w * float(masked_mse(pred, nn_pick, mask_j))
            sm += w * float(masked_mse(pred, jnp.broadcast_to(mean_j, q.shape), mask_j))
            nnhits += w * float((argmin == es.nn_idx[i:i + chunk]).mean())
            if es.present:
                hits += w * float((argmin == es.tgt_idx[i:i + chunk]).mean())
        out[cond] = dict(
            mse=se, nmse=se / es.mse_mean,
            mse_nn=es.mse_nn, nmse_nn=es.mse_nn / es.mse_mean,
            mse_mean=es.mse_mean,
            id_acc=hits if es.present else float("nan"),
            nn_agree=nnhits,          # does the model's answer point at the look-up pick?
            mse_to_nn=smn,            # how close is the output TO the look-up answer
            mse_to_meanimg=sm,        # ...and to the dataset prior
            preds=preds,
        )
    return out


# ── driver ────────────────────────────────────────────────────────────────────

def already_done(exp_name: str) -> bool:
    if not JSONL.exists():
        return False
    for line in JSONL.read_text().strip().splitlines():
        try:
            if json.loads(line).get("experiment") == exp_name:
                return True
        except json.JSONDecodeError:
            pass
    return False


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def run(rn: Run, make_figs: bool = True) -> dict:
    if already_done(rn.exp_name):
        logging.info(f"{rn.exp_name} already in results.jsonl — skipping")
        return {}

    t0 = time.perf_counter()
    pools_np, _ = build_pools(rn)
    mask = jnp.array(row_mask(rn.mask_rows))
    mean_img = pools_np["train"].mean(0)
    logging.info(f"pools: train={pools_np['train'].shape} held={pools_np['held'].shape}")

    ev = evalsets.build(pools_np, np.asarray(mask), rn.M, rn.Q, rn.n_eval,
                        mean_img, conditions=rn.conditions)
    for c, es in ev.items():
        logging.info(f"  {c:<16} mse_mean={es.mse_mean:.4f}  mse_lookup={es.mse_nn:.4f}"
                     f"  (ratio {es.mse_nn / es.mse_mean:.3f})")

    pool = jnp.array(pools_np["train"])
    key = jax.random.key(rn.seed)
    k_init, k_train = jax.random.split(key)
    p = init_params(k_init, rn.cfg)
    if rn.init_from:
        # Fine-tuning probe: start from another run's weights instead of noise.
        # The shapes must match exactly, so this only works between runs sharing
        # a Cfg — a mismatch raises here rather than training something silently
        # different.
        src = PROJECT / f"params_{rn.init_from}.pkl"
        with open(src, "rb") as f:
            loaded = pickle.load(f)
        shapes = lambda t: [x.shape for x in jax.tree_util.tree_leaves(t)]
        if shapes(loaded) != shapes(p):
            raise ValueError(f"{src} has different shapes than this Cfg")
        p = jax.tree_util.tree_map(jnp.asarray, loaded)
        logging.info(f"  initialised from {src.name}")
    np_ = n_params(p)
    logging.info(f"{rn.exp_name}: {np_/1e6:.2f}M params, state={rn.cfg.state_floats} floats "
                 f"vs {rn.M * PIX} floats of context content")

    warmup = min(300, rn.steps // 10)
    sched = optax.warmup_cosine_decay_schedule(0.0, rn.lr, warmup, rn.steps, rn.lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    st = opt.init(p)

    block = make_block(rn, opt, mask)
    eval_fn = make_eval(rn, mask)

    hist = {"step": [], "loss": [],
            "nmse": {c: [] for c in ev}, "id_acc": {c: [] for c in ev}}
    n_blocks = rn.steps // rn.eval_every
    for b in range(n_blocks):
        k_train, kb = jax.random.split(k_train)
        tb = time.perf_counter()
        p, st, loss = block(p, st, pool, kb)
        loss = float(loss)
        step = (b + 1) * rn.eval_every
        m = evaluate(eval_fn, p, ev, mask, mean_img)
        hist["step"].append(step)
        hist["loss"].append(loss)
        for c in ev:
            hist["nmse"][c].append(m[c]["nmse"])
            hist["id_acc"][c].append(m[c]["id_acc"])
        parts = "  ".join(f"{c.split('_')[0]}:{m[c]['nmse']:.3f}"
                          + (f"/{m[c]['id_acc']:.2f}" if ev[c].present else "")
                          for c in ev)
        logging.info(f"  step {step:>6}  loss {loss:.5f}  {parts}  [{time.perf_counter()-tb:.0f}s]")

    final = evaluate(eval_fn, p, ev, mask, mean_img)
    elapsed = time.perf_counter() - t0

    # Params and history so a figure can be redrawn without a five-minute rerun.
    with open(PROJECT / f"params_{rn.exp_name}.pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, p), f)
    with open(PROJECT / f"history_{rn.exp_name}.pkl", "wb") as f:
        pickle.dump(hist, f)

    urls = {}
    if make_figs:
        rows = []
        for c, es in ev.items():
            nn_pick = np.asarray(jnp.take_along_axis(es.ctx[:8], es.nn_idx[:8][..., None], axis=1))
            rows.append(dict(label=c.split("_")[0],
                             qry=np.asarray(es.qry[:8, 0]), pred=final[c]["preds"][:, 0],
                             nn=nn_pick[:, 0]))
        urls["grid"] = viz.completion_grid(f"recallgen_{rn.exp_name}_grid", rows,
                                           np.asarray(mask))
        urls["curves"] = viz.learning_curves(f"recallgen_{rn.exp_name}_curves", hist,
                                             {c: ev[c].mse_nn / ev[c].mse_mean for c in ev})
        logging.info(f"  grid   -> {urls['grid']}")
        logging.info(f"  curves -> {urls['curves']}")

    row = dict(
        experiment=rn.exp_name, name=rn.name,
        **{k: (list(v) if isinstance(v, tuple) else v)
           for k, v in asdict(rn).items() if k not in ("exp_name", "name", "cfg", "conditions")},
        cfg=rn.cfg._asdict(), state_floats=rn.cfg.state_floats,
        conditions={c: [es_pool, es_present] for c, (es_pool, es_present)
                    in (rn.conditions or evalsets.DEFAULT_CONDITIONS).items()},
        n_params=np_, time_s=round(elapsed, 1),
        final={c: {k: v for k, v in final[c].items() if k != "preds"} for c in final},
        # Derived, but logged rather than left to be recomputed: the paper's
        # numeric lint reads `final`, not the curves, so anything quoted in prose
        # that lives only in `history` has to be allow-listed by hand.
        gain=final["D_novel_absent"]["nmse"] - final["B_novel_present"]["nmse"],
        best_nmse={c: min(hist["nmse"][c]) for c in ev},
        history={"step": hist["step"], "loss": hist["loss"],
                 "nmse": hist["nmse"], "id_acc": hist["id_acc"]},
        urls=urls,
    )
    append_result(row)

    logging.info(f"=== {rn.exp_name} done in {elapsed:.0f}s ===")
    for c in final:
        f = final[c]
        logging.info(f"  {c:<16} nmse={f['nmse']:.4f}  lookup={f['nmse_nn']:.4f}  "
                     f"id_acc={f['id_acc']:.3f}  nn_agree={f['nn_agree']:.3f}  "
                     f"d(out,lookup)={f['mse_to_nn']:.4f}  d(out,mean)={f['mse_to_meanimg']:.4f}")
    return row
