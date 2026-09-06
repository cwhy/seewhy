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

from .core import (Cfg, init_params, n_params, predict, masked_mse,
                   augment as warp)
from . import domains
from . import evalsets
from . import viz

PROJECT = Path(__file__).parent.parent
JSONL = PROJECT / "results.jsonl"


@dataclass
class Run:
    exp_name: str
    name: str
    # task
    domain: str = "mnist"        # "mnist" | "fashion_mnist" | "chess" — what a token IS
    M: int = 16                  # context items per episode
    Q: int = 4                   # queries per episode
    mask_rows: int = 14          # slices hidden: image rows from the bottom (14 =
                                 # bottom half), or board files from the queenside
    # Sample a fresh random mask per TRAINING episode instead of using the
    # domain's fixed one. A synthetic prior meant to cover several real domains
    # has to cover their masks too: MNIST hides its bottom 392 coordinates and
    # chess its queenside 416, and a network trained on one meets a shift at the
    # other on top of the distribution shift being measured. Evaluation always
    # uses the domain's real fixed mask.
    random_mask: bool = False
    mask_frac: tuple = (0.3, 0.7)     # per-episode share of VALID coords hidden
    # Draw a fresh training pool before every block of `eval_every` steps.
    # A synthetic prior's whole premise is that the network infers the world in
    # front of it rather than recognising one it has memorised, and a fixed pool
    # of SYNTH_WORLDS worlds is seen thousands of times over a full run — which
    # is precisely what lets it memorise. TabPFN never reuses a prior draw.
    # Resampling makes the pool effectively unbounded at the cost of one pool
    # regeneration (~1s) per block.
    resample_pool: bool = False
    # training
    batch: int = 256      # the token scan is launch-bound; 4x the batch costs ~1.2x
    steps: int = 12000
    lr: float = 3e-4
    seed: int = 0
    train_mode: str = "recall"   # "recall" | "gen" | "mix"
    augment_train: bool = False  # warp every training image → the pool never repeats
    ctx_mode: str = "iid"        # "iid" | "class" | "knn" — what the context is made of
    knn_offset: int = 0          # ranks skipped in knn mode; dials informativeness
    train_only: tuple | None = None   # top-level params to train; None = all
    p_gen: float = 0.5           # only for train_mode="mix"
    init_from: str | None = None  # exp_name whose params_*.pkl to start from
    snapshot_best: str | None = None  # condition to track; saves params_<exp>_best.pkl
    # data — named "digits" for MNIST's sake; they are class ids in any domain
    # (Fashion-MNIST garment ids, chess game-phase buckets).
    train_digits: tuple | None = None   # None = every class
    held_digits: tuple | None = None    # classes forming the "novel" pool
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
    held       the "novel" pool — the dataset's test split, optionally restricted
               to `held_digits`. When `held_digits` is disjoint from
               `train_digits` this is novel CLASSES, not merely novel items.
    held_same  the test split restricted to the TRAINING classes. Identical to
               `held` unless a class split is in play, in which case it is the
               control that separates "item never seen" from "class never seen".

    What a class is depends on the domain: an MNIST digit, a Fashion-MNIST
    garment, or a chess game-phase bucket. `lib/domains.py` owns that; this
    function only splits on the label it is handed.
    """
    Xtr, ytr, Xte, yte = domains.raw_pools(rn.domain)
    if rn.train_digits is not None:
        keep = np.isin(ytr, rn.train_digits)
        Xtr, ytr = Xtr[keep], ytr[keep]
    same = np.isin(yte, rn.train_digits) if rn.train_digits is not None \
        else np.ones(len(yte), bool)
    Xsame, ysame = Xte[same], yte[same]
    if rn.held_digits is not None:
        keep = np.isin(yte, rn.held_digits)
        Xte, yte = Xte[keep], yte[keep]
    return ({"train": Xtr, "held": Xte, "held_same": Xsame},
            {"train": ytr, "held": yte, "held_same": ysame})


def build_mask(rn: Run) -> np.ndarray:
    """1.0 on the coordinates the query hides, flattened to (d_in,)."""
    return domains.mask_vector(rn.domain, rn.mask_rows)


def build_visible(rn: Run) -> np.ndarray:
    """1.0 on coordinates that are real AND shown — what look-ups are built from."""
    return domains.visible_vector(rn.domain, rn.mask_rows)


# ── training ──────────────────────────────────────────────────────────────────

# Which top-level parameters count as "the embedding". The KDA stack is a fixed
# algorithm — forget, predict, correct, write — so retrieval needs usable keys,
# not learned weights. Freezing `layers` asks whether the similarity metric the
# paper attributes to training lives in the embedding rather than in the mixer.
EMBED = ("W_pix", "W_msk", "role")
EMBED_HEAD = EMBED + ("lnf_g", "lnf_b", "head_W", "head_b")


def freeze_labels(p, train_only: tuple) -> dict:
    """Label tree for optax.multi_transform: 'train' or 'freeze' per leaf.

    Everything inside `layers` is frozen at its random init; a top-level
    parameter is trained only if named in `train_only`.
    """
    lab = {k: ("train" if k in train_only else "freeze") for k in p if k != "layers"}
    lab["layers"] = [{k: "freeze" for k in L} for L in p["layers"]]
    return lab


def n_trainable(p, train_only: tuple | None) -> int:
    if train_only is None:
        return n_params(p)
    return int(sum(np.prod(p[k].shape) for k in train_only))


def class_table(labels: np.ndarray) -> np.ndarray:
    """(n_classes, n_min) index table, every class truncated to the smallest.

    Equal-sized rows so a class can be picked and sampled from with two uniform
    draws inside the jit, with no per-class control flow.
    """
    members = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    n_min = min(len(m) for m in members)
    return np.stack([m[:n_min] for m in members])


def make_block(rn: Run, opt, mask, class_tab: np.ndarray | None = None):
    """One jitted block of `eval_every` optimiser steps, scanned."""
    cfg, M, Q, B = rn.cfg, rn.M, rn.Q, rn.batch
    P = cfg.d_in
    gen_frac = {"recall": 0.0, "gen": 1.0, "mix": rn.p_gen}[rn.train_mode]
    vis_idx = np.flatnonzero(
        domains.visible_vector(rn.domain, rn.mask_rows) > 0.5)     # static
    if rn.ctx_mode == "class":
        assert class_tab is not None, "ctx_mode='class' needs a class table"
        ctab = jnp.array(class_tab)

    valid_j = jnp.array(domains.valid_vector(rn.domain))

    def sample_mask(k):
        """A fresh (B,1,P) mask, or the domain's fixed one when not randomising.

        Bernoulli at a per-episode rate, intersected with the valid coordinates
        so padding is never hidden — there is nothing there to reconstruct.
        """
        if not rn.random_mask:
            return mask
        ku, kf = jax.random.split(k)
        frac = jax.random.uniform(kf, (B, 1, 1), minval=rn.mask_frac[0],
                                  maxval=rn.mask_frac[1])
        return (jax.random.uniform(ku, (B, 1, P)) < frac).astype(jnp.float32) * valid_j

    def block(p, st, pool, key):
        n = pool.shape[0]
        # Hoisted out of the step scan: the visible-half view of the pool is what
        # a knn context is built from, and it costs a gather over the whole pool.
        if rn.ctx_mode == "knn":
            pool_vis = pool[:, vis_idx]
            pool_sq = (pool_vis ** 2).sum(-1)[None, :]

        def structured_sample(k):
            """B1's constructions: a context that is ABOUT the query.

            Built exactly as `evalsets` builds it, so training and evaluation
            agree: a filler context that excludes the queries, then — for the
            target-present case — Q randomly chosen slots overwritten with them.
            """
            kq, ks, kg = jax.random.split(k, 3)
            if rn.ctx_mode == "knn":
                qidx = jax.random.randint(kq, (B * Q,), 0, n)
                qry = pool[qidx]
                qv = qry[:, vis_idx]
                d = (qv ** 2).sum(-1)[:, None] + pool_sq - 2.0 * (qv @ pool_vis.T)
                # Rank 0 is the query itself (distance exactly 0), so drop it and
                # the target is exactly absent from the filler context.
                nb = jax.lax.top_k(-d, M // Q + 1 + rn.knn_offset)[1][:, 1 + rn.knn_offset:]
                filler = pool[nb].reshape(B, M, P)
                qry = qry.reshape(B, Q, P)
            else:   # "class" — the whole episode is one digit class
                kc, kp = jax.random.split(kq)
                c = jax.random.randint(kc, (B, 1), 0, ctab.shape[0])
                # WITHOUT replacement. An earlier version drew positions
                # uniformly and noted the collision rate as ~0.3%, which is true
                # for an MNIST class of several thousand images. A synthetic
                # world holds SYNTH_PER_WORLD items — 48 — and there the rate is
                # 1 - (1 - 1/48)^16 = 29%. That is harmless for recall training,
                # where the target is deliberately written into the context
                # afterwards, and fatal for completion training, whose whole
                # premise is that the answer is absent.
                pos = jnp.argsort(jax.random.uniform(kp, (B, ctab.shape[1])),
                                  axis=1)[:, :M + Q]
                idx = jnp.take_along_axis(ctab[c[:, 0]], pos, axis=1)
                filler, qry = pool[idx[:, :M]], pool[idx[:, M:]]

            slots = jnp.argsort(jax.random.uniform(ks, (B, M)), axis=1)[:, :Q]
            bi = jnp.arange(B)[:, None]
            present = filler.at[bi, slots].set(qry)
            if gen_frac == 0.0:
                return present, qry
            if gen_frac == 1.0:
                return filler, qry
            use = (jax.random.uniform(kg, (B, 1, 1)) < gen_frac)
            return jnp.where(use, filler, present), qry

        def sample(k):
            kc, kq, kf, kg, ka, kb = jax.random.split(k, 6)
            ctx = pool[jax.random.randint(kc, (B, M), 0, n)]              # (B,M,P)
            if rn.augment_train:
                # Warp before the target is selected, so the target-present query
                # is the augmented context image exactly — recall stays exact and
                # only the pool's finiteness is removed.
                ctx = warp(ka, ctx.reshape(B * M, P)).reshape(B, M, P)
            sel = jax.random.randint(kq, (B, Q), 0, M)
            from_ctx = jnp.take_along_axis(ctx, sel[..., None], axis=1)   # target present
            fresh = pool[jax.random.randint(kf, (B, Q), 0, n)]            # target absent
            if rn.augment_train:
                fresh = warp(kb, fresh.reshape(B * Q, P)).reshape(B, Q, P)
            if gen_frac == 0.0:
                return ctx, from_ctx
            if gen_frac == 1.0:
                return ctx, fresh
            use = (jax.random.uniform(kg, (B, 1, 1)) < gen_frac)
            return ctx, jnp.where(use, fresh, from_ctx)

        draw = sample if rn.ctx_mode == "iid" else structured_sample

        def step(carry, k):
            p, st = carry
            kd, km = jax.random.split(k)
            ctx, qry = draw(kd)
            m = sample_mask(km)
            loss, g = jax.value_and_grad(
                lambda pp: masked_mse(predict(pp, ctx, qry, m, cfg), qry, m))(p)
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


def evaluate(eval_fn, p, ev: dict, mask, mean_img, chunk=128, domain: str = "mnist"):
    """Score every condition. `sq_acc` is board-only and NaN elsewhere.

    Normalised MSE is the project's common currency across domains, but on a
    chess board it is unreadable: `sq_acc` says the same thing in the units the
    domain is about — the fraction of hidden squares whose most-likely piece is
    the right one. Its reference point is the all-empty guess, which the
    baselines row carries as `sq_acc_empty`.
    """
    board = domains.get(domain).kind == "board"
    mask_j, mean_j = jnp.array(mask), jnp.array(mean_img)
    out = {}
    for cond, es in ev.items():
        E = es.ctx.shape[0]
        se = sn = sm = smn = 0.0
        hits = nnhits = sqacc = 0.0
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
            if board:
                sqacc += w * domains.piece_accuracy(pred, q, mask, domain)
        out[cond] = dict(
            mse=se, nmse=se / es.mse_mean,
            mse_nn=es.mse_nn, nmse_nn=es.mse_nn / es.mse_mean,
            mse_mean=es.mse_mean,
            id_acc=hits if es.present else float("nan"),
            sq_acc=sqacc if board else float("nan"),
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
    assert not (rn.augment_train and rn.ctx_mode != "iid"), \
        "augmentation and structured contexts have not been combined — the warp " \
        "would move an image away from the neighbours it was chosen for"
    dom = domains.get(rn.domain)
    assert not (rn.augment_train and dom.kind != "image"), \
        f"the warp resamples on a 28x28 grid; domain {rn.domain!r} is not an image"
    assert rn.cfg.d_in == dom.d_in, \
        f"cfg.d_in={rn.cfg.d_in} but domain {rn.domain!r} has tokens of {dom.d_in}"
    pools_np, labels_np = build_pools(rn)
    mask = jnp.array(build_mask(rn))
    mean_img = pools_np["train"].mean(0)
    logging.info(f"pools: train={pools_np['train'].shape} held={pools_np['held'].shape}")

    ev = evalsets.build(pools_np, np.asarray(mask), rn.M, rn.Q, rn.n_eval,
                        mean_img, conditions=rn.conditions, ctx_mode=rn.ctx_mode,
                        labels=labels_np, knn_offset=rn.knn_offset,
                        vis=build_visible(rn))
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
                 f"vs {rn.M * rn.cfg.d_in} floats of context content")

    warmup = min(300, rn.steps // 10)
    sched = optax.warmup_cosine_decay_schedule(0.0, rn.lr, warmup, rn.steps, rn.lr * 0.1)
    inner = optax.adamw(sched, weight_decay=0.01)
    if rn.train_only is None:
        opt = optax.chain(optax.clip_by_global_norm(1.0), inner)
    else:
        missing = [k for k in rn.train_only if k not in p]
        assert not missing, f"train_only names parameters that do not exist: {missing}"
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.multi_transform(
            {"train": inner, "freeze": optax.set_to_zero()}, freeze_labels(p, rn.train_only)))
        logging.info(f"  training {n_trainable(p, rn.train_only)/1e6:.2f}M of "
                     f"{np_/1e6:.2f}M params ({', '.join(rn.train_only)}); "
                     f"all {len(p['layers'])} KDA layers frozen at init")
    st = opt.init(p)

    block = make_block(rn, opt, mask,
                       class_table(labels_np["train"]) if rn.ctx_mode == "class" else None)
    eval_fn = make_eval(rn, mask)

    hist = {"step": [], "loss": [],
            "nmse": {c: [] for c in ev}, "id_acc": {c: [] for c in ev}}
    best_seen = (float("inf"), -1)
    n_blocks = rn.steps // rn.eval_every
    for b in range(n_blocks):
        k_train, kb = jax.random.split(k_train)
        tb = time.perf_counter()
        if rn.resample_pool and b > 0:
            # Seeded off the block index so a run stays reproducible.
            pool = jnp.array(domains.resample(rn.domain, seed=1_000_000 + b))
        p, st, loss = block(p, st, pool, kb)
        loss = float(loss)
        step = (b + 1) * rn.eval_every
        m = evaluate(eval_fn, p, ev, mask, mean_img, domain=rn.domain)
        hist["step"].append(step)
        hist["loss"].append(loss)
        for c in ev:
            hist["nmse"][c].append(m[c]["nmse"])
            hist["id_acc"][c].append(m[c]["id_acc"])
        # The completion arm overfits, so its quoted ceiling is the best value over
        # training rather than the last one. Without a checkpoint at that step the
        # number can be reported but never looked at — no figure, no samples.
        if rn.snapshot_best and m[rn.snapshot_best]["nmse"] < best_seen[0]:
            best_seen = (m[rn.snapshot_best]["nmse"], step)
            with open(PROJECT / f"params_{rn.exp_name}_best.pkl", "wb") as f:
                pickle.dump(jax.tree_util.tree_map(np.asarray, p), f)

        parts = "  ".join(f"{c.split('_')[0]}:{m[c]['nmse']:.3f}"
                          + (f"/{m[c]['id_acc']:.2f}" if ev[c].present else "")
                          for c in ev)
        logging.info(f"  step {step:>6}  loss {loss:.5f}  {parts}  [{time.perf_counter()-tb:.0f}s]")

    final = evaluate(eval_fn, p, ev, mask, mean_img, domain=rn.domain)
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
                                           np.asarray(mask), domain=rn.domain)
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
        best_step=best_seen[1] if rn.snapshot_best else None,
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
