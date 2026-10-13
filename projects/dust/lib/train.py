"""One training driver for backprop, DUST and EGGROLL, so the numbers compare.

A `Cell` is one (method, budget, population, seed) run; `run_cells` executes a
list of them, skips any already in `results.jsonl`, and appends one row each.

Protocol, copied from the DUST paper (Section 3.1):

* one epoch over a fixed training prefix: 100K, 1M or 10M tokens, i.e. 6, 61 or 610
  updates of 8 sequences x 2048 tokens;
* validation loss on 544 held-out sequences every `EVAL_EVERY` updates, and the
  reported number is the *test* loss (544 other sequences) of the checkpoint with the
  best validation loss;
* backprop and DUST share SGD with momentum and the authors' per-group learning rates;
  EGGROLL takes Adam, as in this repo's other EGGROLL arm.

Costs are measured, not assumed: `step_s` is the median wall-clock of an update after
the first (which carries XLA compilation), blocked on the device.
"""

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib import data, dust, eggroll, optim
from lib.model import SIZES, Cfg, cast, forward_loss, init_params, mean_loss, n_params, rope_tables

JSONL = Path(__file__).resolve().parents[1] / "results.jsonl"
CKPT = Path(__file__).resolve().parents[1] / "ckpt"
METHODS = ("bp", "dust", "eggroll")


@dataclass(frozen=True)
class Cell:
    exp: str
    method: str                       # bp | dust | eggroll
    budget: str = "1M"                # 100K | 1M | 10M
    pop: int = 0                      # dust: direct draws K;  eggroll: forward passes (2 per pair);  bp: 0
    seed: int = 42
    size: str = "L8d512"
    lr: float | None = None           # bp/dust: SGD base rate;  eggroll: Adam rate
    momentum: float | None = None     # bp/dust: SGD momentum;   eggroll: Adam beta1
    sigma: float | None = None        # eggroll perturbation scale
    beta2: float = 0.999              # eggroll Adam beta2
    chunk: int = 4                    # eggroll pairs evaluated together
    steps: int | None = None          # override, for smoke tests only
    save_at: tuple = ()               # steps at which to pickle the weights (ckpt/<tag>_step<N>.pkl)
    tag: str = ""
    shaping: str = "raw"              # eggroll fitness: raw loss differences (default) | rank
    dtype: str = "bf16"               # eggroll compute precision: bf16 (default) | f32
    groups: str = "bp"                # bp: per-group learning rates of the released backprop baseline | dust: DUST's
    opt: str = "sgd"                  # bp/dust: sgd | adam | muon (lr = matrix rate, momentum = Adam beta1); eggroll: sgd = its Adam, muon

    @property
    def key(self) -> str:
        k = "|".join(str(v) for v in (self.exp, self.method, self.budget, self.pop, self.seed, self.size,
                                      self.lr, self.momentum, self.sigma, self.beta2, self.tag))
        # appended only when non-default, so keys of rows written before these fields existed are unchanged
        return (k + (f"|shaping={self.shaping}" if self.shaping != "raw" else "") + (f"|dtype={self.dtype}" if self.dtype != "bf16" else "")
                + (f"|groups={self.groups}" if self.groups != "bp" else "") + (f"|opt={self.opt}" if self.opt != "sgd" else ""))


# Backprop's tuned settings from the authors' released baseline (baselines/backprop.py).
BP_DEFAULT = {"1M": (0.175, 0.98), "10M": (0.2, 0.95), "100M": (0.2, 0.95)}


def bp_lrs(params: dict, base: float) -> dict:
    """Backprop uses the base rate everywhere except the token embedding (1000)."""
    lrs = jax.tree_util.tree_map(lambda _: base, params)
    lrs["wte"] = 1000.0
    return lrs


def make_eval(cfg: Cfg, split: str):
    batches = data.eval_batches(split)
    rot = rope_tables(cfg)
    f = jax.jit(lambda pb, x, y: forward_loss(pb, cfg, x, y, rot).mean())

    def evaluate(params) -> float:
        pb = cast(params)
        return float(np.mean([float(f(pb, x, y)) for x, y in batches]))
    return evaluate


def run_cell(cell: Cell) -> dict:
    cfg = SIZES[cell.size]
    rot = rope_tables(cfg)
    steps = cell.steps or data.STEPS[cell.budget]
    params = init_params(cfg, jax.random.PRNGKey(cell.seed))
    batches = data.train_batches(cell.budget, cell.seed, steps)
    val_eval, test_eval = make_eval(cfg, "val"), make_eval(cfg, "test")
    key = jax.random.PRNGKey(cell.seed + 3)

    hp: dict = {}
    if cell.method == "bp":
        lr, mom = (cell.lr, cell.momentum) if cell.lr else BP_DEFAULT[cell.budget]
        # groups="dust" gives backprop DUST's per-group rates (value embeddings 0.3, residual scalars 0.03, token embedding 1000)
        lrs = dust.group_lrs(params, lr) if cell.groups == "dust" else bp_lrs(params, lr)
        buf = jax.tree_util.tree_map(jnp.zeros_like, params)
        grad_fn = jax.jit(jax.value_and_grad(lambda p, x, y: mean_loss(p, cfg, x, y, rot)))
        hp = dict(lr=lr, momentum=mom)
        if cell.opt != "sgd":                      # Adam / Muon, with the paper's Table 7 group rates for backprop
            beta1 = cell.momentum if cell.momentum is not None else 0.8
            buf, opt_update = optim.make(cell.opt, params, optim.group_lrs(params, cell.lr, "bp"), beta1=beta1)
            hp = dict(opt=cell.opt, lr=cell.lr, beta1=beta1, beta2=0.95)

            def update(params, buf, ids, tg, step):
                loss, g = grad_fn(params, ids, tg)
                params, buf = opt_update(params, buf, g)
                return params, buf, float(loss)
        else:
            def update(params, buf, ids, tg, step):
                loss, g = grad_fn(params, ids, tg)
                params, buf = dust.sgd_update(params, buf, g, lrs, jnp.float32(mom))
                return params, buf, float(loss)

    elif cell.method == "dust":
        adaptive = cell.opt != "sgd"
        recipe = dust.make_recipe(cell.budget, cell.pop, cell.lr, None if adaptive else cell.momentum, opt=cell.opt)
        lrs, buf = dust.group_lrs(params, recipe.lr), jax.tree_util.tree_map(jnp.zeros_like, params)
        hp = dict(lr=recipe.lr, momentum=recipe.momentum, gamma=recipe.gamma, nrep=recipe.nrep,
                  population=recipe.population, head_draws=recipe.head)
        if adaptive:                               # Adam / Muon, with the paper's Table 7 group rates for DUST
            beta1 = cell.momentum if cell.momentum is not None else 0.7
            buf, opt_update = optim.make(cell.opt, params, optim.group_lrs(params, cell.lr, "dust"), beta1=beta1)
            hp.update(opt=cell.opt, beta1=beta1, beta2=0.95, sigma_embed=recipe.sigma_embed)
            hp.pop("momentum")

            def update(params, buf, ids, tg, step):
                g, loss = dust.estimate_gradient(params, ids, tg, jax.random.fold_in(key, step), recipe, cfg)
                params, buf = opt_update(params, buf, g)
                return params, buf, float(loss)
        else:
            def update(params, buf, ids, tg, step):
                g, loss = dust.estimate_gradient(params, ids, tg, jax.random.fold_in(key, step), recipe, cfg)
                params, buf = dust.sgd_update(params, buf, g, lrs, jnp.float32(recipe.momentum))
                return params, buf, float(loss)

    elif cell.method == "eggroll":
        assert cell.pop % (2 * cell.chunk) == 0, "pop is forward passes: 2 per pair"
        n_pairs = cell.pop // 2
        opt = eggroll.make_optimizer(cell.lr, cell.momentum if cell.momentum is not None else 0.9, cell.beta2)
        buf = opt.init(params)
        sigma = jnp.float32(cell.sigma)
        hp = dict(lr=cell.lr, beta1=cell.momentum if cell.momentum is not None else 0.9, beta2=cell.beta2,
                  sigma=cell.sigma, n_pairs=n_pairs, chunk=cell.chunk, rank=1)
        clean_loss = jax.jit(lambda p, x, y: mean_loss(p, cfg, x, y, rot))
        apply = jax.jit(lambda p, u: optax.apply_updates(p, u))

        grad_fn = eggroll.es_gradient_ranked if cell.shaping == "rank" else eggroll.es_gradient
        compute = jnp.float32 if cell.dtype == "f32" else jnp.bfloat16
        hp.update(shaping=cell.shaping, dtype=cell.dtype)

        if cell.opt == "muon":                     # Muon on the hidden matrices at cell.lr, Adam at 0.003 on the rest
            beta1 = cell.momentum if cell.momentum is not None else 0.5
            buf, opt_update = optim.make("muon", params, optim.group_lrs(params, cell.lr, "eggroll"), beta1=beta1, muon_lr=cell.lr)
            hp.update(opt="muon", beta1=beta1, beta2=0.95, adam_lr=0.003)

            def update(params, buf, ids, tg, step):
                g, _ = grad_fn(params, ids, tg, jax.random.fold_in(key, step), sigma,
                               cfg=cfg, n_pairs=n_pairs, chunk=cell.chunk, dtype=compute)
                params, buf = opt_update(params, buf, g)
                return params, buf, float(clean_loss(params, ids, tg))
        else:
            def update(params, buf, ids, tg, step):
                g, _ = grad_fn(params, ids, tg, jax.random.fold_in(key, step), sigma,
                               cfg=cfg, n_pairs=n_pairs, chunk=cell.chunk, dtype=compute)
                updates, buf = opt.update(g, buf, params)
                return apply(params, updates), buf, float(clean_loss(params, ids, tg))
    else:
        raise ValueError(cell.method)

    every = data.EVAL_EVERY[cell.budget]
    v0 = val_eval(params)
    best_val, best_params = v0, params
    hist = dict(step=[0], val_loss=[v0], train_loss=[float("nan")], elapsed=[0.0])
    step_s: list[float] = []
    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        ids, tg = [jnp.asarray(a) for a in data.split_xy(batches[step - 1])]
        t0 = time.perf_counter()
        params, buf, train_loss = update(params, buf, ids, tg, step)
        jax.block_until_ready(params)
        step_s.append(time.perf_counter() - t0)
        if step in cell.save_at:
            CKPT.mkdir(exist_ok=True)
            with open(CKPT / f"{cell.tag}_step{step}.pkl", "wb") as f:
                pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        if not np.isfinite(train_loss):
            logging.warning("non-finite train loss at step %d", step)
        if step == 1 or step % every == 0 or step == steps:
            v = val_eval(params)
            hist["step"].append(step); hist["val_loss"].append(v)
            hist["train_loss"].append(train_loss); hist["elapsed"].append(time.perf_counter() - t_start)
            if np.isfinite(v) and v < best_val:
                best_val, best_params = v, params
            logging.info("%s step %d/%d  train %.4f  val %.4f  (%.1fs/step)", cell.key, step, steps,
                         train_loss, v, np.median(step_s[1:] or step_s))
    test = test_eval(best_params)
    return dict(
        experiment=cell.exp, name=cell.key, **{k: v for k, v in asdict(cell).items() if k not in ("exp", "steps")},
        steps=steps, tokens=steps * data.BATCH_SEQS * cfg.seq, n_params=n_params(params), hp=hp,
        val_init=v0, best_val=best_val, test_at_best_val=test, final_val=hist["val_loss"][-1],
        step_s=float(np.median(step_s[1:] or step_s)), first_step_s=step_s[0],
        time_s=float(sum(step_s)), history=hist,
    )


def done_keys() -> set[str]:
    if not JSONL.exists():
        return set()
    out = set()
    for line in JSONL.read_text().splitlines():
        try:
            out.add(json.loads(line)["name"])
        except Exception:
            pass
    return out


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_cells(cells: list[Cell]) -> None:
    done = done_keys()
    for cell in cells:
        if cell.key in done:
            logging.info("skip (done): %s", cell.key)
            continue
        row = run_cell(cell)
        if not cell.exp.startswith("smoke"):        # smoke rows are for eyes, not for results.jsonl
            append_result(row)
        logging.info("DONE %s  best_val %.4f  test %.4f  %.1fs/step", cell.key, row["best_val"],
                     row["test_at_best_val"], row["step_s"])
