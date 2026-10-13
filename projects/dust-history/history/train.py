"""Train with DUST estimates whose population is the last N updates' estimates.

Each update makes one gradient estimate from one noisy pass (history/estimator.py) and pushes it into a window of the last
N estimates. The weights move along the window's *consensus*:

    mean    the plain average of the window (a boxcar average: momentum with a hard edge)
    agree   the average, kept only where at least `tau` of the window agrees on the sign, zero elsewhere

Learning rates are DUST's per-group rates (token embedding 1000, value embeddings 0.3, scalars 0.03) times a base rate `lr`.
No momentum: the window plays its part. N = 1 is plain one-nudge SGD, the baseline everything else is read against.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import history  # noqa: F401  (path setup)
from history import estimator
from lib import data, dust
from lib.model import SIZES, init_params, n_params
from lib.train import make_eval

JSONL = Path(__file__).resolve().parents[1] / "results.jsonl"


@dataclass(frozen=True)
class Cell:
    exp: str
    N: int                      # window: number of past estimates averaged (1 = no history)
    lr: float
    steps: int | None = None    # updates; default is the budget's own count
    budget: str = "1M"          # which token file to draw batches from: 1M (61 updates) or 10M (610)
    agg: str = "mean"           # mean | agree
    tau: float = 0.0            # agree: fraction of the window that must share the sign (0.5 = no filter)
    seed: int = 42
    size: str = "L8d512"
    tag: str = ""

    @property
    def key(self) -> str:
        return "|".join(str(v) for v in (self.exp, self.budget, self.steps, self.N, self.lr, self.agg, self.tau, self.seed, self.size, self.tag))


class Window:
    """The last N estimates, kept in bfloat16, with a running float32 sum of exactly what is kept."""

    def __init__(self, N: int, like: dict):
        self.N, self.ring = N, []
        self.total = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, jnp.float32), like)
        self._add = jax.jit(lambda t, q, sign: jax.tree_util.tree_map(lambda a, b: a + sign * b.astype(jnp.float32), t, q))
        self._q = jax.jit(lambda g: jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), g))

    def push(self, g: dict) -> None:
        q = self._q(g)
        self.total = self._add(self.total, q, 1.0)
        self.ring.append(q)
        if len(self.ring) > self.N:
            self.total = self._add(self.total, self.ring.pop(0), -1.0)

    def mean(self) -> dict:
        n = len(self.ring)
        return jax.tree_util.tree_map(lambda t: t / n, self.total)

    def agreed(self, tau: float) -> dict:
        """Mean, zeroed where fewer than a fraction `tau` of the window shares the mean's sign."""
        n = len(self.ring)

        def leaf(t, *qs):
            share = sum((jnp.sign(q.astype(jnp.float32)) == jnp.sign(t)).astype(jnp.float32) for q in qs) / n
            return jnp.where(share >= tau, t / n, 0.0)

        return jax.tree_util.tree_map(leaf, self.total, *self.ring)


@jax.jit
def _step(params: dict, direction: dict, lrs: dict) -> dict:
    return jax.tree_util.tree_map(lambda p, g, lr: p - lr * g, params, direction, lrs)


def run_cell(cell: Cell) -> dict:
    cfg = SIZES[cell.size]
    steps = cell.steps or data.STEPS[cell.budget]
    params = init_params(cfg, jax.random.PRNGKey(cell.seed))
    batches = data.train_batches(cell.budget, cell.seed, steps)
    val_eval, test_eval = make_eval(cfg, "val"), make_eval(cfg, "test")
    key = jax.random.PRNGKey(cell.seed + 3)
    lrs = dust.group_lrs(params, cell.lr)
    window = Window(cell.N, params)

    every = max(1, steps // 30) if steps > 61 else data.EVAL_EVERY["1M"]
    v0 = val_eval(params)
    best_val, best_params = v0, params
    hist = dict(step=[0], val_loss=[v0], train_loss=[float("nan")], elapsed=[0.0])
    step_s: list[float] = []
    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        ids, tg = [jnp.asarray(a) for a in data.split_xy(batches[step - 1])]
        t0 = time.perf_counter()
        g, loss = estimator.estimate(params, ids, tg, jax.random.fold_in(key, step), cfg)
        window.push(g)
        direction = window.agreed(cell.tau) if cell.agg == "agree" else window.mean()
        params = _step(params, direction, lrs)
        jax.block_until_ready(params)
        step_s.append(time.perf_counter() - t0)
        train_loss = float(loss)
        if step == 1 or step % every == 0 or step == steps:
            v = val_eval(params)
            hist["step"].append(step); hist["val_loss"].append(v)
            hist["train_loss"].append(train_loss); hist["elapsed"].append(time.perf_counter() - t_start)
            if np.isfinite(v) and v < best_val:
                best_val, best_params = v, params
            logging.info("%s step %d/%d  train %.4f  val %.4f  (%.2fs/step)", cell.key, step, steps, train_loss, v,
                         np.median(step_s[1:] or step_s))
    test = test_eval(best_params)
    return dict(
        experiment=cell.exp, name=cell.key, **{k: v for k, v in asdict(cell).items() if k != "exp"},
        tokens=steps * data.BATCH_SEQS * cfg.seq, n_params=n_params(params), val_init=v0, best_val=best_val,
        test_at_best_val=test, final_val=hist["val_loss"][-1], step_s=float(np.median(step_s[1:] or step_s)),
        first_step_s=step_s[0], time_s=float(sum(step_s)), history=hist,
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


def run_cells(cells: list[Cell]) -> None:
    done = done_keys()
    for cell in cells:
        if cell.key in done:
            logging.info("skip (done): %s", cell.key)
            continue
        row = run_cell(cell)
        if not cell.exp.startswith("smoke"):
            with open(JSONL, "a") as f:
                f.write(json.dumps(row) + "\n")
        logging.info("DONE %s  best_val %.4f  test %.4f  %.2fs/step", cell.key, row["best_val"], row["test_at_best_val"], row["step_s"])
