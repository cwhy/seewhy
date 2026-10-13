"""Does the one-nudge estimator point at the true gradient, and more so when N estimates are averaged?

At fixed (initial) weights and one fixed batch, draw 32 independent one-nudge estimates, and compare the mean of the first
1, 4, 16 and 32 of them with the backprop gradient: cosine, and the projection coefficient <mean, g>/<g, g> (1.0 means the
estimate has the right size along the true gradient). No history effects here: the weights never move, so this is the
noise-averaging half of the idea on its own, before staleness enters.

    CUDA_VISIBLE_DEVICES= SIZE=L2d128 uv run --no-sync python projects/dust-history/scripts/check_estimator.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import history  # noqa: F401
from history import estimator
from lib import data
from lib.model import SIZES, init_params, mean_loss, rope_tables

size = os.environ.get("SIZE", "L2d128")
n_draws = int(os.environ.get("DRAWS", "32"))
cfg = SIZES[size]
params = init_params(cfg, jax.random.PRNGKey(42))
ids, tg = [jnp.asarray(a) for a in data.split_xy(data.train_batches("1M", 42, 1)[0])]
rot = rope_tables(cfg)
true = jax.grad(lambda p: mean_loss(p, cfg, ids, tg, rot))(params)


def groups(tree):
    out = {"wte": tree["wte"], "head": tree["head"]}
    for kind in ("proj", "mproj", "fc", "q", "k", "v"):
        out[kind] = jnp.concatenate([b[kind].ravel() for b in tree["blocks"]])
    out["ve"] = jnp.concatenate([b["ve"].ravel() for b in tree["blocks"] if "ve" in b])
    out["gate"] = jnp.concatenate([b["gate"].ravel() for b in tree["blocks"] if "gate" in b])
    return {k: np.asarray(v, np.float64).ravel() for k, v in out.items()}


G = groups(true)
total = jax.tree_util.tree_map(jnp.zeros_like, params)
mark = {1, 4, 16, 32}
print(f"{'N':>3} {'group':>6} {'cosine':>8} {'coef':>8}")
for n in range(1, n_draws + 1):
    g, _ = estimator.estimate(params, ids, tg, jax.random.PRNGKey(1000 + n), cfg)
    total = jax.tree_util.tree_map(lambda a, b: a + b, total, g)
    if n in mark:
        E = groups(jax.tree_util.tree_map(lambda t: t / n, total))
        allc = np.concatenate
        rows = {k: E[k] for k in E}
        for k in ("head", "proj", "mproj", "fc", "q", "k", "v", "wte"):
            e, t = E[k], G[k]
            print(f"{n:>3} {k:>6} {e @ t / (np.linalg.norm(e) * np.linalg.norm(t) + 1e-30):>8.3f} {e @ t / (t @ t + 1e-30):>8.3f}")
        print()
