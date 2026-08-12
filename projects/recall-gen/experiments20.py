"""Recall-Gen — exp20: recall training on a context worth using (B1).

The paper's finding — recall training buys no generalisation — was measured on a
task where the context contained nothing to generalise from: at M=16 the best
possible soft look-up from an i.i.d. context scores 1.002 on an absent target,
i.e. exactly as well as ignoring the context. No objective can reward using a
context that carries no information.

Here the context is the query's own 16 nearest neighbours in the training pool,
ranked by distance on the VISIBLE half — the half the model is given. That is
the same quantity the soft-look-up ceiling is built from, so the ceiling is set
rather than discovered, and it is label-free, unlike the same-class construction
in exp23. Measured before training (`baselines_M16_r14_knn_Q1`):

    soft look-up, absent target   0.552      the context IS worth using
    ridge, which ignores context  0.631      ...and using it beats ignoring it
    same numbers on i.i.d. context 1.002 / 0.645

Q=1 rather than 4 because with a knn context Q divides the neighbours: each query
gets M/Q of them, and 16 neighbours (0.552) are worth appreciably more than 4
(0.604).

The question the paper could not ask: **does a recall objective learn to use a
context that is genuinely worth using?** Compare against exp21 (completion
training, the ceiling arm) and exp22 (mixed).

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp20
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp20" if SMOKE else "exp20",
    name="recall training, knn context (M=16 nearest neighbours of the query)",
    M=16, Q=1, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="knn",
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
