"""Recall-Gen — exp21: completion training on a knn context (B1 ceiling arm).

The partner of exp20. Same context construction — the query's 16 nearest
neighbours — but the target is never in it, so the only route to a low loss is to
use the context to predict something it does not contain. This is the arm that
says how much of the 0.552 soft-look-up ceiling is reachable by this model at
all, which is what exp20's number has to be read against.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp21
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
    exp_name="smoke_exp21" if SMOKE else "exp21",
    name="completion training, knn context (target never present)",
    M=16, Q=1, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="gen", ctx_mode="knn",
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
