"""Recall-Gen — exp12: seed replicate of exp2.

The ceiling needs error bars as much as the claim does.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp12
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
    exp_name="smoke_exp12" if SMOKE else "exp12",
    name="exp2 replicate, seed 1",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=1,
    train_mode="gen",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
