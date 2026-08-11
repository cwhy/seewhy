"""Recall-Gen — exp7: the completion ceiling at M=64, to pair with exp4.

exp4's reference point. Without it, exp4's absent-target numbers cannot be read
as anything other than 'worse than exp1's ceiling at a different M'.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp7
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
    exp_name="smoke_exp7" if SMOKE else "exp7",
    name="generalisation-trained ceiling at M=64",
    M=64, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="gen",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=68),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
