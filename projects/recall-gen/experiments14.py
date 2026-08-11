"""Recall-Gen — exp14: the from-scratch control for exp13.

2 000 steps of completion training from random initialisation. Identical to
exp13 in every respect except that exp13 starts from exp1's recall-trained
weights and this one starts from noise. The pair is the whole measurement: on
its own, "the fine-tuned model reached X" says nothing.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp13 exp14
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
    exp_name="smoke_exp14" if SMOKE else "exp14",
    name="from-scratch control: 2000 steps of completion training",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 2000, lr=3e-4, seed=0,
    train_mode="gen",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 200,
)

if __name__ == "__main__":
    run(RN)
