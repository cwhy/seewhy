"""Recall-Gen — exp5: the M-sweep continued, deep into the compressive regime.

256 x 784 = 200 704 floats of context against a 16 384-float state: the model
cannot store the context, only a summary of it. Batch is reduced to 64 because
the token scan is now 260 steps long: reverse-mode AD stores the (B,H,dk,dk)
state at every one of those steps, so activation memory goes as batch x tokens,
and 64 x 260 lands just under exp4's 256 x 68.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp5
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
    exp_name="smoke_exp5" if SMOKE else "exp5",
    name="recall training, M=256 context images",
    M=256, Q=4, mask_rows=14,
    batch=64, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=260),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
