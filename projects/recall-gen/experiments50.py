"""Recall-Gen — exp50: the completion arm at twice the width.

exp45, exp46 and exp47 compared the three training objectives at d_model=256.
exp49 then showed that width was the binding constraint there: at d_model=512 the
recall arm's identification on held-out worlds of its own prior went from 0.167
to 0.540. A comparison of objectives run at a capacity where none of them learns
much is not a comparison of objectives.

This is the completion counterpart of exp49, so the three arms can be compared again
at a width where the recall arm demonstrably works.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp50
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.domains import PAD_W, get
from lib.train import Run, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
TRAIN_CLS = get("synth").split[0]

RN = Run(
    exp_name="smoke_exp50" if SMOKE else "exp50",
    name="synthetic prior, completion arm, d_model=512",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="gen", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
