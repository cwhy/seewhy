"""Recall-Gen — exp48: exp45 for four times as many steps.

exp45's identification curve flattens over its last decile, and report 16 read
that as the task saturating. It is not. The optimiser uses a cosine decay to a
tenth of its peak learning rate over whatever `steps` is set to, so EVERY run in
this project flattens at its own end. exp41 flattened at 0.488 after 12 000
steps and exp43 flattened at 0.617 after 48 000 — the same shape at different
values, which is the schedule's signature and not the task's.

Three more facts point the same way. The training loss is still falling. The gap
between worlds seen in training and worlds never seen is small (0.614 against
0.542), so the network is underfitting rather than memorising. And quadrupling
the budget once already bought 0.13 of identification.

This run quadruples it again. Nothing else changes.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp48
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
    exp_name="smoke_exp48" if SMOKE else "exp48",
    name="synthetic prior, recall arm, 192k steps",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 192000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 8000,
)

if __name__ == "__main__":
    run(RN)
