"""Recall-Gen — exp45: the recall arm: its answer was ALWAYS one of the sixteen context items.

exp45 is the RECALL arm of the three-arm comparison reports 17 and 18 are built
on. Its answer was always one of the sixteen context items during training, so
copying was always a valid strategy for it.

It repeats exp43's configuration under a corrected class sampler. exp43 drew an
episode's items from its world WITH replacement, which put a query into the
context 29% of the time; harmless for recall training, fatal for the completion
arm, and the three arms have to share a sampler to be comparable.

All three arms share one prior, one sampler, one seed and one step budget, so a
difference between them is the training objective and nothing else.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp45
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
    exp_name="smoke_exp45" if SMOKE else "exp45",
    name="synthetic prior, the recall arm: its answer was ALWAYS one of the sixteen context items",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
