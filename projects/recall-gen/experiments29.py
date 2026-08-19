"""Recall-Gen — exp29: frozen layers, digit split, recall training.

The frozen-layer runs so far trained on all ten digits, so they cannot appear in
a figure about digits 0-4 versus 5-9: nothing would be novel to them. This is the
frozen counterpart of exp8 — same split, same objective, same everything except
that the four KDA layers stay at their random initialisation and only the
embedding and output head train.

The question it answers: the frozen network is the better generaliser on
held-out IMAGES. Does that survive held-out CLASSES, where the recall-trained
network collapses to the average-digit score?

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp29
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run, EMBED_HEAD
from experiments8 import SPLIT_CONDITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp29" if SMOKE else "exp29",
    name="digit split 0-4, recall training, KDA layers frozen at init",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", train_only=EMBED_HEAD,
    train_digits=(0, 1, 2, 3, 4), held_digits=(5, 6, 7, 8, 9),
    conditions=SPLIT_CONDITIONS,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
