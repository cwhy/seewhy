"""Recall-Gen — exp34: chess phase split, completion training

The chess arm of the same question. A token is one position, encoded as 8x8x13
one-hot piece planes — 832 numbers against MNIST's 784 — and the query hides the
four queenside files, a-d, leaving files e-h visible.

The class axis is game phase by piece count. The training pool is positions with
24 or more pieces on the board; the novel pool is endgames with 10 or fewer. The
middle bucket, 11 to 23 pieces, is in neither, so the two ends are far apart
rather than adjacent. `held_same` is unseen 24+ positions from games the network
never saw, which separates "position never seen" from "phase never seen".

Chess is the interesting case because its completion task has real structure a
digit's bottom half does not: material is roughly balanced, pawns sit on files,
kings castle to known squares. If prediction transfers anywhere, it should
transfer here.\n\nThis is the completion arm: its answer was NEVER in the context during\ntraining, so copying was never available to it and it could only ever predict.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp34
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run
from experiments8 import SPLIT_CONDITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp34" if SMOKE else "exp34",
    name="phase split: train on 24+ piece positions, novel pool is endgames, completion training",
    domain="chess",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="gen",
    train_digits=(0,), held_digits=(2,),
    conditions=SPLIT_CONDITIONS,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=832),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
