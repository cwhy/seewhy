"""Recall-Gen — exp9: digit split, completion training. exp8's ceiling.

Same 0-4 / 5-9 split as exp8, same six conditions, but the query target is never
in the context, so the model must learn to complete rather than to look up.

The pairing matters for one specific question: exp2 showed that a
completion-trained model memorises the training images instead of using the
context. Under a digit split that memorisation has nowhere to go on B/D — the
classes are new — so this run measures how much of a completion prior actually
transfers across digit identity.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp9
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

SPLIT_CONDITIONS = {
    "A_seen_present":  ("train",     True),
    "B_novel_present": ("held",      True),    # novel image AND novel class
    "C_seen_absent":   ("train",     False),
    "D_novel_absent":  ("held",      False),
    "E_same_present":  ("held_same", True),    # novel image, seen class
    "F_same_absent":   ("held_same", False),
}

RN = Run(
    exp_name="smoke_exp9" if SMOKE else "exp9",
    name="digit split: train on 0-4, novel pool is 5-9, completion training",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="gen",
    train_digits=(0, 1, 2, 3, 4), held_digits=(5, 6, 7, 8, 9),
    conditions=SPLIT_CONDITIONS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
