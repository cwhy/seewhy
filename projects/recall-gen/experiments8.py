"""Recall-Gen — exp8: digit split. Train on 0-4, test recall on 5-9.

exp1 showed recall transfers perfectly to novel IMAGES. This asks the harder
version: novel CLASSES. The training pool is MNIST digits 0-4 only, so the model
has never seen a 5, 6, 7, 8 or 9 in any role — not as context, not as a query,
not as a target. The novel pool is digits 5-9 from the test split.

Six conditions rather than four, because "novel image" and "novel class" must be
told apart. `held_same` is the test split restricted to the TRAINING digits, so
E/F isolate image-novelty and B/D add class-novelty on top of it:

    A/C  train pool          seen images, seen classes
    E/F  held_same pool      novel images, SEEN classes
    B/D  held pool           novel images, NOVEL classes

If B matches E, the retrieval mechanism is class-agnostic — it is matching
pixels, not digits. If B is much worse than E, the mechanism leans on
class-specific features it could only have learned from the training digits.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp8
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
    exp_name="smoke_exp8" if SMOKE else "exp8",
    name="digit split: train on 0-4, novel pool is 5-9, recall training",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall",
    train_digits=(0, 1, 2, 3, 4), held_digits=(5, 6, 7, 8, 9),
    conditions=SPLIT_CONDITIONS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
