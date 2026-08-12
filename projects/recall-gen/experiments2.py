"""Recall-Gen — exp2: the generalisation ceiling. Train with the target ABSENT.

Identical to exp1 in every way except `train_mode="gen"`: the query image is
never one of the M context images, so retrieval is useless and the model must
learn a completion prior. This is the reference exp1 needs — without it,
"recall-only training does not generalise" has no scale: we would not know
whether the architecture *could* have.

Note the scoring is unchanged, so exp2 is also the mirror-image question: does a
model trained only to generalise retain any ability to recall (conditions A/B)?

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp2
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
    exp_name="smoke_exp2" if SMOKE else "exp2",
    name="generalisation-trained ceiling (target never in context)",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="gen", snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
