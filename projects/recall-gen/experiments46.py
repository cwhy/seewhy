"""Recall-Gen — exp46: the completion arm: its answer was NEVER in the context.

exp46 is the COMPLETION arm. Its answer was never among the sixteen context
items during training, so copying was never available and it could only predict.

This is the closer analogue of what TabPFN actually does: a context of examples
from one world, and a query that is not one of them. If the synthetic prior buys
anything at all, this is the arm that should show it.

All three arms share one prior, one sampler, one seed and one step budget, so a
difference between them is the training objective and nothing else.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp46
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
    exp_name="smoke_exp46" if SMOKE else "exp46",
    name="synthetic prior, the completion arm: its answer was NEVER in the context",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="gen", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
