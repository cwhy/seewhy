"""Recall-Gen — exp26: frozen mixer on the ORIGINAL i.i.d. task.

The comparison against exp1, which is where the in-weights degradation was first
measured: absent-target error reaches 0.635 early and rises to 0.843 by the end.
If that rise is in-weights overfitting, freezing the weights should flatten it.

Matches exp1 exactly — i.i.d. context, M=16, Q=4 — except that the four KDA
layers stay at their random init. Note the i.i.d. context carries no information
about an absent target (soft-look-up ceiling 1.002), so the interesting number
here is not how low D goes but whether its CURVE still turns upward.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp26
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run, EMBED_HEAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp26" if SMOKE else "exp26",
    name="iid context, recall, KDA layers frozen at init — the exp1 comparison",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", train_only=EMBED_HEAD,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
