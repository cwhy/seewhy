"""Recall-Gen — exp19: the M=16 control for exp18.

exp18 blocks memorisation at M=256, where the two-route account says the
in-weights route is the one being taken. At M=16 retrieval already wins, so
blocking memorisation should change little — which is exactly why the run is
needed: without it, any difference exp18 shows could be the warp doing something
to the task rather than to which route is cheaper.

Compare against exp1 (identical but for `augment_train`).

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp19
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
    exp_name="smoke_exp19" if SMOKE else "exp19",
    name="recall training, M=16, augmented pool (memorisation blocked)",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", augment_train=True,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
