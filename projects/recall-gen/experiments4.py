"""Recall-Gen — exp4: does a bigger context make pure look-up generalise?

Nearest-neighbour is a universal learner in the limit of enough neighbours. At
M=16 copying the closest context image is WORSE than predicting the dataset
mean, so a look-up model has nothing to gain from generalising. As M grows the
look-up ceiling improves on its own — the question is whether the recall-trained
model tracks it. Also a capacity test: the state is 16 384 floats against
64 x 784 = 50 176 floats of context content.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp4
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
    exp_name="smoke_exp4" if SMOKE else "exp4",
    name="recall training, M=64 context images",
    M=64, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=68),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
