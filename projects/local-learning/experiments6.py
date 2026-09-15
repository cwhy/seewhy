"""exp6 — check our PC / PC-ALM against the authors' own published cell.

The PC-ALM repository states the numbers its reference implementation produces
for one Fashion-MNIST cell — width 32, depth 32, ReLU, seed 0, one epoch:
78.66% for backprop, 68.13% for PC, 77.75% for PC-ALM, with gradient cosines to
backprop of 1.000, 0.604 and 0.909.

Ours is a separate implementation written from the paper's equations. Running
the same cell on the same dataset is the one check that says whether the
reimplementation is faithful, as opposed to merely self-consistent. Three seeds,
because their number is a single CPU run and ours will not match it to the
decimal.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp6
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
SEEDS = (0,) if SMOKE else (0, 1, 2)
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke6" if SMOKE else "exp6"

CELLS = [
    Cell(exp=EXP, method=m, seed=s, width=32, depth=32, epochs=1,
         dataset="fashion_mnist", n_train=N_TRAIN, n_test=N_TEST)
    for m in ("bp", "pc", "pcalm")
    for s in SEEDS
]

if __name__ == "__main__":
    run_cells(CELLS)
