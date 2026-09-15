"""exp8 — does our measured activity step explain the gap to the reference run?

exp6 reproduces the authors' Fashion-MNIST cell to within a point for backprop
but lands PC about five points low. The one deliberate deviation in our
implementation is the activity step: they ship a lookup table of
eta_h = 1/lambda_max per cell (0.235875 here), we measure lambda_max by power
iteration at initialisation and get a step about 2.5% smaller.

This runs the same cell with their tabulated value. If PC moves up, the gap is
the step size; if it does not, the gap is seed variance and batch order, and the
measured step is fine to keep for the depth sweep, where no table exists.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp8
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
EXP = "smoke8" if SMOKE else "exp8"

ETA_H_TABLE = 0.2358854990087397     # configs/eta_best_by_cell.csv, fashion/relu/32/32

CELLS = [
    Cell(exp=EXP, method=m, seed=s, width=32, depth=32, epochs=1,
         state_lr=ETA_H_TABLE, tag="etatable", dataset="fashion_mnist",
         n_train=N_TRAIN, n_test=N_TEST)
    for m in ("pc", "pcalm")
    for s in SEEDS
]

if __name__ == "__main__":
    run_cells(CELLS)
