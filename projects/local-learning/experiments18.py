"""exp18 — put EGGROLL on the learning-rate grid with the other arms.

exp3 swept the outer rate for backprop, PC, PC-ALM, Forward-Forward and plain
backprop over five values at two seeds, and that grid is the paper's figure 3.
EGGROLL was left off it, because its own grid (exp10) sweeps three knobs at once
and its population was not yet settled. Now that the population is fixed at 512,
its rate axis is directly comparable and belongs on the same figure.

Same five rates, same two seeds, same one epoch as every other arm on that grid.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp18
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
SEEDS = (0,) if SMOKE else (0, 1)
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke18" if SMOKE else "exp18"

RATES = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)      # exactly exp3's grid

CELLS = [
    Cell(exp=EXP, method="eggroll", seed=s, width=32, depth=32, epochs=1,
         sigma=3e-3, n_pop=512, lr=lr, tag=f"lr{lr:g}",
         n_train=N_TRAIN, n_test=N_TEST)
    for lr in RATES for s in SEEDS
]

if __name__ == "__main__":
    run_cells(CELLS)
