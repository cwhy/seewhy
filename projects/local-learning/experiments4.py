"""exp4 — accuracy per second of training, not accuracy per epoch.

The tweet's table puts 87.39% next to 0.324s and 86.28% next to 13.485s and
leaves the reader to do the arithmetic. PC-ALM buys its local updates with a
forty-fold increase in training time, so the honest question is what backprop
does with the same forty-fold budget.

Ten epochs for every method, logging test accuracy and cumulative training
seconds after each one. The curves cross, or they do not.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp4
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
EPOCHS = 2 if SMOKE else 10
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke4" if SMOKE else "exp4"

CELLS = (
    [Cell(exp=EXP, method=m, seed=s, width=32, depth=32, epochs=EPOCHS,
          n_train=N_TRAIN, n_test=N_TEST)
     for m in ("bp", "pc", "pcalm") for s in SEEDS]
    + [Cell(exp=EXP, method=m, seed=s, hidden=(66, 66), epochs=EPOCHS,
            n_train=N_TRAIN, n_test=N_TEST)
       for m in ("ff", "bp_plain") for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
