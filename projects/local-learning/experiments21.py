"""exp21 — EGGROLL on the plain network was still improving at the bottom of exp20.

Four of the five arms exp20 extended turned over as expected. The fifth did not:
EGGROLL on the 784-66-66 stack went from 0.8934 at 3e-3 to 0.9109 at 1e-3, so
its five-epoch peak is still unbracketed. Two more points below it.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp21
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
EPOCHS = 2 if SMOKE else 5
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke21" if SMOKE else "exp21"

CELLS = [
    Cell(exp=EXP, method="eggroll_plain", seed=0, hidden=(66, 66), epochs=EPOCHS,
         lr=lr, sigma=3e-4, n_pop=512, tag=f"lr{lr:g}",
         n_train=N_TRAIN, n_test=N_TEST)
    for lr in (1e-4, 3e-4)
]

if __name__ == "__main__":
    run_cells(CELLS)
