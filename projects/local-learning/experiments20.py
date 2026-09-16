"""exp20 — bracket the five-epoch peaks that exp19 left at the edge of its grid.

exp19 found five arms whose best five-epoch rate was the smallest rate it tried,
which means those peaks are unbracketed and "best" only means "best of what was
run". One more point below each, same single seed, same five epochs.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp20
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
EXP = "smoke20" if SMOKE else "exp20"

CELLS = (
    [Cell(exp=EXP, method="pc", seed=0, width=32, depth=32, epochs=EPOCHS, lr=3e-4,
          tag="lr0.0003", n_train=N_TRAIN, n_test=N_TEST)]
    + [Cell(exp=EXP, method="ff", seed=0, hidden=(66, 66), epochs=EPOCHS, lr=1e-4,
            tag="lr0.0001", n_train=N_TRAIN, n_test=N_TEST)]
    + [Cell(exp=EXP, method="bp_plain", seed=0, hidden=(66, 66), epochs=EPOCHS, lr=3e-4,
            tag="lr0.0003", n_train=N_TRAIN, n_test=N_TEST)]
    + [Cell(exp=EXP, method="ff", seed=0, hidden=(32,) * 31, epochs=EPOCHS, lr=3e-4,
            tag="deep_lr0.0003", n_train=N_TRAIN, n_test=N_TEST)]
    + [Cell(exp=EXP, method="eggroll_plain", seed=0, hidden=(66, 66), epochs=EPOCHS,
            lr=1e-3, sigma=3e-4, n_pop=512, tag="lr0.001",
            n_train=N_TRAIN, n_test=N_TEST)]
)

if __name__ == "__main__":
    run_cells(CELLS)
