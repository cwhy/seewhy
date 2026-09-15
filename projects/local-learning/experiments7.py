"""exp7 — extend the learning-rate grid where exp3 ran out of road.

exp3's grid put backprop's best at its top end (3e-2) and PC-ALM's one notch
below, which means neither peak is bracketed: a method whose best rate sits at
the edge of the grid may simply want a larger one. Add 0.1 for both.

The deep Forward-Forward stack also needs its own grid, since exp2 ran it at the
rate that suits the two-layer network and exp5 sweeps depth at that same rate.
If the 31-layer collapse is a learning-rate artifact rather than a property of
the method, this is where it shows up.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp7
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
EXP = "smoke7" if SMOKE else "exp7"

CELLS = (
    [Cell(exp=EXP, method=m, seed=s, width=32, depth=32, lr=0.1, tag="lr0.1",
          epochs=1, n_train=N_TRAIN, n_test=N_TEST)
     for m in ("bp", "pcalm") for s in SEEDS]
    + [Cell(exp=EXP, method="ff", seed=s, hidden=(32,) * 31, lr=lr,
            tag=f"deep_lr{lr:g}", epochs=1, n_train=N_TRAIN, n_test=N_TEST)
       for lr in (3e-4, 1e-3, 3e-3, 1e-2) for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
