"""exp3 — is any method being held back by a learning rate that suits another?

exp1 and exp2 give every method Adam at the rate its own source prescribes:
the PC-ALM paper's rule eta0*sqrt(N/L) for the residual runs, 1e-3 for the plain
ones. A comparison across learning rules is only worth reading if none of them
is sitting at a rate that happens to suit a different one, so this sweeps four
rates per method and reports the best each can do in one epoch.

Two seeds per cell: enough to see a rate that is simply broken, not enough to
call a winner, which is not what this experiment is for.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp3
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
EXP = "smoke3" if SMOKE else "exp3"

RATES = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)

CELLS = (
    [Cell(exp=EXP, method=m, seed=s, width=32, depth=32, lr=lr, tag=f"lr{lr:g}",
          epochs=1, n_train=N_TRAIN, n_test=N_TEST)
     for m in ("bp", "pc", "pcalm") for lr in RATES for s in SEEDS]
    + [Cell(exp=EXP, method=m, seed=s, hidden=(66, 66), lr=lr, tag=f"lr{lr:g}",
            epochs=1, n_train=N_TRAIN, n_test=N_TEST)
       for m in ("ff", "bp_plain") for lr in RATES for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
