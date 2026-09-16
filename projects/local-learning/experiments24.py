"""exp24 — Forward-Forward with the negatives Hinton actually uses.

The audit of our Forward-Forward implementation (reports/03) found the mechanics
correct but one documented deviation expensive: we sample the negative label
uniformly at random, where Hinton samples the wrong label the network currently
finds plausible. A uniformly random wrong label is rejected by the first layer
on the label pixels alone, so every layer above it is handed a task that is
already solved — which is exactly the symptom the audit found, layers 3 and 4 of
a four-layer net sitting near chance.

This runs the arm again with hard negatives, on both the two-layer network and
the deep stack, over a rate grid. The headline question is narrow and matters:
the paper claims backprop beats Forward-Forward on Forward-Forward's own
architecture by 2.5 points, and that claim was measured against the weaker
negatives.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp24
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
EXP = "smoke24" if SMOKE else "exp24"

RATES = (1e-3,) if SMOKE else (1e-4, 3e-4, 1e-3, 3e-3)

CELLS = (
    [Cell(exp=EXP, method="ff", seed=s, hidden=(66, 66), epochs=1, lr=lr,
          hard_negatives=True, tag=f"hard_lr{lr:g}", n_train=N_TRAIN, n_test=N_TEST)
     for lr in RATES for s in SEEDS]
    + [Cell(exp=EXP, method="ff", seed=s, hidden=(32,) * 31, epochs=1, lr=lr,
            hard_negatives=True, tag=f"harddeep_lr{lr:g}", n_train=N_TRAIN, n_test=N_TEST)
       for lr in RATES for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
