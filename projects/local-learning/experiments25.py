"""exp25 — the Forward-Forward arms again, with hard negatives, five seeds.

The audit (reports/03) found our uniform random negatives cost Forward-Forward
2 points on the two-layer network and 28 on the deep stack: a random wrong label
is rejected by layer 1 on the label pixels alone, so the layers above it are
handed a solved task. Hinton samples the wrong label the network currently finds
plausible, which keeps it hard all the way up.

Two of the paper's claims were measured against the weaker negatives and have to
be re-measured: that backprop beats Forward-Forward on Forward-Forward's own
architecture, and that Forward-Forward collapses on the deep-narrow one.

Best rates from exp24's grid: 3e-3 on the two-layer net, 1e-3 on the deep stack.
Five seeds at one epoch to match the headline table, three at five epochs to
match the five-epoch report.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp25
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
SEEDS = (0,) if SMOKE else (0, 1, 2, 3, 4)
SEEDS_5EP = (0,) if SMOKE else (0, 1, 2)
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke25" if SMOKE else "exp25"

SHALLOW, DEEP = (66, 66), (32,) * 31

CELLS = (
    [Cell(exp=EXP, method="ff", seed=s, hidden=SHALLOW, epochs=1, lr=3e-3,
          hard_negatives=True, tag="hard", n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
    + [Cell(exp=EXP, method="ff", seed=s, hidden=DEEP, epochs=1, lr=1e-3,
            hard_negatives=True, tag="hard_deep", n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
    + [Cell(exp=EXP, method="ff", seed=s, hidden=SHALLOW, epochs=5, lr=3e-3,
            hard_negatives=True, tag="hard_5ep", n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS_5EP]
    + [Cell(exp=EXP, method="ff", seed=s, hidden=DEEP, epochs=5, lr=1e-3,
            hard_negatives=True, tag="hard_deep_5ep", n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS_5EP]
)

if __name__ == "__main__":
    run_cells(CELLS)
