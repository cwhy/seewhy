"""exp19 — re-tune the learning rate at five epochs before comparing at five epochs.

Every rate in this project was chosen on a one-epoch budget, and the best rate
for 937 updates is not generally the best rate for 4,685. Larger budgets usually
prefer smaller steps, so carrying the one-epoch rates over would quietly favour
whichever arm happens to be least sensitive to that.

Three rates per arm, bracketing its one-epoch best, one seed, five epochs. The
head-to-head in exp20 uses whatever wins here.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp19
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
EXP = "smoke19" if SMOKE else "exp19"

# (method, hidden, tag suffix, rates to try) — rates bracket each arm's one-epoch best
RESIDUAL = [
    ("bp", (3e-3, 1e-2, 3e-2)),
    ("pc", (1e-3, 3e-3, 1e-2)),
    ("pcalm", (3e-3, 1e-2, 3e-2)),
    ("eggroll", (1e-2, 3e-2, 1e-1)),
]
PLAIN = [
    ("ff", (66, 66), "", (3e-4, 1e-3, 3e-3)),
    ("bp_plain", (66, 66), "", (1e-3, 3e-3, 1e-2)),
    ("ff", (32,) * 31, "deep", (1e-3, 3e-3, 1e-2)),
    ("eggroll_plain", (66, 66), "", (3e-3, 1e-2, 3e-2)),
]

CELLS = (
    [Cell(exp=EXP, method=m, seed=0, width=32, depth=32, epochs=EPOCHS, lr=lr,
          sigma=3e-3, n_pop=512, tag=f"lr{lr:g}", n_train=N_TRAIN, n_test=N_TEST)
     for m, rates in RESIDUAL for lr in rates]
    + [Cell(exp=EXP, method=m, seed=0, hidden=h, epochs=EPOCHS, lr=lr,
            sigma=3e-4 if m == "eggroll_plain" else 3e-3, n_pop=512,
            tag=f"{t + '_' if t else ''}lr{lr:g}", n_train=N_TRAIN, n_test=N_TEST)
       for m, h, t, rates in PLAIN for lr in rates]
)

if __name__ == "__main__":
    run_cells(CELLS)
