"""exp14 — EGGROLL on the wall-clock curve, and down the depth sweep.

Two additions to figures that already exist. Ten epochs on the residual network,
so EGGROLL joins the accuracy-against-training-seconds comparison; and the same
depth sweep the other three rules were run through, because EGGROLL is the one
rule here with no notion of a layer at all. If credit assignment through depth is
what separates the local rules, a method that never assigns credit to a layer
should be flat against depth until the network itself stops training.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp14
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
EXP = "smoke14" if SMOKE else "exp14"

POP, SIGMA, LR = 512, 3e-3, 3e-2
DEPTHS = (4,) if SMOKE else (2, 4, 8, 16, 32, 64)

CELLS = (
    # the wall-clock curve, at the shared default rate the other arms use there
    [Cell(exp=EXP, method="eggroll", seed=s, width=32, depth=32, epochs=EPOCHS,
          sigma=SIGMA, n_pop=POP, lr=LR, tag="walltime",
          n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
    # and the depth sweep, one epoch per cell as in exp5
    + [Cell(exp=EXP, method="eggroll", seed=s, width=32, depth=L, epochs=1,
            sigma=SIGMA, n_pop=POP, lr=LR, tag=f"L{L}",
            n_train=N_TRAIN, n_test=N_TEST) for L in DEPTHS for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
