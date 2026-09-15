"""exp13 — EGGROLL as a row in the table, on both architectures.

Best configuration from exp10 and exp11: 512 antithetic pairs, sigma 3e-3, outer
Adam rate 3e-2, both peaks bracketed. Five seeds, one epoch, exactly the protocol
every other arm got.

Three arms. The first two are EGGROLL on the residual network at the shared
default rate of 1e-3 and at its own best rate, because EGGROLL is the starkest
case of the point exp3 made: the same estimator is at chance under the rate the
original comparison shared out, and level with backprop at its own. The third is
EGGROLL on Forward-Forward's two-layer network, which asks whether a
learning rule that is indifferent to architecture cares about this one.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp13
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
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke13" if SMOKE else "exp13"

POP, SIGMA, BEST_LR = 512, 3e-3, 3e-2

CELLS = (
    [Cell(exp=EXP, method="eggroll", seed=s, width=32, depth=32, epochs=1,
          sigma=SIGMA, n_pop=POP, lr=1e-3, tag="default",
          n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
    + [Cell(exp=EXP, method="eggroll", seed=s, width=32, depth=32, epochs=1,
            sigma=SIGMA, n_pop=POP, lr=BEST_LR, tag="tuned",
            n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
    + [Cell(exp=EXP, method="eggroll_plain", seed=s, hidden=(66, 66), epochs=1,
            sigma=SIGMA, n_pop=POP, lr=BEST_LR, tag="tuned",
            n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
