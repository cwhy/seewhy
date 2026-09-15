"""exp9 — the same table, with every method given its own best learning rate.

exp3 and exp7 swept the learning rate for each rule over five values spanning
two orders of magnitude, with both peaks bracketed. The best rate differs by a
factor of thirty across methods: backprop on the residual network wants 3e-2,
PC-ALM 1e-2, PC 3e-3, Forward-Forward 1e-3. The shared default of 1e-3 that the
original comparison used is therefore near-optimal for exactly one of the four,
and off by a factor of thirty for another.

This re-runs the whole head-to-head at each method's own best rate, five seeds,
one epoch, everything else unchanged. It is the version of the table that holds
the learning rate fixed *in effort* rather than fixed in value.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp9
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
EXP = "smoke9" if SMOKE else "exp9"

# best rate per arm, from the exp3 grid extended by exp7
BEST_LR = {"bp": 3e-2, "pc": 3e-3, "pcalm": 1e-2, "ff": 1e-3, "bp_plain": 3e-3}
BEST_LR_FF_DEEP = 3e-3

CELLS = (
    [Cell(exp=EXP, method=m, seed=s, width=32, depth=32, lr=BEST_LR[m], tag="tuned",
          epochs=1, n_train=N_TRAIN, n_test=N_TEST)
     for m in ("bp", "pc", "pcalm") for s in SEEDS]
    + [Cell(exp=EXP, method=m, seed=s, hidden=(66, 66), lr=BEST_LR[m], tag="tuned",
            epochs=1, n_train=N_TRAIN, n_test=N_TEST)
       for m in ("ff", "bp_plain") for s in SEEDS]
    + [Cell(exp=EXP, method="ff", seed=s, hidden=(32,) * 31, lr=BEST_LR_FF_DEEP,
            tag="tuned_deep", epochs=1, n_train=N_TRAIN, n_test=N_TEST)
       for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
