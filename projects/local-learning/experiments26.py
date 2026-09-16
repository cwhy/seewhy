"""exp26 — feedback alignment as an arm, and the depth at which it stops working.

The implementation is verified (scripts/tmp/check_fa.py: the forward pass is
untouched, and setting B = W reproduces the backprop gradient exactly). What it
does on this project's networks is a real result rather than a bug:

  * on the residual net it tracks backprop at 2 and 4 layers and falls to chance
    by 16, with the B-to-W alignment falling from 0.28 to 0.01 alongside;
  * on a plain ReLU net with two hidden layers it collapses by killing units —
    21% dead at 3e-4, 76% at 3e-3 — where the same net with tanh units reaches
    0.9685 against backprop's 0.9766, which is the literature result.

So the arm is run on the same grid as everything else, and swept over depth,
because depth is what it is sensitive to.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp26
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
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke26" if SMOKE else "exp26"

DEPTHS = (4,) if SMOKE else (2, 4, 8, 16, 32, 64)

CELLS = (
    # the headline cell, five seeds, at the best rate from exp23
    [Cell(exp=EXP, method="fa", seed=s, width=32, depth=32, epochs=1, lr=1e-3,
          tag="headline", n_train=N_TRAIN, n_test=N_TEST) for s in (0, 1, 2, 3, 4)]
    # and the depth sweep, which is where the story is
    + [Cell(exp=EXP, method="fa", seed=s, width=32, depth=L, epochs=1, lr=1e-2,
            tag=f"L{L}", n_train=N_TRAIN, n_test=N_TEST)
       for L in DEPTHS for s in SEEDS]
    # plain net, best rate from exp23
    + [Cell(exp=EXP, method="fa_plain", seed=s, hidden=(66, 66), epochs=1, lr=1e-2,
            tag="headline", n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
