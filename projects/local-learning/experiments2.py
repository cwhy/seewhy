"""exp2 — the tweet's third row: Forward-Forward, and what its 56,232 hides.

The tweet's Forward-Forward row has 56,232 parameters. That is exactly a plain
784-66-66 MLP with biases, which is what Forward-Forward needs: no output layer,
because classification is done by sweeping all ten labels through the network
and taking the one with the highest goodness.

So the three rows are matched on parameter count and on nothing else. Rows 1-2
are 32 layers deep and 32 wide; row 3 is two layers of 66. This experiment runs
Forward-Forward on its own architecture, and then runs backprop on that same
architecture, which is the control the tweet is missing: it separates "this
learning rule is better" from "this rule was allowed a better-shaped network".

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp2
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
EXP = "smoke2" if SMOKE else "exp2"

CELLS = (
    # Forward-Forward on its own two-layer network — the tweet's row 3
    [Cell(exp=EXP, method="ff", seed=s, hidden=(66, 66), epochs=1,
          n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
    # backprop on the same two-layer network: the architecture control
    + [Cell(exp=EXP, method="bp_plain", seed=s, hidden=(66, 66), epochs=1,
            n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
    # and Forward-Forward forced into the deep-narrow shape the other two rules use
    + [Cell(exp=EXP, method="ff", seed=s, hidden=(32,) * 31, tag="deep", epochs=1,
            n_train=N_TRAIN, n_test=N_TEST) for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
