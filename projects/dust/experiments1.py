"""exp1 — does the port reproduce the paper's backprop baseline?

Before any zeroth-order method is worth comparing, the network, the data and the
SGD protocol have to give the paper's backprop numbers (Table 1): 5.959 at 1M tokens
and 4.989 at 10M, each a three-seed mean of the test loss at the best-validation
checkpoint. Backprop uses the authors' released settings (lr 0.175 / momentum 0.98
at 1M, 0.2 / 0.95 at 10M).

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 1 exp1
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
EXP = "smoke1" if SMOKE else "exp1"

CELLS = [Cell(exp=EXP, method="bp", budget="1M", seed=s, steps=3 if SMOKE else None) for s in (42, 43, 44)]
CELLS += [Cell(exp=EXP, method="bp", budget="10M", seed=s, steps=3 if SMOKE else None) for s in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS)
