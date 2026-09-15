"""exp11 — bracket EGGROLL's grid, which exp10 left open at two edges.

exp10's best cell sat at the top of both the population axis (128 pairs) and the
outer learning-rate axis (3e-2), so neither peak is bracketed and the honest
reading of that grid is "at least this good". Extend both: 512 pairs, and a rate
of 0.1.

Population size is the interesting axis. The gradient estimate's alignment with
the true gradient should scale as sqrt(P/d) for d parameters — 0.048 at P = 128,
0.095 at P = 512 — so quadrupling the population should double the cosine and
roughly quadruple the cost per step.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp11
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
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke11" if SMOKE else "exp11"

SIGMA = 3e-3          # exp10: sigma barely matters, Adam normalises it away
GRID = ((128, 0.1), (512, 0.03), (512, 0.1), (512, 0.3))

CELLS = [
    Cell(exp=EXP, method="eggroll", seed=0, width=32, depth=32, epochs=1,
         sigma=SIGMA, n_pop=pop, lr=lr, tag=f"p{pop}lr{lr:g}",
         n_train=N_TRAIN, n_test=N_TEST)
    for pop, lr in GRID
]

if __name__ == "__main__":
    run_cells(CELLS)
