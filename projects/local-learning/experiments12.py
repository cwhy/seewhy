"""exp12 — EGGROLL's population size, the knob the other three rules do not have.

The gradient estimate is an average of P paired perturbations in d = 56,128
dimensions. Theory says its cosine with the true gradient grows as sqrt(P/d),
so the alignment is terrible at any affordable P — 0.02 at 32 pairs — and the
method has to work by averaging a nearly-orthogonal estimate over many steps
rather than by following a good one.

Sweep P over two orders of magnitude at the best rate from exp10/exp11, three
seeds, and log both the accuracy and the cosine. The cosine is the interesting
column: it is the one number that separates "estimating the gradient badly" from
"estimating something else".

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp12
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
EXP = "smoke12" if SMOKE else "exp12"

POPS = (32,) if SMOKE else (32, 128, 512, 2048)

CELLS = [
    Cell(exp=EXP, method="eggroll", seed=s, width=32, depth=32, epochs=1,
         sigma=3e-3, n_pop=pop, lr=3e-2, tag=f"p{pop}",
         n_train=N_TRAIN, n_test=N_TEST)
    for pop in POPS for s in SEEDS
]

if __name__ == "__main__":
    run_cells(CELLS)
