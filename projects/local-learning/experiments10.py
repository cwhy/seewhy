"""exp10 — what does EGGROLL need before it can be compared with anything?

EGGROLL is the fourth learning rule here and the only gradient-free one: it
estimates the gradient from paired weight perturbations rather than computing
it. Two knobs decide whether that estimate is worth anything — the perturbation
scale sigma, and the number of antithetic pairs per update — and neither has a
default that transfers from the papers, which use very different networks.

This is the same courtesy the other three arms got in exp3: a grid, run before
any head-to-head, so the comparison is against a configuration that works
rather than one that happened to be first. Grid over sigma, population size and
the outer Adam rate, one seed, one epoch.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp10
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
EXP = "smoke10" if SMOKE else "exp10"

SIGMAS = (1e-2,) if SMOKE else (1e-3, 3e-3, 1e-2, 3e-2)
POPS = (32,) if SMOKE else (32, 128)
RATES = (1e-2,) if SMOKE else (1e-3, 1e-2, 3e-2)

CELLS = [
    Cell(exp=EXP, method="eggroll", seed=0, width=32, depth=32, epochs=1,
         sigma=sig, n_pop=pop, lr=lr, tag=f"s{sig:g}p{pop}lr{lr:g}",
         n_train=N_TRAIN, n_test=N_TEST)
    for sig in SIGMAS for pop in POPS for lr in RATES
]

if __name__ == "__main__":
    run_cells(CELLS)
