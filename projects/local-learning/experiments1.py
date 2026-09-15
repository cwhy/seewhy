"""exp1 — the tweet's first two rows, replicated: backprop vs PC-ALM.

The tweet compares three learning rules on MNIST at ~56K parameters. Its first
two rows are the PC-ALM paper's headline cell: a residual MLP of width 32 and
depth 32, one epoch, batch 64 — 56,128 parameters, exactly the count in the
table. Standard predictive coding is added as the third arm, because it is the
control that says whether the dual variables are doing anything.

Five seeds each, because the tweet's numbers are single runs 1.1 points apart
and the interesting question is whether that gap is a gap.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp1
"""

import logging
import os
import sys
from pathlib import Path

# repo root last: the GPU box has an untracked datasets.py at its root that
# would otherwise shadow the HuggingFace `datasets` package
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
SEEDS = (0,) if SMOKE else (0, 1, 2, 3, 4)
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke1" if SMOKE else "exp1"

CELLS = [
    Cell(exp=EXP, method=m, seed=s, width=32, depth=32, epochs=1,
         n_train=N_TRAIN, n_test=N_TEST)
    for m in ("bp", "pc", "pcalm")
    for s in SEEDS
]

if __name__ == "__main__":
    run_cells(CELLS)
