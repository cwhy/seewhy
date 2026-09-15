"""exp5 — depth, which is the axis the three rules actually disagree on.

PC-ALM's claim is about deep, narrow networks: standard predictive coding loses
the supervision signal before it reaches the early layers, and the dual
variables are what carry it. At depth 32 — the tweet's cell — that gap should
already be visible; at depth 2 it should vanish.

Forward-Forward is swept along the same axis on its own plain stack (a local
rule with no dual variables at all, and no skip connections), with backprop on
that same stack as the control. Widths are held at 32 so that depth is the only
thing moving; parameter count therefore grows with depth, which is the same
choice the PC-ALM paper makes.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp5
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
DEPTHS = (4, 8) if SMOKE else (2, 4, 8, 16, 32, 64)
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke5" if SMOKE else "exp5"

CELLS = (
    [Cell(exp=EXP, method=m, seed=s, width=32, depth=L, tag=f"L{L}", epochs=1,
          n_train=N_TRAIN, n_test=N_TEST)
     for m in ("bp", "pc", "pcalm") for L in DEPTHS for s in SEEDS]
    # the plain stack: L-1 hidden layers of 32 units, so the layer count matches
    + [Cell(exp=EXP, method=m, seed=s, hidden=(32,) * (L - 1), tag=f"L{L}", epochs=1,
            n_train=N_TRAIN, n_test=N_TEST)
       for m in ("ff", "bp_plain") for L in DEPTHS if L >= 3 for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
