"""exp23 — feedback alignment: what if the backward weights are just noise?

Backpropagation's backward pass multiplies the error by W^T, the transpose of
the forward weights. That is the weight transport problem: a synapse would have
to know the strength of a different synapse pointing the other way. Feedback
alignment replaces W^T with a fixed random matrix B of the same shape, never
trained, and the forward weights drift into partial alignment with it.

It belongs in this comparison for the same reason the other arms do: it is a way
of training a network that is not backpropagation, and unlike PC-ALM or
Forward-Forward it changes exactly one thing, which makes it the cleanest
possible ablation of the backward pass.

Rate grid first, as every other arm got: the same five rates, two seeds, one
epoch, on both architectures.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp23
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
SEEDS = (0,) if SMOKE else (0, 1)
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke23" if SMOKE else "exp23"

RATES = (3e-4,) if SMOKE else (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)

CELLS = (
    [Cell(exp=EXP, method="fa", seed=s, width=32, depth=32, epochs=1, lr=lr,
          tag=f"lr{lr:g}", n_train=N_TRAIN, n_test=N_TEST)
     for lr in RATES for s in SEEDS]
    + [Cell(exp=EXP, method="fa_plain", seed=s, hidden=(66, 66), epochs=1, lr=lr,
            tag=f"lr{lr:g}", n_train=N_TRAIN, n_test=N_TEST)
       for lr in RATES for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
