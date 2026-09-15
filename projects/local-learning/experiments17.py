"""exp17 — a timing calibration, because the GPU stopped being ours alone.

exp1 through exp9 were measured on an idle GPU. Every EGGROLL cell (exp10-16)
was measured while an unrelated job of ours held about 10 GB and a share of the
SMs on the same device, so EGGROLL's wall-clock numbers are not comparable to
the earlier ones, and quietly putting them in the same table would be wrong.

This re-measures two arms with known idle-GPU timings — backprop and PC-ALM at
the headline cell — under the same shared conditions. The ratio between these
and their exp1 numbers is the contention factor, and it is what makes EGGROLL's
times readable against the rest of the table.

Accuracy is unaffected by contention and should reproduce exp1 exactly; if it
does not, something other than scheduling has changed and the calibration is
void.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp17
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
EXP = "smoke17" if SMOKE else "exp17"

CELLS = [
    Cell(exp=EXP, method=m, seed=s, width=32, depth=32, epochs=1, tag="sharedgpu",
         n_train=N_TRAIN, n_test=N_TEST)
    for m in ("bp", "pcalm", "ff")
    for s in SEEDS
]
CELLS = [c for c in CELLS if c.method != "ff"] + [
    Cell(exp=EXP, method="ff", seed=s, hidden=(66, 66), epochs=1, tag="sharedgpu",
         n_train=N_TRAIN, n_test=N_TEST)
    for s in SEEDS
]

if __name__ == "__main__":
    run_cells(CELLS)
