"""exp5 — EGGROLL at 1M tokens: the Adam learning-rate and noise-scale grid.

EGGROLL here is the setting of `local-learning/lib/es.py` -- rank-1 perturbations,
antithetic pairs, raw loss differences, Adam -- applied to the paper's transformer. Its
two free constants, the Adam rate and the perturbation scale sigma, are tuned per
population on seed 42, as the paper tunes every cell. sigma's range comes from the
cosine probe (`scripts/tmp/test_eggroll.py`): alignment with backprop peaks near 1e-2 and
is already worse at 3e-2, so 3e-3 and 1e-2 bracket it. Population is forward passes of the
batch, two per antithetic pair.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 1 exp5
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
EXP = "smoke5" if SMOKE else "exp5"

# 64 passes is cheap, so it gets the full lr x sigma grid. sigma = 3e-3 trained better than 1e-2
# there (7.815 vs 7.925 at lr 3e-3), although the cosine probe could not tell them apart, so the
# 256-pass cells -- ~9 minutes each -- use 3e-3 and three rates.
GRID = {64: [(lr, sigma) for sigma in (3e-3, 1e-2) for lr in (3e-4, 1e-3, 3e-3, 1e-2)],
        256: [(lr, 3e-3) for lr in (1e-3, 3e-3, 1e-2)]}
CELLS = [Cell(exp=EXP, method="eggroll", budget="1M", pop=pop, seed=42, lr=lr, sigma=sigma, momentum=0.9,
              tag=f"lr{lr:g}_s{sigma:g}", steps=3 if SMOKE else None)
         for pop in ((64,) if SMOKE else (64, 256)) for lr, sigma in (GRID[pop][:1] if SMOKE else GRID[pop])]

if __name__ == "__main__":
    run_cells(CELLS)
