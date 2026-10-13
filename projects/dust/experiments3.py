"""exp3 — DUST at 1M tokens, populations 256 and 1,024, the authors' tuned settings.

The first half of the paper's comparison (Table 1, 1M column): does the JAX port of DUST
reach the published loss? Published, three-seed means of the test loss at the best
validation checkpoint:  256 draws 6.067,  1k draws 5.934,  backprop 5.959.

Learning rate and momentum are the authors' released values (lr 0.25 / mom 0.95 at 256,
0.225 / 0.95 at 1k), not re-tuned here. Seeds 42, 43, 44 as in the release.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp3
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
EXP = "smoke3" if SMOKE else "exp3"

POPS = (256,) if SMOKE else (256, 1024)
CELLS = [Cell(exp=EXP, method="dust", budget="1M", pop=pop, seed=seed, steps=3 if SMOKE else None)
         for pop in POPS for seed in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS)
