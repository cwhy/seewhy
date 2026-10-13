"""exp24 — backprop and DUST at 64 and 256 under Adam and Muon: three seeds at each stage-1 winner.

Stage 1 (exp19-23) tuned the matrix learning rate of each (method, optimiser, population) on seed 42 by validation loss.
This repeats each winner on seeds 42, 43 and 44, as the SGD ladder was. 1,024 draws is exp25.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu N exp24
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.train import Cell, run_cells
from lib.tuning import optimiser_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke24" if SMOKE else "exp24"
GRID = ("smoke19", "smoke20", "smoke21", "smoke22") if SMOKE else ("exp19", "exp20", "exp21", "exp22")

if __name__ == "__main__":
    cells = []
    for opt in ("adam", "muon"):
        for method, pops in (("bp", (0,)), ("dust", (64, 256))):
            for pop in pops:
                hp = optimiser_settings(method, pop, opt, GRID)
                cells += [Cell(exp=EXP, method=method, budget="1M", pop=pop, seed=s, steps=3 if SMOKE else None, **hp) for s in (42, 43, 44)]
    run_cells(cells)
