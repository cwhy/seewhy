"""exp26 — the EGGROLL ladder under Muon: 64 and 256 passes at three seeds, 1,024 at one.

Stage 1 (exp23) tuned Muon's rate on seed 42. EGGROLL under Adam is already in the ladder (exp10), so this is the Muon half.
Muon on the hidden matrices at the stage-1 winner, Adam at 0.003 (beta1 0.5, beta2 0.95) on everything else, sigma 0.003.
1,024 passes takes the 256-pass winner's rate.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu N exp26
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
EXP = "smoke26" if SMOKE else "exp26"
GRID = ("smoke23",) if SMOKE else ("exp23",)

if __name__ == "__main__":
    cells = []
    for pop, seeds in ((64, (42, 43, 44)), (256, (42, 43, 44)), (1024, (42,))):
        hp = optimiser_settings("eggroll", pop, "muon", GRID, fallback_pop=256)
        cells += [Cell(exp=EXP, method="eggroll", budget="1M", pop=pop, seed=s, steps=3 if SMOKE else None, **hp) for s in seeds]
    run_cells(cells[:1] if SMOKE else cells)
