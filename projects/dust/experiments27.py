"""exp27 — DUST at 4,096 draws under Muon, one seed.

Under SGD DUST was level with backprop at 1,024 draws and 0.047 below it at 4,096 (one seed). Under Muon, DUST at 1,024 draws
(5.466, exp25) is about 0.1 above backprop's 5.372 (exp24). Does the gap keep closing, and does DUST cross backprop at
4,096 draws? (The paper's Adam section fits a power law to its DUST ladder whose limit, 5.248, is below its backprop's 5.361;
it does not say at which population DUST crosses.) A 4,096-draw Muon update is ~3 minutes, so ~3 hours
a run; one seed, at the 256-draw winner's rate (the paper tunes every cell; this does not).

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu N exp27
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
EXP = "smoke27" if SMOKE else "exp27"
GRID = ("smoke22",) if SMOKE else ("exp22",)

if __name__ == "__main__":
    hp = optimiser_settings("dust", 4096, "muon", GRID, fallback_pop=256)
    run_cells([Cell(exp=EXP, method="dust", budget="1M", pop=4096, seed=42, steps=3 if SMOKE else None, **hp)])
