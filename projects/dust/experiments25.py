"""exp25 — DUST at 1,024 draws under Adam and Muon, one seed.

A 1,024-draw update under these optimisers is about 40 seconds (the Adam recipe draws more for the token embedding), so a run
is about 40 minutes: one seed each, at the 256-draw winner's rate (the paper tunes every cell; this does not). The
question is whether the gap to backprop keeps closing under the better optimisers, which is where the paper's Adam section
has DUST ending below backprop.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu N exp25
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
EXP = "smoke25" if SMOKE else "exp25"
GRID = ("smoke21", "smoke22") if SMOKE else ("exp21", "exp22")

if __name__ == "__main__":
    cells = [Cell(exp=EXP, method="dust", budget="1M", pop=1024, seed=42, steps=3 if SMOKE else None,
                  **optimiser_settings("dust", 1024, opt, GRID, fallback_pop=256)) for opt in ("muon", "adam")]
    run_cells(cells)
