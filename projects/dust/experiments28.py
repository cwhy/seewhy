"""exp28 — a rate check for DUST at 1,024 draws under Muon: 0.002 and 0.005, seed 42.

exp25 ran this population at 0.003, the winner of the 256-draw grid (0.001, 0.003, 0.01), because a 1,024-draw
update is ~45 s and a three-point grid is ~2.5 hours. A larger population has a cleaner estimate and may want a larger
step, so the 0.095-nat gap to backprop under Muon could be partly a rate that was carried over. This runs the two
nearest points on either side of 0.003. If neither beats 0.003 the carried-over rate was fine.

Usage:
    uv run --no-sync python projects/dust/scripts/gpu_queue.py exp28
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
EXP = "smoke28" if SMOKE else "exp28"
GRID = ("smoke22",) if SMOKE else ("exp22",)

if __name__ == "__main__":
    hp = optimiser_settings("dust", 1024, "muon", GRID, fallback_pop=256)
    cells = [Cell(exp=EXP, method="dust", budget="1M", pop=1024, seed=42, steps=3 if SMOKE else None,
                  **{**hp, "lr": lr, "tag": f"muon_lr{lr:g}_check"}) for lr in (0.002, 0.005)]
    run_cells(cells[:1] if SMOKE else cells)
