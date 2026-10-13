"""exp4 — DUST at 64 draws: the learning-rate / momentum grid.

The authors released tuned optimiser settings for 256 draws and up, not for 64. The paper
tunes every cell on a 12-point grid (Appendix F, Table 5: momentum 0.9 / 0.95 / 0.98,
lr 0.1 / 0.15 / 0.2 / 0.25 at 1M tokens), seed 42 first, then three seeds on the winner.
This is the first stage at 1M tokens; the draw allocation is the paper's own 64-draw
recipe (Appendix E).

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp4
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
EXP = "smoke4" if SMOKE else "exp4"

GRID = [(lr, m) for m in (0.9, 0.95, 0.98) for lr in (0.1, 0.15, 0.2, 0.25)]
CELLS = [Cell(exp=EXP, method="dust", budget="1M", pop=64, seed=42, lr=lr, momentum=m,
              tag=f"lr{lr:g}_m{m:g}", steps=3 if SMOKE else None)
         for lr, m in (GRID[:1] if SMOKE else GRID)]

if __name__ == "__main__":
    run_cells(CELLS)
