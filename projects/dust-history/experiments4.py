"""exp4 — equal compute with plain DUST at 1,024 draws: 6,100 one-nudge updates on the 100M-token file.

exp2 (500 updates, the compute of DUST at 64 draws) plateaued at about 7.06 for every window up to 16: the loss of a model that
knows only token frequencies. The head learns that quickly and the interior layers never got started. DUST at 1,024 draws takes
about 30 minutes for its 61 updates (5.939); 6,100 one-nudge updates take about the same. This asks whether the interior
layers start learning given that many small steps, and whether a longer window matters then. Window 16.

Usage:  uv run --no-sync python projects/dust-history/scripts/gpu_queue.py exp4
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from history.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke4" if SMOKE else "exp4"

if __name__ == "__main__":
    run_cells([Cell(exp=EXP, N=N, lr=lr, steps=6100, budget="100M") for N, lr in [(16, 1 / 64), (16, 1 / 128), (16, 1 / 32)]])
