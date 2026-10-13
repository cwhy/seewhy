"""exp22 — DUST at 256 draws under Muon, 1M tokens.

As exp20 at 256 draws, Muon on the hidden matrices. Seed 42.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu N exp22
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
EXP = "smoke22" if SMOKE else "exp22"

CELLS = [Cell(exp=EXP, method="dust", budget="1M", pop=256, seed=42, opt="muon", lr=lr, tag=f"muon_lr{lr:g}", steps=3 if SMOKE else None)
         for lr in (0.001, 0.003, 0.01)]

if __name__ == "__main__":
    run_cells(CELLS[:1] if SMOKE else CELLS)
