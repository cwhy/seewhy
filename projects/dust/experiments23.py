"""exp23 — EGGROLL under Muon, 1M tokens, 64 and 256 forward passes.

EGGROLL's estimate is a sum of rank-one matrices, so for P pairs below the matrix size it has rank at most P and a very uneven
spectrum; Muon flattens the spectrum. Hidden matrices take Muon at the rate in the grid, everything else Adam at 0.003 with beta1 0.5,
beta2 0.95 (the paper's EGGROLL Adam setting, which exp8 and exp10 already ran). sigma 0.003. Seed 42.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu N exp23
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
EXP = "smoke23" if SMOKE else "exp23"

CELLS = [Cell(exp=EXP, method="eggroll", budget="1M", pop=pop, seed=42, opt="muon", lr=lr, sigma=3e-3, tag=f"muon_lr{lr:g}",
              steps=3 if SMOKE else None)
         for pop in (64, 256) for lr in (0.001, 0.003, 0.01)]

if __name__ == "__main__":
    run_cells(CELLS[:1] if SMOKE else CELLS)
