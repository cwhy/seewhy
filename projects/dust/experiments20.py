"""exp20 — DUST at 64 draws under Adam and Muon, 1M tokens.

DUST's gradient estimate fed to Adam (Table 7: matrices and head at lr, token and value embeddings 300x lr, residual scalars
100x lr, beta1 0.7, beta2 0.95; the paper uses 0.00175 at 64 draws) or to Muon on the hidden matrices. Draw streams are
re-weighted as the paper does under Adam (more embedding draws at noise 0.4). Seed 42.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu N exp20
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
EXP = "smoke20" if SMOKE else "exp20"

CELLS = [Cell(exp=EXP, method="dust", budget="1M", pop=64, seed=42, opt=o, lr=lr, tag=f"{o}_lr{lr:g}", steps=3 if SMOKE else None)
         for o, lrs in (("adam", (0.001, 0.00175, 0.003)), ("muon", (0.001, 0.003, 0.01))) for lr in lrs]

if __name__ == "__main__":
    run_cells(CELLS[:1] if SMOKE else CELLS)
