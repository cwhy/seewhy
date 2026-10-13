"""exp19 — backprop under Adam and Muon, 1M tokens.

The baseline half of the optimiser comparison. Adam with the paper's Table 7 group rates for backprop (token and
value embeddings 0.3, residual scalars 0.2, beta1 0.8, beta2 0.95), Muon on the hidden matrices with Adam on the rest.
Three matrix rates each, seed 42; the paper's own backprop Adam rate is 0.002.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu N exp19
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
EXP = "smoke19" if SMOKE else "exp19"

CELLS = [Cell(exp=EXP, method="bp", budget="1M", seed=42, opt=o, lr=lr, tag=f"{o}_lr{lr:g}", steps=3 if SMOKE else None)
         for o, lrs in (("adam", (0.001, 0.002, 0.004)), ("muon", (0.001, 0.003, 0.01))) for lr in lrs]

if __name__ == "__main__":
    run_cells(CELLS[:1] if SMOKE else CELLS)
