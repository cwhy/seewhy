"""exp2 — backprop-trained checkpoints at 10M and 100M tokens.

The paper measures how well DUST's gradient estimate aligns with backprop's on
checkpoints trained with backprop for 10M, 100M and 1B tokens (Section 4.2, Figure 5;
Table 8 fits the 100M one). This trains one network for 100M tokens (6,100 updates) and
keeps the weights after 610 updates (10M tokens) and after all of them. 1B tokens is out
of reach here. Settings are the paper's backprop recipe for 10M (lr 0.2, momentum 0.95);
the paper does not say what it used for its 100M checkpoint.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 1 exp2
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
EXP = "smoke2" if SMOKE else "exp2"

CELLS = [Cell(exp=EXP, method="bp", budget="100M", seed=42, tag="bp100M",
              save_at=(3,) if SMOKE else (610, 6100), steps=4 if SMOKE else None)]

if __name__ == "__main__":
    run_cells(CELLS)
