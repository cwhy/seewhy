"""exp1 — one nudge per update, population = the last N updates. The first look, at 1M tokens (61 updates).

N = 1 is one-nudge SGD with no history. N = 4 and 16 average that many past estimates. The learning rate is searched, because
a one-nudge estimate is very noisy and its best step is much smaller than DUST's 0.25. Plain DUST with 64 draws (6.315) and
backprop (5.937) are the reference points at this token budget.

Usage:  CUDA_VISIBLE_DEVICES=N ~/.local/bin/uv run --no-sync python projects/dust-history/experiments1.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from history.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke1" if SMOKE else "exp1"

if __name__ == "__main__":
    if SMOKE:        # a 2-layer model on whatever device: checks the plumbing, not the science
        run_cells([Cell(exp=EXP, N=3, lr=0.05, steps=4, size="L2d128"), Cell(exp=EXP, N=3, lr=0.05, steps=4, size="L2d128", agg="agree", tau=0.6)])
    else:
        run_cells([Cell(exp=EXP, N=N, lr=lr) for N in (16, 4, 1) for lr in (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)])
