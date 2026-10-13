"""exp2 — the same compute as plain DUST at 64 draws, spent on many more one-nudge updates.

exp1 (equal text, 61 updates) found that the window changes nothing: N = 1, 4, 16 all land near 7.23, against 6.315 for plain
DUST at 64 draws. But a one-nudge update takes 0.27 s where a 64-draw DUST update takes 2.2 s, so DUST's whole 61-update run
(134 s) buys about 500 one-nudge updates. This runs 500 of them, on the first 500 batches of the 10M-token file (8 times the text
of exp1), and asks whether a longer window pays when there are many small steps. The window's `agree` rule is tried on the
larger windows.

Usage:  uv run --no-sync python projects/dust-history/scripts/gpu_queue.py exp2
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from history.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke2" if SMOKE else "exp2"
STEPS = 500

if __name__ == "__main__":
    cells = [Cell(exp=EXP, N=N, lr=lr, steps=STEPS, budget="10M") for N in (64, 16, 4, 1) for lr in (1 / 128, 1 / 64, 1 / 32, 1 / 16)]
    cells += [Cell(exp=EXP, N=N, lr=lr, steps=STEPS, budget="10M", agg="agree", tau=tau)
              for N in (64, 16) for tau in (0.6, 0.75) for lr in (1 / 64, 1 / 32)]
    run_cells(cells)
