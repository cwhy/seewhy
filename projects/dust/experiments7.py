"""exp7 — is our backprop baseline tuned as well as the paper's?

The paper tunes backprop "on the same grid at every budget" (Appendix F, Table 5) and then
reads DUST against it, so a baseline tuned worse than theirs would flatter DUST. 100K has no
released setting at all; 1M and 10M have the authors' defaults, whose neighbourhood is checked
here. Seed 42 only; the winner is then repeated on three seeds.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 1 exp7
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
EXP = "smoke7" if SMOKE else "exp7"

GRIDS = {
    "100K": [(lr, m) for m in (0.2, 0.3, 0.5) for lr in (0.3, 0.4, 0.5, 0.6)],        # Table 5, 100k row
    "1M":   [(lr, m) for m in (0.95, 0.98) for lr in (0.125, 0.175, 0.25)],            # around 0.175 / 0.98
    "10M":  [(lr, m) for m in (0.95, 0.98) for lr in (0.15, 0.2, 0.25)],               # around 0.2 / 0.95
}
CELLS = [Cell(exp=EXP, method="bp", budget=b, seed=42, lr=lr, momentum=m, tag=f"lr{lr:g}_m{m:g}",
              steps=3 if SMOKE else None)
         for b, grid in GRIDS.items() for lr, m in (grid[:1] if SMOKE else grid)]

if __name__ == "__main__":
    run_cells(CELLS)
