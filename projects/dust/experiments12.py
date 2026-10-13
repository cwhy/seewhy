"""exp12 — the 100K-token column: tuning grids for DUST and EGGROLL.

100K tokens is six updates, so a cell is seconds to minutes and every method can be tuned.
The paper tunes momentum x rate on a 12-point grid at this budget (Appendix F, Table 5:
momentum 0.2 / 0.3 / 0.5, lr 0.3 / 0.4 / 0.5 / 0.6), seed 42. DUST gets the 100K draw
allocation of its Appendix E (a 6 : 1.5 : 0.5 stream ratio). EGGROLL gets four Adam rates
at its 1M-token noise scale. exp13 repeats each winner on three seeds.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp12
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
EXP = "smoke12" if SMOKE else "exp12"

DUST_GRID = [(lr, m) for m in (0.2, 0.3, 0.5) for lr in (0.3, 0.4, 0.5, 0.6)]
EGG_LRS = (1e-3, 3e-3, 1e-2, 3e-2)

CELLS = []
for pop in ((256,) if SMOKE else (64, 256, 1024)):
    CELLS += [Cell(exp=EXP, method="dust", budget="100K", pop=pop, seed=42, lr=lr, momentum=m,
                   tag=f"lr{lr:g}_m{m:g}", steps=3 if SMOKE else None)
              for lr, m in (DUST_GRID[:1] if SMOKE else DUST_GRID)]
    CELLS += [Cell(exp=EXP, method="eggroll", budget="100K", pop=pop, seed=42, lr=lr, sigma=3e-3, momentum=0.9,
                   tag=f"lr{lr:g}_s0.003", steps=3 if SMOKE else None)
              for lr in (EGG_LRS[:1] if SMOKE else EGG_LRS)]

if __name__ == "__main__":
    run_cells(CELLS)
