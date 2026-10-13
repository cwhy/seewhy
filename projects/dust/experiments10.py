"""exp10 — the EGGROLL ladder at 1M tokens: 64, 256 and 1,024 forward passes, three seeds.

The other half of the paper's headline comparison (Table 1, 1M column; published EGGROLL
7.428 / 7.200 / 7.046). Settings are the best-validation cells of exp5 / exp8 at 64 and 256
passes. 1,024 passes costs ~40 minutes a run, so it is not tuned separately: it takes the
256-pass winner, and its tag says so.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 1 exp10
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.train import Cell, run_cells
from lib.tuning import eggroll_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke10" if SMOKE else "exp10"
GRID_EXPS = ("smoke5",) if SMOKE else ("exp5", "exp8", "exp14")

if __name__ == "__main__":
    cells = []
    for pop in ((64,) if SMOKE else (64, 256, 1024)):
        hp = eggroll_settings("1M", pop, exps=GRID_EXPS, fallback_pop=256)
        cells += [Cell(exp=EXP, method="eggroll", budget="1M", pop=pop, seed=s, steps=3 if SMOKE else None, **hp)
                  for s in (42, 43, 44)]
    run_cells(cells)
