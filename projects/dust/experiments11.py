"""exp11 — the 10M-token column at 256 draws / passes, one seed.

Published (Table 1, 10M): DUST 256 draws 5.231, EGGROLL 256 passes 6.906, backprop 4.989. This
is the budget where the paper says the gap to backprop is still open at small populations and
closes with population. 610 updates at ~8 s is ~80 minutes a run, so one seed each, and
EGGROLL reuses its 1M-token winner (the paper tunes per budget; this does not -- noted in the
paper's limitations). DUST uses the released 10M settings (lr 0.175 / momentum 0.95).

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp11
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
EXP = "smoke11" if SMOKE else "exp11"

if __name__ == "__main__":
    hp = eggroll_settings("1M", 256, exps=("smoke5",) if SMOKE else ("exp5", "exp8", "exp14"))
    steps = 3 if SMOKE else None
    run_cells([Cell(exp=EXP, method="dust", budget="10M", pop=256, seed=42, steps=steps),
               Cell(exp=EXP, method="eggroll", budget="10M", pop=256, seed=42, steps=steps,
                    **{**hp, "tag": "lr_from_1M"})])
