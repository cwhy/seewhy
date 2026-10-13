"""exp17 — the 4,096-population rung at 1M tokens, one seed each.

Published (Table 1, 1M): DUST 4k draws 5.911, EGGROLL 4k passes 6.892, backprop 5.959. A DUST update at
4,096 draws is ~2 minutes (~2 hours a run), an EGGROLL update ~2.2 minutes; one seed each. DUST uses the
authors' released settings (lr 0.225 / momentum 0.98). EGGROLL has no tuning at this population and takes
its 256-pass winner. The rung gives EGGROLL's ladder a fourth point for the log-population extrapolation
(Appendix D of the paper fits lines through the top three populations).

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 1 exp17
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
EXP = "smoke17" if SMOKE else "exp17"

if __name__ == "__main__":
    hp = eggroll_settings("1M", 4096, exps=("smoke5",) if SMOKE else ("exp5", "exp8", "exp14"), fallback_pop=256)
    steps = 3 if SMOKE else None
    run_cells([Cell(exp=EXP, method="dust", budget="1M", pop=4096, seed=42, steps=steps),
               Cell(exp=EXP, method="eggroll", budget="1M", pop=4096, seed=42, steps=steps, **hp)][: 1 if SMOKE else None])
