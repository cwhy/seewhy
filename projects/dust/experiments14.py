"""exp14 — extend EGGROLL's noise-scale grid downward.

In exp5, sigma = 3e-3 beat 1e-2 at every learning rate tried (64 passes, lr 3e-3: 7.815 vs 7.925),
and 3e-3 is the lowest value on that grid. The paper's tuning protocol (Appendix F) extends a grid
one step along its line whenever the best point sits on the edge, until it is interior. So:
sigma = 1e-3 and 3e-4, at the paper's Adam constants (beta1 0.5, beta2 0.95), which exp8 found to
be at least as good as ours, lr 3e-3. 256 passes gets the next step only, 1e-3.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 1 exp14
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
EXP = "smoke14" if SMOKE else "exp14"

CELLS = [Cell(exp=EXP, method="eggroll", budget="1M", pop=pop, seed=42, lr=3e-3, sigma=s, momentum=0.5, beta2=0.95,
              tag=f"paperAdam_lr0.003_s{s:g}", steps=3 if SMOKE else None)
         for pop, sigmas in ((64, (1e-3, 3e-4)), (256, (1e-3,))) for s in ((sigmas[0],) if SMOKE else sigmas)][: 1 if SMOKE else None]

if __name__ == "__main__":
    run_cells(CELLS)
