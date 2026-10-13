"""exp8 — does the paper's Adam setting make EGGROLL a stronger baseline than ours?

`local-learning`'s EGGROLL uses Adam at optax's defaults (beta1 0.9, beta2 0.999). The DUST
paper's tuned EGGROLL-Transformer under Adam uses lr 0.003, beta1 0.5, beta2 0.95 (Appendix F,
Table 7), with z-scored loss differences. z-scoring only rescales the whole estimate, which
Adam divides out, so the two things that can matter are the betas. If they help, the EGGROLL
arm of the ladder should use them: a baseline weaker than the one the paper tuned would make
DUST look better than it is.

Compared at the same lr and sigma as exp5's best cell, 1M tokens, seed 42, populations 64 and 256.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 1 exp8
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
EXP = "smoke8" if SMOKE else "exp8"

SIGMA = 3e-3     # set from exp5 once its sigma = 1e-2 cells are in
CELLS = [Cell(exp=EXP, method="eggroll", budget="1M", pop=pop, seed=42, lr=lr, sigma=SIGMA,
              momentum=0.5, beta2=0.95, tag=f"paperAdam_lr{lr:g}_s{SIGMA:g}", steps=3 if SMOKE else None)
         for pop in ((64,) if SMOKE else (64, 256)) for lr in ((3e-3,) if SMOKE else (1e-3, 3e-3))]

if __name__ == "__main__":
    run_cells(CELLS)
