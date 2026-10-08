"""exp6 — membership with no generator, gradient penalty swept.

The baseline for exp5. exp4 found the penalty (weight 10) lowers membership
AUROC from 1.000 to 0.981 with no generator at all. Is that a smooth cost of
the weight, and how much does weight 1 cost? All-real context, M=16, penalty
0 / 1 / 10, seeds 42-44, 20k steps.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp5 exp6
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
EXP = "smoke6" if SMOKE else "exp6"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500

CELLS = [Cell(exp=EXP, arm="none", m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              r1=pen, tag=f"pen{pen:g}")
         for pen in (0.0, 1.0, 10.0) for s in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS)
