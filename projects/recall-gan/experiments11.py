"""exp11 — exp10 (the game as proposed, context-aware generator) with the
generator's learning rate halved: 1e-4 instead of 2e-4. Nothing else changes.

Why: in exp10 seed 42 the generator collapsed suddenly at step ~11k (variety
0.19 -> 0.05 within 500 steps) after a slow drift in the D's favour, and the
discriminator stopped relying on its KDA state and never went back. A smaller
generator step is the recipe-level change aimed at that one big jump. Same
seeds as exp10 so each run pairs with its exp10 counterpart.

New indicator, evaluation only, every evaluation: `class_acc` — 10-way
accuracy of the discriminator used as a classifier with one 16-image context
per digit (fixed draws; chance 0.10).

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --bg exp11
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
EXP = "smoke11" if SMOKE else "exp11"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
ARM = "ctx"

CELLS = [Cell(exp=EXP, arm=ARM, m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag="gan_glr1e-4", lr_g=1e-4)
         for s in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS)
