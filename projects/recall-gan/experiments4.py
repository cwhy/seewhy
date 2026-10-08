"""exp4 — exp1 again, with the GAN fixed: R1 penalty on the discriminator.

exp1's generators collapsed, and exp2 showed the same collapse in a plain GAN,
so exp1 said nothing about the membership game. exp3 found the fix: R1 (weight
10) alone gives diverse, digit-like samples in the plain game, with D and G
balanced (D loss ~0.69). This is exp1's three arms with only that changed.

In the membership game R1 is taken on the queries labelled "in context" (the
members), the analogue of "real" in an ordinary GAN.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --bg exp4
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
EXP = "smoke4" if SMOKE else "exp4"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500

CELLS = [Cell(exp=EXP, arm=a, m_ctx=16, seed=42, steps=STEPS, eval_every=EVAL_EVERY, r1=10.0)
         for a in ("none", "blind", "ctx")]

if __name__ == "__main__":
    run_cells(CELLS)
