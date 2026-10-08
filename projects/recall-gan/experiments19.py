"""exp19 — final recipe, blind generator: the control for the method report.

Final recipe (recipe C, exp15): the game as proposed (real = 16 context images,
fake = generated, 4 + 4 queries per episode), gradient penalty 10 on the
context images, Adam beta1 0 / beta2 0.99 for both networks. exp15 ran it
with the context-aware generator (0/12 failures). This runs the same recipe
with the generator that does NOT see the context, same seeds 50-61, so the
report can compare the two generators under one recipe.

exp18 = seeds 50-55, exp19 = seeds 56-61 (one file per GPU).

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp18 exp19
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
EXP = "smoke19" if SMOKE else "exp19"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
SEEDS = (50,) if SMOKE else tuple(range(56, 62))

CONFIGS = {
    "base": dict(),
    "fakepen": dict(r2=10.0),
    "adam": dict(beta1=0.0, adam_b2=0.99),
    "both": dict(r2=10.0, beta1=0.0, adam_b2=0.99),
}
MINE = ["adam"]

CELLS = [Cell(exp=EXP, arm="blind", m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag=name, **CONFIGS[name])
         for name in MINE for s in SEEDS]

if __name__ == "__main__":
    run_cells(CELLS)
