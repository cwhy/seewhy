"""exp14 — stabilising the game: GAN-recipe tricks, 12 runs each.

The game is exp10's, unchanged (context-aware generator, real = 16 context
images, fake = generated, gradient penalty 10 on the context images). At those
settings 1 of 9 runs failed (exp10/12/13); the goal is no failures. Recipe
changes only, chosen with the user:

    A  original recipe (baseline, rerun with the same code)
    B  + gradient penalty on the generated images too, same weight (R2)
    C  + Adam beta1 = 0, beta2 = 0.99, both networks
    D  B and C together

exp14 runs A and B, exp15 runs C and D (one file per GPU). Seeds 50-61.

Failure, fixed before the runs: class probe ends < 0.5, or generator variety
ends < 0.06. Collapse events (variety < 0.06 at any point after 2k steps)
counted separately.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp14 exp15
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
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
SEEDS = (50,) if SMOKE else tuple(range(50, 62))

CONFIGS = {
    "base": dict(),
    "fakepen": dict(r2=10.0),
    "adam": dict(beta1=0.0, adam_b2=0.99),
    "both": dict(r2=10.0, beta1=0.0, adam_b2=0.99),
}
MINE = [c.split(":")[1] for c in "A:base,B:fakepen".split(",")]

CELLS = [Cell(exp=EXP, arm="ctx", m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag=name, **CONFIGS[name])
         for name in MINE for s in SEEDS]

if __name__ == "__main__":
    run_cells(CELLS)
