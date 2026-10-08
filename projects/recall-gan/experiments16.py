"""exp16 — relativistic loss, 12 runs each (seeds 50-61; compare exp14/15).

The game is exp10's, unchanged: context-aware generator, real = 16 context
images, fake = generated, gradient penalty 10 on the context images only (the
penalty on generated images hurt in exp14). One change, agreed with the user:
the loss is relativistic. D is trained on pairs (context image i, generated
image i) from the same episode to score the context image higher; G to make
its image outscore the paired context image. softplus(-(D(a) - D(b))), the
RpGAN loss used in R3GAN.

    tag "rel": dict(loss="relativistic")

exp16 = with the original Adam (beta1 0.5, beta2 0.999): does it fix the
instability on its own?  exp17 = with Adam beta1 0, beta2 0.99 (recipe C,
0/12 failures in exp15): does it add anything on top?

Failure, as exp14/15: class probe ends < 0.5, or generator variety ends < 0.06.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp16 exp17
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
EXP = "smoke16" if SMOKE else "exp16"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
SEEDS = (50,) if SMOKE else tuple(range(50, 62))

CONFIGS = {"rel": dict(loss="relativistic")}
MINE = ["rel"]

CELLS = [Cell(exp=EXP, arm="ctx", m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag=name, **CONFIGS[name])
         for name in MINE for s in SEEDS]

if __name__ == "__main__":
    run_cells(CELLS)
