"""exp10 — the game as originally proposed: real = context images, fake = generated.

The user's design, with nothing added. Per episode: 16 real images written into
the discriminator's KDA memory; queries are 4 context images (label "in
context") and 4 generated images (label "not in context"). No other query
types. The generator either does not see the context (blind) or does (ctx). This file: the generator that sees the context; exp9 is the blind one.
The gradient penalty (weight 10) on the "in context" queries is the GAN-recipe
fix from exp3, the same role it plays on real data in an ordinary GAN.

exp1-8 trained with extra unstored real images labelled "not in context",
which the design never asked for; this is the first run without them.

Measured, not predicted:
    acc_mem, acc_gen, auroc_gen   context images vs generated images
    *_swap                        the same, scored against another episode's
                                  context (evaluation only): does D use its memory?
    gen_std, sat, closer_gen      what the generator does
    auroc_real                    eval-only probe: context vs unseen real images

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp9 exp10
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
EXP = "smoke10" if SMOKE else "exp10"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
ARM = "ctx"

CELLS = [Cell(exp=EXP, arm=ARM, m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag="gan")
         for s in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS)
