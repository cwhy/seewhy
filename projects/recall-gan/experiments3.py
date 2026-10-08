"""exp3 — fix the ordinary GAN: which stabiliser stops the collapse?

exp2 showed the generator collapsing in a plain real-vs-fake GAN too, with
every output pixel's sigmoid saturated by step 10k — so exp1's failure was the
GAN setup, not the membership game. Same plain game, exp2's base rates
(Adam 2e-4 / 2e-4, beta1 0.5), one standard stabiliser at a time, then all:

    lrelu    LeakyReLU(0.2) in the discriminator instead of GELU
    r1       R1 gradient penalty on D, weight 10 (Mescheder et al. 2018)
    gln      LayerNorm after each generator hidden layer
    all      all three

Success = digit-like samples, `gen_std` well above 0, `sat` well below 1, and
`p_gen` not pinned at 0.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --bg exp3
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
EXP = "smoke3" if SMOKE else "exp3"
STEPS = 20 if SMOKE else 10_000
EVAL_EVERY = 10 if SMOKE else 500

common = dict(exp=EXP, arm="blind", disc="plain", seed=42, steps=STEPS, eval_every=EVAL_EVERY)
CELLS = [Cell(**common, d_act="lrelu", tag="lrelu"),
         Cell(**common, r1=10.0, tag="r1"),
         Cell(**common, g_ln=True, tag="gln"),
         Cell(**common, d_act="lrelu", r1=10.0, g_ln=True, tag="all")]

if __name__ == "__main__":
    run_cells(CELLS)
