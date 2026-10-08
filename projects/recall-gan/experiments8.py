"""exp8 — exp7 with the trained generator warm-started from the frozen one.

exp7: D became a real membership test (stored vs fresh frozen-generator
images, AUROC 1.000), but the trained generator collapsed in 11 of 12 runs.
Its images were the only ones never in a context, so D could reject them by
their look, and without a strong gradient penalty G could not follow D's slope
toward the right look.

Here the trained generator starts AS the frozen generator (same weights; the
`ctx` arm's context-reading weights start at zero). At step 0 its images are
indistinguishable in style from the frozen generator's, which are members a
quarter of the time, so D cannot reject them by look and G starts with the
style it needs. What it must then find is images D wrongly calls members.

Same grid as exp7: penalty 0 / 1 x generator (blind, ctx) x seeds 42-44.
Generator rate also halved (1e-4) so the warm start is not immediately undone.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --bg exp8
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
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
SOURCE = "exp3_blind_plain_m16_s42_r1"

CELLS = [Cell(exp=EXP, arm=a, m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              ctx_gen=4, q_mem=3, q_non=2, q_gen=1, r1=pen, frozen_gen=SOURCE,
              init_gen_from=SOURCE, lr_g=1e-4, tag=f"warm_pen{pen:g}")
         for pen in (0.0, 1.0) for a in ("blind", "ctx") for s in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS)
