"""exp7 — mixed context with the generated items from a FROZEN generator.

exp5's flaw: its generated context items were the trained generator's own
outputs, so a collapsed generator planted its single image in every context
and D could do no better than 0.5 on it. Here the generated context items come
from exp3's working plain-GAN generator (gradient penalty 10), frozen. Its
images are members or non-members in equal share (1 in 4 on each side). The
trained generator's images are only ever non-members and never enter a
context, so collapsing no longer pays.

Per training episode: 3 real members + 1 frozen-G member vs 2 real
non-members + 1 fresh frozen-G image + 1 trained-G image.

`auroc_genmem` is now stored vs fresh frozen-G images: membership among
generated images, independent of what the trained generator does.

Swept: penalty 0 / 1 (10 cost membership in exp6) x generator (blind, ctx)
x seeds 42-44. 20k steps.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --bg exp7
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
EXP = "smoke7" if SMOKE else "exp7"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
SOURCE = "exp3_blind_plain_m16_s42_r1"

CELLS = [Cell(exp=EXP, arm=a, m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              ctx_gen=4, q_mem=3, q_non=2, q_gen=1, r1=pen, frozen_gen=SOURCE,
              tag=f"frozen_pen{pen:g}")
         for pen in (0.0, 1.0) for a in ("blind", "ctx") for s in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS)
