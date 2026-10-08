"""exp5 — the game without the real-vs-fake shortcut, gradient penalty swept.

exp4 showed the shortcut is real: with a working generator, D spent itself on
spotting generated images and called most real non-members "in context".
Here 4 of the 16 context slots hold generated images. Each training episode
asks about 3 real members, 1 generated member, 3 real non-members and 1 fresh
generated non-member, so the share of generated images is 1 in 4 on both
sides and "looks generated" carries no information about membership.

Swept: gradient penalty weight on D (0, 1, 10; exp4 used 10, the value from
real-image GAN papers, and it cost membership accuracy by itself) x
generator (blind, ctx) x seeds (42, 43, 44). 20k steps.

New metric: `auroc_genmem` — stored generated vs fresh generated images,
membership with no realness signal at all.

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
EXP = "smoke5" if SMOKE else "exp5"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500

CELLS = [Cell(exp=EXP, arm=a, m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              ctx_gen=4, q_mem=3, q_non=3, q_gen=1, r1=pen, tag=f"g4_pen{pen:g}")
         for pen in (0.0, 1.0, 10.0) for a in ("blind", "ctx") for s in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS)
