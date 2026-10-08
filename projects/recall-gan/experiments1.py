"""exp1 — with plain labels, does a context-aware generator move toward the context?

Generated images are always labelled "not in context". If the generator learns
to use its context, its outputs drift toward context images, and once they are
closer than the discriminator can resolve, that label asks for the impossible:
D's accuracy on generated images should fall toward chance as `closer_gen`
rises above 0.5. If neither moves, the plain rule is fine and no margin is
needed. Three arms at M=16, seed 42:

    none   no generator — membership positive control
    blind  generator without the context — the control for `closer_gen`
    ctx    generator with the context in its own KDA state

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --bg exp1
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
EXP = "smoke1" if SMOKE else "exp1"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500

CELLS = [Cell(exp=EXP, arm=a, m_ctx=16, seed=42, steps=STEPS, eval_every=EVAL_EVERY)
         for a in ("none", "blind", "ctx")]

if __name__ == "__main__":
    run_cells(CELLS)
