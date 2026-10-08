"""exp12 — exp10 rerun, unchanged, with one more measurement: diagnosis only.

Why: some seeds of exp10/exp11 work and some collapse. "The discriminator pulls
ahead" did not separate them (the runs that worked show it more). End-of-run
checkpoints suggest the difference is whether the GENERATOR comes to use its
context: in the working runs, swapping its context changes its output almost
as much as new noise; in the failed seed 42, about 3x less. Only measured at
the end, so this logs it at every evaluation:

    g_ctx_dep   output change when the generator's context is swapped,
                divided by the change when its noise is redrawn (evaluation
                only; 0 = ignores the context)

Same settings as exp10. Seeds 42-44 should reproduce exp10's runs; 45-47 add
examples. Split over two files (exp12, exp13) for the two GPUs.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp12 exp13
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
EXP = "smoke13" if SMOKE else "exp13"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
ARM = "ctx"

CELLS = [Cell(exp=EXP, arm=ARM, m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag="gan")
         for s in (45, 46, 47)]

if __name__ == "__main__":
    run_cells(CELLS)
