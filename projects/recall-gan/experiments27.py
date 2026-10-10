"""exp27 — prefix training, blind generator (newest image + one random earlier as real queries).

As exp24, but the generator does not see the context (arm "blind"), so it
cannot copy a context image. Asked by the user (2026-10-10) after report 3
found the context-aware generator copying the image just written.

Original description of exp24:

Agreed with the user (2026-10-09). The method's recipe (context-aware
generator, gradient penalty 10 on real queries, Adam lr 2e-4, beta1 0, beta2
0.99, M = 16, 20k steps), with training on every prefix of the context: after
each of the 16 writes, the discriminator answers 2 real queries (the image just
written, one random earlier context image) and 2 fakes (the "twin" — the
generator image from the previous prefix — and a fresh one from this prefix).

exp23: the first 4 context images are random (warm start); each later one is
the real training image nearest in pixel distance to the generator's image at
the current prefix, so the twin fake is queried right after its nearest real
image is written. exp24: every context image random (ablation).

Evaluation is the usual end-of-context set, plus `twin_auroc` (newest context
image vs its twin, held-out, evaluation only). Seeds 50-55.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp27 exp28
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
EXP = "smoke27" if SMOKE else "exp27"
STEPS = 20 if SMOKE else 20000
EVAL_EVERY = 10 if SMOKE else 500
SEEDS = (50,) if SMOKE else tuple(range(50, 56))

CONFIGS = {"prefix_blind": dict(beta1=0.0, adam_b2=0.99, **dict(prefix=True))}
MINE = ["prefix_blind"]

CELLS = [Cell(exp=EXP, arm="blind", m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag=name, **CONFIGS[name])
         for name in MINE for s in SEEDS]

if __name__ == "__main__":
    run_cells(CELLS)
