"""exp26 — prefix training only, with RANDOM real queries.

As exp24 except the 2 real queries at each prefix are two random images
from the context so far (`prefix_real="random"`), instead of the image just
written + one random earlier one. Agreed with the user (2026-10-10): in exp23/24
the generator learned to copy the newest image's digit, because the newest
image was always a real query; this removes that guarantee.

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
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp25 exp26
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
EXP = "smoke26" if SMOKE else "exp26"
STEPS = 20 if SMOKE else 20000
EVAL_EVERY = 10 if SMOKE else 500
SEEDS = (50,) if SMOKE else tuple(range(50, 56))

CONFIGS = {"prefix_rand": dict(beta1=0.0, adam_b2=0.99, **dict(prefix=True), prefix_real="random")}
MINE = ["prefix_rand"]

CELLS = [Cell(exp=EXP, arm="ctx", m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag=name, **CONFIGS[name])
         for name in MINE for s in SEEDS]

if __name__ == "__main__":
    run_cells(CELLS)
