"""exp29 — pretrained discriminator, then the method (context-aware generator).

User's idea (2026-10-10). Phase 1, 5,000 steps: the discriminator alone learns
to tell context images ("in") from real dataset images that are not in the
context ("not in"): 4 + 4 queries per 16-image episode, cross-entropy, no
gradient penalty, Adam lr 2e-4, beta1 0, beta2 0.99 (choices: Claude's).
Phase 2: the paper's method unchanged for 20k steps (real = context images,
fake = generated, gradient penalty 10 on reals), starting from the pretrained
discriminator with a fresh optimizer. Non-members are never used again.
Seeds 50-55; compare exp15 (context-aware) on the same seeds.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp29 exp30
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
EXP = "smoke29" if SMOKE else "exp29"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
SEEDS = (50,) if SMOKE else tuple(range(50, 56))

CONFIGS = {"pretrain": dict(beta1=0.0, adam_b2=0.99, pretrain_steps=20 if SMOKE else 5000)}
MINE = ["pretrain"]

CELLS = [Cell(exp=EXP, arm="ctx", m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag=name, **CONFIGS[name])
         for name in MINE for s in SEEDS]

if __name__ == "__main__":
    run_cells(CELLS)
