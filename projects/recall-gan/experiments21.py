"""exp21 — long context: the method at M = 64.

The method of the paper, unchanged except for the setting named: context-aware
generator; real = the M context images, fake = generated, 4 + 4 queries per
episode; gradient penalty 10 on the real queries; Adam lr 2e-4, beta1 0,
beta2 0.99. Seeds 50-55.

exp20: in the paper's runs (exp15, 20k steps) the generator's context use was
still rising in 12/12 runs while the discriminator's measures had levelled
off; this trains 3x longer. exp21/22: contexts of 64 and 256 images (the KDA
state stays 16,384 numbers). For M != 16, `class_acc_m` probes with M images
per digit as well as the usual 16 (evaluation only).

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp20 exp21 exp22
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
EXP = "smoke21" if SMOKE else "exp21"
STEPS = 20 if SMOKE else 20000
EVAL_EVERY = 10 if SMOKE else 500
SEEDS = (50,) if SMOKE else tuple(range(50, 56))

CONFIGS = {
    "base": dict(),
    "fakepen": dict(r2=10.0),
    "adam": dict(beta1=0.0, adam_b2=0.99),
    "both": dict(r2=10.0, beta1=0.0, adam_b2=0.99),
}
MINE = ["adam"]

CELLS = [Cell(exp=EXP, arm="ctx", m_ctx=64, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag=name, **CONFIGS[name])
         for name in MINE for s in SEEDS]

if __name__ == "__main__":
    run_cells(CELLS)
