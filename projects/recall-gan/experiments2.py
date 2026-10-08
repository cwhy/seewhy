"""exp2 — positive control: can this generator make digits in an ordinary GAN?

exp1's generators collapsed to one noisy blob. Before blaming the membership
game, run the same generator, optimiser and loop against a plain real-vs-fake
discriminator with no context. If it still fails, the fault is the generator,
the loop or the tuning. Logs G's gradient norm (`g_grad`) and the fraction of
output pixels whose sigmoid is saturated (`sat`, |pre-activation| > 6).

    base     exp1's settings exactly (Adam 2e-4 / 2e-4, beta1 0.5)
    lr1e-4   both rates halved
    dslow    D at 5e-5, G at 2e-4 — D held back

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --bg exp2
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
EXP = "smoke2" if SMOKE else "exp2"
STEPS = 20 if SMOKE else 10_000
EVAL_EVERY = 10 if SMOKE else 500

common = dict(exp=EXP, arm="blind", disc="plain", seed=42, steps=STEPS, eval_every=EVAL_EVERY)
CELLS = [Cell(**common),
         Cell(**common, lr_d=1e-4, lr_g=1e-4, tag="lr1e-4"),
         Cell(**common, lr_d=5e-5, lr_g=2e-4, tag="dslow")]

if __name__ == "__main__":
    run_cells(CELLS)
