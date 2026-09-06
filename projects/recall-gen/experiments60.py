"""Recall-Gen — exp60: the M=16 cell again at twice the training budget.

Report 25 concluded that a bounded memory does worse with more context, from a
2x2 trained at 48000 steps. That conclusion is only safe if the M=64 arm had
finished learning, and at 48000 neither arm had fully converged:

    exp58 (M=16)   last quarter: loss -7.6%, novel worlds -3.6%, best at the
                   final step — still descending
    exp59 (M=64)   last quarter: loss -3.9%, novel worlds +0.3%, best at 42000 —
                   flat within 0.004 for the last 14000 steps while the training
                   loss kept falling

So the M=64 arm looked converged on the metric that matters and the M=16 arm did
not, which means more training should WIDEN the gap rather than close it. This
run tests that rather than asserting it. Both cells are re-run at 96000 steps so
the comparison stays matched, with one cosine schedule over the full budget
rather than a restart on top of the old weights.

If the M=64 arm still plateaus near 0.50 on novel worlds, report 25's finding
holds as stated. If it descends, the finding has to be restated as holding at
equal training budget rather than in general.

Everything else is exp58: same width, heads, depth, learning rate, seed, pool,
batch and mask schedule.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp60
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.domains import PAD_W, get
from lib.train import Run, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
TRAIN_CLS = get("synth_long").split[0]

RN = Run(
    exp_name="smoke_exp60" if SMOKE else "exp60",
    name="synthetic prior, delta rule, train M=16 (long pool, double budget)",
    domain="synth_long",
    M=16, Q=4, mask_rows=4,
    batch=128, steps=200 if SMOKE else 96000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
    snapshot_best="B_novel_present",
)

if __name__ == "__main__":
    run(RN)
