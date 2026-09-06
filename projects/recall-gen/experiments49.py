"""Recall-Gen — exp49: exp45 at twice the width.

The companion to exp48. exp45 underfits — small train/held gap, falling loss,
and a large gain from more steps — which implicates capacity as well as budget,
and those are separable.

exp48 holds the model and multiplies the steps. This holds the steps and doubles
the width: d_model 256 to 512, heads 4 to 8, so roughly four times the parameters
and twice the recurrent state, 32 768 floats against 16 384.

If steps alone close the gap, exp48 moves and this does not. If capacity is the
binding constraint, the reverse.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp49
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
TRAIN_CLS = get("synth").split[0]

RN = Run(
    exp_name="smoke_exp49" if SMOKE else "exp49",
    name="synthetic prior, recall arm, d_model=512",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
