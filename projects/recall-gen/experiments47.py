"""Recall-Gen — exp47: the frozen arm: recall training with the four mixing layers held at init.

exp47 is the FROZEN arm. It trains like exp45 but its four KDA layers keep
their random initialisation forever; only the input embedding and the output
head learn, 0.64M of 4.06M numbers.

Across reports 12 to 15 this arm has been the one that trades retrieval for
prediction. Whether that trade exists when the training data is synthetic is the
question it is here to answer.

All three arms share one prior, one sampler, one seed and one step budget, so a
difference between them is the training objective and nothing else.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp47
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.domains import PAD_W, get
from lib.train import Run, run, EMBED_HEAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
TRAIN_CLS = get("synth").split[0]

RN = Run(
    exp_name="smoke_exp47" if SMOKE else "exp47",
    name="synthetic prior, the frozen arm: recall training with the four mixing layers held at init",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    train_only=EMBED_HEAD,
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
