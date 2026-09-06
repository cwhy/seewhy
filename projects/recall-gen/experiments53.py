"""Recall-Gen — exp53: the recall arm on a prior whose recall task is well posed.

exp49 identifies at 0.705 on worlds it trained on, against a ceiling of 0.999.
That reads as a network failing to learn recall, and it is not.

Breaking those episodes down by how far the target sits from its nearest rival
shows identification is 1.000 wherever that margin exceeds about 0.6, and
collapses only where it is near zero — and in exactly those episodes the network
reconstructs BEST (error 0.028). It is not failing to rebuild the item. It is
being asked to name one of sixteen near-identical items.

The cause is the prior. Its latent dimension is drawn loguniform(1, 64), so a
third of worlds have k <= 4, and sixteen draws from a low-dimensional world lie
almost on top of each other. At k=1, 42% of episodes have a nearest rival within
0.1; the median margin only reaches MNIST's 0.63 at about k=8.

This run raises the floor to k >= 8 and changes nothing else.

The cost is real and is the point of running it rather than assuming it: small-k
worlds are exactly the ones where COMPLETION is easy, because a low-dimensional
manifold is what makes a seventeenth item predictable from sixteen. Expect
identification to rise toward its ceiling and absent-target error to get worse.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp53
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
TRAIN_CLS = get("synth_k8").split[0]

RN = Run(
    exp_name="smoke_exp53" if SMOKE else "exp53",
    name="synthetic prior with latent dimension >= 8, recall arm, d_model=512",
    domain="synth_k8",
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
