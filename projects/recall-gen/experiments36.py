"""Recall-Gen — exp36: train on Fashion-MNIST, test on MNIST, recall training

Every novelty axis this project has run so far is a class held out of one
dataset. Digits 0-4 against 5-9, garments against footwear, middlegames against
endgames. In all three the novel pool still comes from the same source, the same
preprocessing, the same statistics.

This holds out the dataset instead. Training is Fashion-MNIST, all ten classes.
The novel pool is MNIST. Both are 28x28 grey images, so one token means the same
thing on each side and a single network can be scored on both without changing a
weight.

The three bands are therefore:

    A/C  Fashion-MNIST train split     images seen in training
    E/F  Fashion-MNIST test split      new images, same world
    B/D  MNIST test split              a different world entirely

E/F is the control that matters. If retrieval only degrades on B, the cause is
the distribution shift and not image novelty.

Two things make the outcome genuinely uncertain rather than obvious. MNIST
digits are sparser and more distinct from each other than Fashion classes are,
so retrieval among sixteen digits could be EASIER than among sixteen garments
even though the network never saw one. And MNIST is mostly black, so a network
carrying a Fashion prior — filled silhouettes — should complete it badly, while
the do-nothing reference it is scored against is the average FASHION image,
which is a poor constant for MNIST and therefore easy to beat.

Both effects are measured rather than assumed; see the baselines row, which
carries predict-black and the ridge map alongside the average image.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp36
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.domains import PAIRED_SPLIT
from lib.train import Run, run
from experiments8 import SPLIT_CONDITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp36" if SMOKE else "exp36",
    name="cross-dataset: train on Fashion-MNIST, novel pool is MNIST, recall training",
    domain="fashion_to_mnist",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall",
    train_digits=PAIRED_SPLIT[0], held_digits=PAIRED_SPLIT[1],
    conditions=SPLIT_CONDITIONS,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=784),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
