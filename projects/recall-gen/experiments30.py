"""Recall-Gen — exp30: Fashion-MNIST class split, recall training

Report 12 found, on MNIST, that retrieval crosses a class boundary intact while
prediction collapses to the average-image score. Whether that is a fact about
the task or a fact about MNIST is unanswerable from MNIST alone.

Fashion-MNIST is the cheapest way to ask. Same 28x28 grey images, same bottom-
half mask, same six conditions — only the pictures change. Its class split is
also WIDER than MNIST's: classes 0-4 are t-shirt, trouser, pullover, dress and
coat; 5-9 are sandal, shirt, sneaker, bag and ankle boot. Four of the five
novel classes are not garments at all.

If the MNIST result is about the task, retrieval on unseen footwear should be
as exact as retrieval on seen coats, and completion of unseen footwear should
sit at 1.0.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp30
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run
from experiments8 import SPLIT_CONDITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp30" if SMOKE else "exp30",
    name="class split: train on Fashion-MNIST 0-4, novel pool is 5-9, recall training",
    domain="fashion_mnist",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall",
    train_digits=(0, 1, 2, 3, 4), held_digits=(5, 6, 7, 8, 9),
    conditions=SPLIT_CONDITIONS,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=784),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
