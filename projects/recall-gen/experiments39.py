"""Recall-Gen — exp39: train on MNIST, test on Fashion-MNIST, recall training

The reverse direction, and the reason report 15 can say anything causal.

exp36 trains on Fashion-MNIST and tests on MNIST. If retrieval survives that
crossing, the honest question is whether it survives BECAUSE the shift is
crossable or because Fashion-MNIST happens to be the richer thing to have
trained on. Running the same arm the other way separates those.

Recall training only. The completion and frozen arms are not repeated here
because this run exists to be compared against exp36, not to be characterised in
its own right.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp39
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
    exp_name="smoke_exp39" if SMOKE else "exp39",
    name="cross-dataset control: train on MNIST, novel pool is Fashion-MNIST, recall training",
    domain="mnist_to_fashion",
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
