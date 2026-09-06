"""Recall-Gen — exp40: train on nearest-neighbour contexts, Fashion-MNIST to MNIST.

exp36 is the same run with `ctx_mode="iid"`. Everything else here is identical to
it — same domain, same split, same objective, same Q — so the pair isolates one
thing: what the sixteen context images WERE during training.

    iid   sixteen unrelated images. Report 12's baselines showed such a context
          carries almost nothing about a seventeenth, so an absent-target
          episode has nothing to use and no objective can reward using it.
    knn   the query's own sixteen nearest neighbours by visible half. Now the
          context genuinely predicts the answer, and reading it pays.

Q stays at 4, matching exp36. A knn context gives each query M/Q neighbours, so
Q changes what the context IS and not merely how many queries score on it; the
Q=1 knn figures elsewhere in this project are a different context and are not
comparable to these.

The question is whether a network trained on informative contexts learns to read
them, and what that costs it when the context stops being informative. Evaluated
both ways in `scripts/gen_report_15.py`.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp40
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
    exp_name="smoke_exp40" if SMOKE else "exp40",
    name="cross-dataset: Fashion-MNIST to MNIST, recall training on knn contexts",
    domain="fashion_to_mnist",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="knn",
    train_digits=PAIRED_SPLIT[0], held_digits=PAIRED_SPLIT[1],
    conditions=SPLIT_CONDITIONS,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=784),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
