"""Recall-Gen — exp1: does a model trained ONLY to recall generalise?

Each MNIST image is one token. An episode is M context images followed by Q
masked query images; the model must fill in the hidden bottom half. A KDA linear
RNN carries the context in a matrix-valued state; context tokens write, query
tokens only read, so that state is the sole channel between them.

Training is condition A only — the query image is always one of the M context
images. Pure look-up, nothing else. We then score four conditions:

    A  context seen in training,  target IS in context   <- trained on this
    B  context never seen,        target IS in context
    C  context seen in training,  target NOT in context
    D  context never seen,        target NOT in context

A->B asks whether retrieval is content-addressed or memorised. A->C/D asks what
a recall-only model does when there is nothing to recall: fall back to the
dataset prior, copy the nearest context image, or something better than both.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp1
"""

import logging
import os
import sys
from pathlib import Path

# repo root goes at the END: the GPU box has an untracked datasets.py at its
# repo root that would otherwise shadow the HuggingFace `datasets` package.
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp1" if SMOKE else "exp1",
    name="pure-recall training, 2x2 novelty x presence eval",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
