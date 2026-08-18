"""Recall-Gen — exp24: freeze the mixer, train only the embedding.

The hypothesis this tests: the degradation we keep measuring is in-weights
overfitting, so what happens if the weights are never trained at all?

It is a sharper question than it first looks, because the KDA stack is a *fixed
algorithm* — forget, predict, correct, write — and running it needs usable keys,
not learned weights. A random projection may already produce keys good enough to
retrieve with. If so, the "similarity metric" the paper attributes to recall
training lives in the embedding, and the four KDA layers are scaffolding.

Here every KDA layer stays at its random init. Trained: the pixel and mask
embeddings, the role vectors, the final layernorm and the output head — 0.80M of
4.03M parameters, and none of it inside the sequence-processing stack. The
context is exp20's knn construction, so this is exp20 with one thing changed.

Three outcomes, all worth having:

  retrieval survives (id ~ 1.0)        the metric is in the embedding; the stack
                                       is a fixed algorithm that needed no training
  retrieval collapses                  the mixer had to learn something after all
  D stops degrading                    the U-shape was in-weights overfitting,
                                       which is the hypothesis as stated

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp24
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run, EMBED_HEAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp24" if SMOKE else "exp24",
    name="knn context, recall, KDA layers frozen at init (embedding + head only)",
    M=16, Q=1, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="knn", train_only=EMBED_HEAD,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
