"""Recall-Gen — exp56: exp49 with softmax attention instead of the delta rule.

The control this project has never run, and the one report 22 says is now the
question. Measured as content rather than as an index, recall inside the training
worlds is finished: recall quality 0.987, and the output sits 0.125 of the
episode's item spacing from a stored item. On real data the same checkpoint
collapses — 0.405 on MNIST, 0.342 on Fashion-MNIST, and on chess it reads 0.834
identification while sitting 0.799 from any stored position, which is the metric
saying pass while nothing in memory came back.

So the network learned to commit to stored content using weights tuned to the
synthetic prior, not a mechanism that addresses content wherever it comes from.

`kda` compresses sixteen items into one dk x dk matrix via S += e k^T. A read is
a linear combination of what was written, and two similar keys blur. Softmax
attention keeps every item and its addressing can be arbitrarily sharp, at a cost
that is irrelevant here: seventeen tokens.

If the transfer failure is the compression, this run keeps `committed` low on
MNIST, Fashion-MNIST and chess. If it fails the same way, the compression is
exonerated and the problem is the embedding or the objective, which are the next
two things to vary.

Everything else is exp49: same width, heads, depth, steps, lr, seed, episodes.
The delta rule's decay and write-strength gates have nothing to drive under
attention and are not allocated, so this run carries 13.88M parameters against
exp49's 14.95M — 7.1% fewer, in attention's disfavour. That is small next to the
gap being tested (0.405 against 0.987) but it is not nothing: if attention LOSES,
a parameter-matched variant is needed before concluding anything.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp56
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
    exp_name="smoke_exp56" if SMOKE else "exp56",
    name="synthetic prior, recall arm, softmax attention",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W,
            mixer="softmax"),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
