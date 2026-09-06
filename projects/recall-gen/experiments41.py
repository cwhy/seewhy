"""Recall-Gen — exp41: trained only on a synthetic prior, tested on real data.

The TabPFN analogue. The network never sees MNIST, Fashion-MNIST or a chess
position during training. It sees episodes drawn from `lib/domains`' synthetic
prior: a latent-factor world sampled per episode, with the latent dimension
swept so the prior spans the difficulty range the real datasets occupy.

Three things make one network scorable on all three real datasets:

  * every domain is padded to 832, so widths match;
  * the mask is RANDOMISED per training episode, so the network is not tuned to
    one dataset's particular hole — MNIST hides its bottom 392 coordinates and
    chess its queenside 416, and a fixed-mask network would meet a mask shift on
    top of the distribution shift being measured;
  * `ctx_mode="class"` makes every item in an episode come from ONE world, so
    the task is "infer this world from sixteen examples" rather than "recognise
    a world you memorised".

exp42 is the control with a single world, where the generative process can be
memorised in the weights instead of inferred from the context.

Scored by `scripts/standard_eval.py` on `synth_to_mnist`, `synth_to_fashion_mnist`
and `synth_to_chess`, which differ only in which dataset fills the B/D band.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp41
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
    exp_name="smoke_exp41" if SMOKE else "exp41",
    name="synthetic prior, many worlds, random masks",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
