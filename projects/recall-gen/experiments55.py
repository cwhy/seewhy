"""Recall-Gen — exp55: recall with 2 heads of dk=256, same parameters as exp49.

Identification on worlds the network TRAINED on tops out at 0.82 and never
reaches its ceiling of 1.0. Breaking those episodes down by how far the target
sits from its nearest rival, identification is 1.000 wherever that margin is
above about 0.6 and 0.234 where it is near zero — and in exactly the failing
episodes the reconstruction is the network's best, at 0.028. It rebuilds the
item and cannot name it.

The hypothesis is the memory rather than the key. `Wk` is a per-token linear map,
as in any transformer, and from layer 2 on its input already carries the whole
episode, so keys are not in fact unable to adapt. What is specific to KDA is that
sixteen items are written into a FIXED-SIZE matrix by `S += e k^T`: two similar
keys write along nearly the same direction, the second write partly erases the
first, and a read near both returns a blend. Softmax attention keeps every item
and would not pay this.

`Wq`, `Wk`, `Wv` and `Wo` are all d_model x d_model, so trading heads against dk
at fixed width leaves the parameter count untouched and changes only the memory:

    exp49   8 x 64     32 768 state floats    64 key dims
    exp54   4 x 128    65 536                128
    exp55   2 x 256   131 072                256

If state capacity is what binds, identification tracks dk and ignores the
parameter count, which is identical across all three. If it tracks neither, the
interference hypothesis is wrong and the bottleneck is somewhere else.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp55
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
    exp_name="smoke_exp55" if SMOKE else "exp55",
    name="synthetic prior, recall arm, 2 heads x dk=256",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=256, n_heads=2, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
