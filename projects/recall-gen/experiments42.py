"""Recall-Gen — exp42: the single-world control for exp41.

exp41 trains on many synthetic worlds with `ctx_mode="class"`, so an episode's
sixteen items share one generative process the network has to infer from them.
This run is identical except that the prior has exactly ONE world, drawn once,
and episodes are sampled iid from it.

That removes in-context inference and leaves memorisation: the generative
process is fixed, so it can be learned into the weights rather than read off the
context. Whatever exp41 does that exp42 does not is attributable to inferring
the world rather than to having seen synthetic data.

Both keep random masks, both are padded to 832, and both are scored the same way
on `synth_to_mnist`, `synth_to_fashion_mnist` and `synth_to_chess`.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp42
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
TRAIN_CLS = get("synth1").split[0]

RN = Run(
    exp_name="smoke_exp42" if SMOKE else "exp42",
    name="synthetic prior, ONE world, random masks",
    domain="synth1",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="iid",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
