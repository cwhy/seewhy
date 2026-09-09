"""Recall-Gen — exp62: exp49 with the KDA paper's neural parameterisation.

Report 23 concluded that swapping the delta rule for softmax attention roughly
doubles recall quality on real data — 0.405 to 0.766 on MNIST, 0.342 to 0.743 on
Fashion-MNIST. That comparison used recall-gen's SIMPLIFIED delta rule: linear
projections, a full-rank decay, no output gate. The KDA paper
(arXiv:2510.26692 section 4) specifies more than that, and attributes real work
to the parts left out.

So the honest question is whether report 23 compared attention against KDA or
against a stripped-down version of it. This run adds back what the paper
specifies, other than the short convolution:

    low-rank decay projection, rank = head dimension
    head-wise RMSNorm before the output projection
    data-dependent output gate, itself low-rank

The short convolution is held out deliberately and tested separately in exp63.
In the paper's setting adjacent tokens are language, where a local window is
meaningful. Here one token is a WHOLE IMAGE and the context images are in random
order, so a short convolution blends unrelated items — it is not obviously
harmless, and folding it in with the rest would confound the two.

Everything else is exp49 exactly: same width, heads, depth, steps, learning rate,
seed, pool, batch and mask schedule, and no best-checkpoint selection, so the
final weights are compared against exp49's and exp56's final weights the way
report 23 compared them.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp62
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
    exp_name="smoke_exp62" if SMOKE else "exp62",
    name="synthetic prior, recall arm, KDA paper parameterisation",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W,
            kda_full=True),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
