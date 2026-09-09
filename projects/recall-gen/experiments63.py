"""Recall-Gen — exp63: exp62 plus the short convolution, to price it separately.

exp62 adds the KDA paper's low-rank decay, head-wise RMSNorm and output gate.
This run adds the remaining piece — a depthwise causal convolution of kernel 4 on
q, k and v, followed by Swish — and it is separated from exp62 for a reason
specific to this project.

In the paper the short convolution sits over language tokens, where a window of
four neighbours is local context. Here one token is a whole image and the context
images are drawn in random order, so the same convolution mixes an image with
three unrelated images. That could plausibly help (the query token would see the
last few context items directly, bypassing the state) or hurt (it injects noise
into every key and value). Either way it is a different intervention from the
rest of the parameterisation and deserves its own row.

The convolution is initialised as an identity tap on the current token, so the
run starts equivalent to exp62 and has to learn any window it uses.

Everything else is exp62, which is everything else is exp49.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp63
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
    exp_name="smoke_exp63" if SMOKE else "exp63",
    name="synthetic prior, recall arm, KDA paper parameterisation + short conv",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W,
            kda_full=True, conv_k=4),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
