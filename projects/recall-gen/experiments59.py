"""Recall-Gen — exp59: the delta rule at M=64, the other cell of the 2x2.

Sixty-four context items of 832 numbers is 53,248 floats of content written into
32,768 floats of state — the first run in this project where the memory is asked
to hold more than it can. Paired with exp58 at M=16 on the same pool, and both
scored at both context lengths, this says whether more context helps a bounded
memory, does nothing, or costs it.

`n_tokens` is 68 rather than 20. It sets nothing but the decay-horizon
initialisation, and exp49's convention is M+Q, so 68 is the setting that keeps
this run consistent with the one it is being compared to rather than a second
variable.

Everything else is exp58: same width, heads, depth, steps, learning rate, seed,
pool and mask schedule.

Batch is 128 rather than exp49's 256. At M=64 the scan keeps a (batch, heads,
dk, dk) carry per token for the backward pass, and 256 asks for 21.53 GiB on a
24 GB card. Both cells of the 2x2 use 128 so the pair stays comparable; exp49,
which is only an outside check, keeps its own 256.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp59
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
TRAIN_CLS = get("synth_long").split[0]

RN = Run(
    exp_name="smoke_exp59" if SMOKE else "exp59",
    name="synthetic prior, delta rule, train M=64 (long pool)",
    domain="synth_long",
    M=64, Q=4, mask_rows=4,
    batch=128, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=68, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
    snapshot_best="B_novel_present",
)

if __name__ == "__main__":
    run(RN)
