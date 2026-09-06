"""Recall-Gen — exp58: the delta rule at M=16, on the longer-context pool.

The M=16 cell of a 2x2 over training context length against test context length.
The delta rule writes every context item into one fixed dk x dk matrix per head,
so the interesting question for a bounded memory is what happens when there is
more to store than the memory holds. At d_model=512 with 8 heads of dk=64 the
state is 32,768 floats. Sixteen items of 832 numbers is 13,312 — comfortably
under. Sixty-four items is 53,248 — comfortably over.

This cell is exp49 in every respect except the pool. It has to be retrained
rather than reused because `synth_long` draws 96 items per world where `synth`
draws 48, and changing that count changes the random stream that builds the
worlds themselves: `synth_long`'s world 0 is not `synth`'s world 0. Reusing
exp49 here would silently make its "worlds seen in training" band novel. Both
cells of the 2x2 therefore train on the same pool, and exp49 stays as an outside
check that the M=16 result reproduces.

Batch is 128 rather than exp49's 256, because the M=64 cell cannot fit 256 and
the two cells must match.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp58
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
    exp_name="smoke_exp58" if SMOKE else "exp58",
    name="synthetic prior, delta rule, train M=16 (long pool)",
    domain="synth_long",
    M=16, Q=4, mask_rows=4,
    batch=128, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
    snapshot_best="B_novel_present",
)

if __name__ == "__main__":
    run(RN)
