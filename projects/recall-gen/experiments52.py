"""Recall-Gen — exp52: the completion arm with an effectively unbounded prior.

exp50 is this run with a FIXED pool of 1024 worlds. Measured on single-world
episodes it reaches 0.188 on worlds it trained on and 0.644 on worlds it has
never seen, against 0.303 for a per-episode least-squares fit that has seen no
world at all. That spread is memorisation: the network recognises worlds instead
of inferring them, which is the one thing a prior-fitted network must not do.

The cause is that the pool is finite. Over 48 000 steps at 256 episodes a run
draws 12.3M episodes from 1024 worlds, so each world is seen some twelve
thousand times. TabPFN never reuses a prior draw.

This run redraws the training pool before every block of 2000 steps, so a world
is seen at most a few dozen times and the prior is effectively unbounded.
Nothing else changes.

Two predictions, and the second is the sharper test. Performance on unseen
worlds should improve toward the in-context least-squares ceiling. And the gap
between the A/C band and the E/F band should collapse, because with the pool
resampled the "worlds seen in training" band is no longer a pool the network
has spent the run memorising.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp52
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
    exp_name="smoke_exp52" if SMOKE else "exp52",
    name="synthetic prior, completion arm, d_model=512, fresh pool every block",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="gen", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7), resample_pool=True,
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
