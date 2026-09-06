"""Recall-Gen — exp57: exp56 with the best checkpoint on novel worlds kept.

exp56 showed that swapping the delta rule for softmax attention roughly doubles
recall quality on real datasets. Its training curve then showed something report
23 could not: error on worlds from the prior it has never seen bottoms out at
0.2464 by step 12000 and drifts up to 0.2642 by step 48000, while the training
loss halves from 0.00631 to 0.00302 over the same stretch.

So the last three quarters of exp56's training bought training-world specificity
and gave back generalisation, and every attention number in report 23 comes from
the final weights rather than the best ones.

`snapshot_best` writes `params_exp57_best.pkl` whenever the tracked condition
improves. `eval_every` is halved to 1000 so the turn is located to a thousand
steps instead of two thousand. Everything else is exp56 exactly.

The question this settles: how much of attention's remaining transfer deficit —
committed 0.578 on MNIST and 0.798 on Fashion-MNIST, against 0.079 on the worlds
it trained on — is overtraining rather than architecture.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp57
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
    exp_name="smoke_exp57" if SMOKE else "exp57",
    name="synthetic prior, softmax attention, best-novel checkpoint kept",
    domain="synth",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=PAD_W,
            mixer="softmax"),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 1000,
    # exp56's curve on worlds it has never seen bottoms out at 0.2464 by step
    # 12000 and ends at 0.2642, while the training loss halves over the same
    # stretch. The final weights are therefore not the best weights, and every
    # number report 23 quotes for attention is taken from the wrong ones. This
    # run is exp56 with the best-on-novel-worlds checkpoint kept, and the eval
    # grid doubled so the turn is located to 1000 steps rather than 2000.
    snapshot_best="B_novel_present",
)

if __name__ == "__main__":
    run(RN)
