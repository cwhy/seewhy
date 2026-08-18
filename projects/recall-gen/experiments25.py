"""Recall-Gen — exp25: the strict reading — train ONLY the input embedding.

exp24 freezes the KDA stack but still trains the output head, which is the
practical reading of "don't train the weights". This is the strict one: only
`W_pix`, `W_msk` and `role` are trained — 0.40M of 4.03M parameters. The readout
stays a fixed random projection, so the embedding has to arrange the residual
stream to land correctly under a map it cannot change.

Worth running alongside exp24 rather than instead of it: if exp24 works and this
fails, the head was doing the work and "the embedding is enough" is too strong.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp25
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run, EMBED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp25" if SMOKE else "exp25",
    name="knn context, recall, only the input embedding trained (head frozen too)",
    M=16, Q=1, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="knn", train_only=EMBED,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
