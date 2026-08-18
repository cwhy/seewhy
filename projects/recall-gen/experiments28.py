"""Recall-Gen — exp28: frozen mixer, knn context with 512 ranks skipped.

Report 7 claims degradation tracks whether retrieval is ACHIEVABLE rather than
which parameters are trainable. That claim currently rests on one usable run:
exp24 is the only non-degenerate low-retrieval point, since exp25 sits in the
same corner only because it never learned anything at all. Two clusters with an
n=1 cell is a correlation, not a dose-response.

`knn_offset` turns retrievability into a dial. Skipping ranks makes the context
progressively less near-duplicate, so the distractors get easier to tell apart
and retrieval gets progressively more reachable. The model-free ceiling moves
with it (`baselines_M16_r14_knn*_Q1`, soft look-up on an absent target):

    offset    0    0.552     exp24: id(B) 0.295, degradation ~0
    offset   64    0.701     exp27
    offset  512    0.886     this run — the far end, nearly i.i.d. in character

Everything else is exp24 exactly — same freeze, same 0.60M trainable parameters.

Prediction, recorded before running: id(B) rises above exp24's 0.295 and the
D-curve starts to turn back upward, with degradation somewhere between exp24's
~0 and exp26's 0.208. If degradation stays at zero while id(B) climbs, the
report's central claim is wrong.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp28
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run, EMBED_HEAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp28" if SMOKE else "exp28",
    name="frozen mixer, knn context with 512 ranks skipped (retrievability dial)",
    M=16, Q=1, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="knn", knn_offset=512, train_only=EMBED_HEAD,
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
