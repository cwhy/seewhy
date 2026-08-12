"""Recall-Gen — exp22: mixed training on a knn context (B1).

The third arm of the triad. Half the queries have their target in the context and
half do not, so both routes are rewarded within one objective. On the i.i.d. task
this arm landed between the two pure arms and closer to recall, which was read as
retrieval being the cheaper solution whenever it is available. With a context
that is genuinely worth using, "available" is no longer all-or-nothing: the
nearest neighbour is informative even when the exact target is absent.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp22
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp22" if SMOKE else "exp22",
    name="mixed training, knn context: half the queries have their target present",
    M=16, Q=1, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="mix", p_gen=0.5, ctx_mode="knn",
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
