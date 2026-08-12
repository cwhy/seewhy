"""Recall-Gen — exp23: recall training on a same-class context (B1, second arm).

The other way to make the context about the query: every image in the episode is
the query's digit class. It is the cheaper construction and the one to check
exp20 against, but it is the weaker instrument for two reasons. It uses a label
MNIST happens to provide and a real setting would not, and its ceiling is worse —
0.743 on an absent target (`baselines_M16_r14_class_Q1`) against the knn
context's 0.552, because sixteen arbitrary 7s say less about the bottom of this 7
than its sixteen nearest neighbours do. That leaves it barely better than ridge's
0.649, which ignores the context altogether.

Run because the two constructions fail differently: same-class holds semantic
identity fixed while letting pixels vary, knn holds pixels close without any
guarantee about class. If recall training uses one context and not the other,
that difference says which property the mechanism keys on.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp23
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
    exp_name="smoke_exp23" if SMOKE else "exp23",
    name="recall training, same-class context (the label-using construction)",
    M=16, Q=1, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    snapshot_best="D_novel_absent",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
