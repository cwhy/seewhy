"""Recall-Gen — exp18: recall at M=256 with an inexhaustible training pool (A2).

The paper's mechanism claim is that recall training has two routes to a lower
loss — retrieve the target from the context, or memorise it in the weights — and
that the second is available only because the target is always an image from a
finite training pool. exp5 (M=256, same everything else) is the run where
retrieval degrades and the in-weights route takes over. If the account is right,
closing the memorisation route should change what that run does.

So: every training image is independently warped (affine + elastic, see
`core.augment`), which makes the pool effectively infinite. No image is ever
presented twice, so nothing about a specific image is worth storing in weights.
The warp is applied BEFORE the target is chosen, so target-present queries are
still exactly a context image — recall stays exactly as solvable as it was, and
the only thing removed is the pool's finiteness.

Evaluation is unchanged and un-warped, so every number is directly comparable to
exp5's. One caveat on reading it: the "seen" conditions A/C are now seen only up
to a warp, which makes them closer in kind to the novel conditions than they were.

Two outcomes, both informative:

  D_novel_absent improves       the recall objective CAN yield a transferable
  and C ~ D                     prior; memorisation was merely the cheaper route
  both near 1.0                 the objective yields nothing once the shortcut is
                                closed, and the paper's conclusion strengthens

Prediction (recorded before running): failure — but weakly held. This is the most
genuinely uncertain experiment in the plan.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp18
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
    exp_name="smoke_exp18" if SMOKE else "exp18",
    name="recall training, M=256, augmented pool (memorisation blocked)",
    M=256, Q=4, mask_rows=14,
    batch=64, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall", augment_train=True,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=260),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
