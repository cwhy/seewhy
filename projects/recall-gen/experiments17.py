"""Recall-Gen — exp17: shrink the STATE, hold the context fixed.

The M-sweep confounds two things. Raising M both (a) overruns the memory, and
(b) puts more digits in the context, which is more information to generalise
from. If generalisation appears at large M because the context became
informative, this experiment should show nothing; if it appears because
retrieval became impossible, this experiment should reproduce it exactly.

So: M stays at 16, the model width stays at 256, and only the shape of the
memory changes. The state is H x dk x dk numbers with H x dk = 256 fixed, so

    dk=64, H=4   -> 16 384   (the default, exp1)
    dk=32, H=8   ->  8 192
    dk=16, H=16  ->  4 096
    dk=8,  H=32  ->  2 048

Here the state is **2,048 numbers against 16 x 784 = 12 544 numbers of
context content**. Parameter count is unchanged — the projections are all
256 x 256 however they are sliced into heads — so this is a memory-capacity
change and nothing else.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp17
"""

import logging
import os
import sys
from pathlib import Path

# repo root goes at the END: the GPU box has an untracked datasets.py at its
# repo root that would otherwise shadow the HuggingFace `datasets` package.
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp17" if SMOKE else "exp17",
    name="recall training, M=16, state 2048 floats (dk=8, heads=32)",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 12000, lr=3e-4, seed=0,
    train_mode="recall",
    cfg=Cfg(d_model=256, n_layers=4, dk=8, n_heads=32, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 500,
)

if __name__ == "__main__":
    run(RN)
