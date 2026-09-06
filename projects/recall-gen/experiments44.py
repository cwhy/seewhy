"""Recall-Gen — exp44: exp43 with the simplex mode removed from the prior.

exp43 retrieves chess positions it has never seen at 0.942, while retrieving
fresh draws from its own training prior at 0.169. That ordering is backwards and
needs an explanation before any of exp43's numbers mean anything.

The leading one: 40% of `synth`'s worlds are drawn in SIMPLEX mode — one active
coordinate per group of 13 — and that is exactly what a chess piece plane is. If
so, chess is not being generalised to; it is inside the prior's support, and
natural images are not.

This run removes that mode. `synth_cont` is identical except every world is
continuous, so nothing the prior generates looks like a board. Everything else —
world count, items per world, random masks, step budget, seed — is unchanged.

    chess transfer collapses  ->  the prior was generating chess-shaped items
    chess transfer survives   ->  the explanation is wrong and something else
                                  is carrying it

Usage:
    uv run --no-sync python projects/recall-gen/scripts/run_experiments.py --bg exp44
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
TRAIN_CLS = get("synth_cont").split[0]

RN = Run(
    exp_name="smoke_exp44" if SMOKE else "exp44",
    name="synthetic prior, continuous worlds only, 48k steps",
    domain="synth_cont",
    M=16, Q=4, mask_rows=4,
    batch=256, steps=200 if SMOKE else 48000, lr=3e-4, seed=0,
    train_mode="recall", ctx_mode="class",
    random_mask=True, mask_frac=(0.3, 0.7),
    train_digits=TRAIN_CLS, held_digits=TRAIN_CLS,
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=PAD_W),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 2000,
)

if __name__ == "__main__":
    run(RN)
