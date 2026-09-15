"""exp15 — give EGGROLL on the plain network the same courtesy as on the residual one.

exp13 ran EGGROLL on Forward-Forward's 784-66-66 stack with the configuration
tuned for the residual network, and it scored 0.4071 against 0.9025. Taking that
at face value would be the exact mistake this paper is about: a configuration
tuned on one architecture, transplanted to another, read as a property of the
method.

There is also a mechanism to suspect. The residual network's weights are all
unit normal by construction — the layer scales live outside them — so one global
perturbation scale sigma suits every layer. The plain network is He-initialised,
so its weight magnitudes differ by layer, and it has bias vectors, which get a
dense Gaussian perturbation rather than a low-rank one. A single sigma cannot be
right for all of them.

So: sweep sigma and the outer rate on the plain network, then rerun the best at
five seeds.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp15
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke15" if SMOKE else "exp15"

POP = 512
SIGMAS = (3e-2,) if SMOKE else (3e-4, 3e-3, 3e-2)
RATES = (3e-2,) if SMOKE else (3e-3, 1e-2, 3e-2, 1e-1)

CELLS = [
    Cell(exp=EXP, method="eggroll_plain", seed=0, hidden=(66, 66), epochs=1,
         sigma=sig, n_pop=POP, lr=lr, tag=f"s{sig:g}lr{lr:g}",
         n_train=N_TRAIN, n_test=N_TEST)
    for sig in SIGMAS for lr in RATES
]

if __name__ == "__main__":
    run_cells(CELLS)
