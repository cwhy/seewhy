"""exp16 — EGGROLL on the plain network at its own best setting, five seeds.

exp15's grid puts the plain network's best at sigma 3e-4, outer rate 1e-2 — a
ten-fold smaller perturbation and a three-fold smaller rate than the residual
network wants, which is what a network whose weight magnitudes differ by layer
should need from a method that perturbs every weight by the same sigma.

This is the row that belongs in the table beside the other 2x66 arms.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp16
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
SEEDS = (0,) if SMOKE else (0, 1, 2, 3, 4)
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke16" if SMOKE else "exp16"

CELLS = [
    Cell(exp=EXP, method="eggroll_plain", seed=s, hidden=(66, 66), epochs=1,
         sigma=3e-4, n_pop=512, lr=1e-2, tag="tuned",
         n_train=N_TRAIN, n_test=N_TEST)
    for s in SEEDS
]

if __name__ == "__main__":
    run_cells(CELLS)
