"""exp22 — the whole comparison at five epochs, each arm at its five-epoch rate.

The paper's headline is one epoch, because that is the budget the table being
replicated used and the budget the PC-ALM paper's grid uses. This asks the
obvious next question: how much of the ranking is the budget?

Every arm runs at the rate exp19-exp21 found best for *five* epochs, not the one
found best for one — six of the eight want a different rate once the budget
grows, so carrying the one-epoch rates over would have favoured whichever arm
was least sensitive to that. Every peak used here is bracketed on both sides.
Three seeds.

Usage:
    uv run python projects/local-learning/scripts/run_experiments.py --bg exp22
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
SEEDS = (0,) if SMOKE else (0, 1, 2)
EPOCHS = 2 if SMOKE else 5
N_TRAIN = 2048 if SMOKE else 60000
N_TEST = 1000 if SMOKE else 10000
EXP = "smoke22" if SMOKE else "exp22"

# best five-epoch rate per arm, from exp19 (grid), exp20 and exp21 (brackets)
RESIDUAL = {"bp": 1e-2, "pc": 1e-3, "pcalm": 1e-2, "eggroll": 3e-2}
PLAIN = [("ff", (66, 66), "", 3e-4, 3e-3),
         ("bp_plain", (66, 66), "", 1e-3, 3e-3),
         ("ff", (32,) * 31, "deep", 1e-3, 3e-3),
         ("eggroll_plain", (66, 66), "", 1e-3, 3e-4)]

CELLS = (
    [Cell(exp=EXP, method=m, seed=s, width=32, depth=32, epochs=EPOCHS, lr=lr,
          sigma=3e-3, n_pop=512, tag="5ep", n_train=N_TRAIN, n_test=N_TEST)
     for m, lr in RESIDUAL.items() for s in SEEDS]
    + [Cell(exp=EXP, method=m, seed=s, hidden=h, epochs=EPOCHS, lr=lr, sigma=sig,
            n_pop=512, tag=f"5ep{'_deep' if t else ''}",
            n_train=N_TRAIN, n_test=N_TEST)
       for m, h, t, lr, sig in PLAIN for s in SEEDS]
)

if __name__ == "__main__":
    run_cells(CELLS)
