"""exp9 — DUST at 64 draws, 1M tokens: three seeds at the winner of exp4's grid.

The authors released no optimiser setting for 64 draws, so `exp4` runs the paper's 12-point
momentum x rate grid on seed 42, and this repeats the best-validation point on seeds 42-44.
The paper's rule (Appendix F): if the best point sits on the edge of the grid, extend the grid one
step along its line until it is interior. Rates here stop at 0.25, so if the winner is at 0.25 this
first runs lr 0.3 and 0.35 at the winner's momentum (rows tagged exp9x), and re-selects.
Published: 6.290.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp9
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.train import Cell, run_cells
from lib.tuning import dust_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke9" if SMOKE else "exp9"
EDGE_LR = 0.25            # top of exp4's rate grid

if __name__ == "__main__":
    grid = ("smoke4",) if SMOKE else ("exp4",)
    hp = dust_settings("1M", 64, exps=grid)
    if hp["lr"] >= EDGE_LR and not SMOKE:
        logging.info("exp4's best point is on the grid edge (lr %.3g); extending: %s", hp["lr"], hp)
        run_cells([Cell(exp="exp9x", method="dust", budget="1M", pop=64, seed=42, lr=lr, momentum=hp["momentum"],
                        tag=f"lr{lr:g}_m{hp['momentum']:g}") for lr in (0.3, 0.35)])
        hp = dust_settings("1M", 64, exps=grid + ("exp9x",))
    logging.info("tuned settings for 64 draws: %s", hp)
    run_cells([Cell(exp=EXP, method="dust", budget="1M", pop=64, seed=s, tag="tuned", steps=3 if SMOKE else None, **hp)
               for s in (42, 43, 44)])
