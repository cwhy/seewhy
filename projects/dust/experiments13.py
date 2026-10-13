"""exp13 — the 100K-token ladder: three seeds at each tuned winner.

Published (Table 1, 100K): DUST 7.240 / 7.165 / 7.151 at 64 / 256 / 1k draws, EGGROLL
7.961 / 7.605 / 7.386 at 64 / 256 / 1k passes, backprop 7.216. Winners are chosen by validation loss
from exp12 (DUST, EGGROLL) and exp7 (backprop), on seed 42.

The paper's rule (Appendix F): a best point on the edge of its grid gets the grid extended one step
along that line until the best point is interior. Before the three-seed repeat, each winner that sits
on an edge of its grid has the next step beyond it run (rows tagged exp13x) and the winner is re-selected.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp13
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib import tuning
from lib.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke13" if SMOKE else "exp13"

SGD_LR, SGD_MOM = (0.3, 0.6), (0.2, 0.5)          # edges of the paper's 100K grid (Table 5)
EGG_LR = (1e-3, 3e-2)                              # edges of exp12's EGGROLL rate grid


def extensions(method: str, pop: int, best: dict) -> list[Cell]:
    """The grid step beyond each edge the winner sits on."""
    base = dict(exp="exp13x", method=method, budget="100K", pop=pop, seed=42)
    cells = []
    if method in ("bp", "dust"):
        if best["momentum"] == SGD_MOM[0]:
            cells += [Cell(**base, lr=best["lr"], momentum=m, tag=f"ext_m{m:g}") for m in (0.1, 0.0)]
        if best["momentum"] == SGD_MOM[1]:
            cells += [Cell(**base, lr=best["lr"], momentum=0.7, tag="ext_m0.7")]
        if best["lr"] == SGD_LR[0]:
            cells += [Cell(**base, lr=0.2, momentum=best["momentum"], tag="ext_lr0.2")]
        if best["lr"] == SGD_LR[1]:
            cells += [Cell(**base, lr=0.8, momentum=best["momentum"], tag="ext_lr0.8")]
    else:
        kw = dict(sigma=best["sigma"], momentum=best["momentum"], beta2=best["beta2"])
        if best["lr"] == EGG_LR[0]:
            cells += [Cell(**base, lr=3e-4, tag="ext_lr0.0003", **kw)]
        if best["lr"] == EGG_LR[1]:
            cells += [Cell(**base, lr=1e-1, tag="ext_lr0.1", **kw)]
    return cells


if __name__ == "__main__":
    grid_exps = {"bp": ("smoke7",) if SMOKE else ("exp7", "exp13x"),
                 "dust": ("smoke12",) if SMOKE else ("exp12", "exp13x"),
                 "eggroll": ("smoke12",) if SMOKE else ("exp12", "exp13x")}
    pops = (256,) if SMOKE else (64, 256, 1024)
    todo = [("bp", 0)] + [(m, p) for p in pops for m in ("dust", "eggroll")]
    # 1. extend any winner that sits on a grid edge, then re-select
    ext = []
    for method, pop in todo:
        b = tuning.best_row(method, "100K", pop, grid_exps[method][:1])
        ext += extensions(method, pop, b)
    if ext and not SMOKE:
        logging.info("extending %d edge winners", len(ext))
        run_cells(ext)
    # 2. three seeds at each winner
    cells = []
    for method, pop in todo:
        b = tuning.best_row(method, "100K", pop, grid_exps[method])
        for s in (42, 43, 44):
            kw = dict(lr=b["lr"], momentum=b["momentum"])
            if method == "eggroll":
                kw.update(sigma=b["sigma"], beta2=b["beta2"])
            cells.append(Cell(exp=EXP, method=method, budget="100K", pop=pop, seed=s, tag="tuned",
                              steps=3 if SMOKE else None, **kw))
    run_cells(cells)
