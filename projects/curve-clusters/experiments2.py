"""exp2 -- Does the advantage of bend survive a bigger budget?

Motivation
----------
exp1 gave a clear shape: going from balls (N=1) to straight sticks (N=2) cuts
distortion by 35-55%, while going from sticks to bent curves (N=3) buys a
further 5-17% and N>=4 is worse. But `spiral2d` -- the most curve-shaped
dataset in the set -- was the one where N=3 did *not* reliably beat N=2 (4/8
seeds). That is the tell for a problem that would sink the whole idea:

    every smooth manifold is locally straight.

At a fixed budget K*N, raising N means fewer pieces, and each piece then spans
a *longer* stretch of the manifold, where curvature matters. Lower N means more,
shorter pieces, each spanning a stretch that is nearly flat. So bend can only
pay while pieces are long relative to the manifold's radius of curvature.

Which makes the asymptotics the real question, not the value at one budget:

    if the N=3-over-N=2 gain SHRINKS as the budget grows -> the idea is
      asymptotically worthless; sticks win in the limit that matters
    if it HOLDS -> bend is buying something structural, and the idea scales

This sweeps the budget over five octaves and reads that trend off directly.
The quantity to watch is the paired per-seed ratio N3/N2 against budget.

Usage:
    python projects/curve-clusters/experiments2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.curves import CurveClusters
from lib import toy

BUDGETS = (16, 32, 64, 128, 256, 512)
ORDERS = (1, 2, 3, 4, 6, 8)
SEEDS = tuple(range(8))
N_PASSES = 1                          # the regime we actually care about
GRID_PER_ORDER = 16
GRID_MIN = 64

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"


def done_names() -> set[str]:
    if not RESULTS.exists():
        return set()
    out = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["experiment"])
        except Exception:
            continue
    return out


def run_one(dataset: str, budget: int, order: int, seed: int) -> dict:
    train, test = toy.load(dataset, seed=seed)
    k = max(1, budget // order)
    grid = max(GRID_MIN, GRID_PER_ORDER * order)
    model = CurveClusters(k, order, train.shape[1], grid)
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    for _ in range(N_PASSES):
        model.partial_fit(train[rng.permutation(len(train))])
    return {
        "dataset": dataset, "budget": budget, "order": order, "n_curves": k,
        "grid": grid, "n_passes": N_PASSES, "seed": seed,
        "dim": int(train.shape[1]), "n_params": model.n_params,
        "distortion_train": model.distortion(train),
        "distortion_test": model.distortion(test),
        "time_s": round(time.time() - t0, 3),
    }


def main() -> None:
    todo = done_names()
    for dataset in toy.DATASETS:
        for budget in BUDGETS:
            for order in ORDERS:
                for seed in SEEDS:
                    name = f"exp2_{dataset}_B{budget}_N{order}_s{seed}"
                    if name in todo:
                        continue
                    row = {"experiment": name} | run_one(dataset, budget, order, seed)
                    with RESULTS.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
            print(f"{dataset:<11s} B={budget:<4d} done", flush=True)


if __name__ == "__main__":
    main()
