"""exp3 -- Is the shrinking advantage geometry, or just data starvation?

exp2 showed the balls->sticks advantage decaying as the budget grows: on
spiral2d the N2/N1 ratio went 0.325 at B=32 to 0.690 at B=512. Read as
geometry, that says the idea does not scale.

But it cannot be read that way yet, because budget and points-per-cluster are
confounded. At B=512 with N=2 there are K=256 clusters sharing 2000 points --
eight points each. Fitting a two-parameter curve from eight points in a single
pass is estimator-noise-dominated, and that alone would erode the advantage
with no geometric content whatsoever.

Worse, the theory points the other way. For data on a 1-D manifold, K points
each cover arc ~1/K with squared error ~1/K^2, while K/2 chords each span arc
~2/K and deviate by curvature*(2/K)^2, giving squared error ~1/K^4. Sticks
should therefore beat points by MORE as the budget grows, not less. The
measured trend contradicts the prediction, which is the signature of a
finite-sample artefact rather than a geometric limit.

This holds the budget fixed and varies n_train over a decade:

    advantage recovers with more data  -> exp2's decay was starvation, and the
      geometric prediction stands
    advantage stays flat               -> the decay is real and the idea does
      not scale

Usage:
    python projects/curve-clusters/experiments3.py
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

DATASETS = ("spiral2d", "square2d", "swissroll")
N_TRAINS = (2_000, 8_000, 32_000)
BUDGETS = (128, 512)
ORDERS = (1, 2, 3)
SEEDS = tuple(range(4))
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


def run_one(dataset: str, n_train: int, budget: int, order: int, seed: int) -> dict:
    train, test = toy.load(dataset, seed=seed, n_train=n_train)
    k = max(1, budget // order)
    grid = max(GRID_MIN, GRID_PER_ORDER * order)
    model = CurveClusters(k, order, train.shape[1], grid)
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    model.partial_fit(train[rng.permutation(len(train))])
    return {
        "dataset": dataset, "n_train": n_train, "budget": budget, "order": order,
        "n_curves": k, "grid": grid, "n_passes": 1, "seed": seed,
        "points_per_curve": round(n_train / k, 1),
        "dim": int(train.shape[1]), "n_params": model.n_params,
        "distortion_train": model.distortion(train),
        "distortion_test": model.distortion(test),
        "time_s": round(time.time() - t0, 3),
    }


def main() -> None:
    todo = done_names()
    for dataset in DATASETS:
        for n_train in N_TRAINS:
            for budget in BUDGETS:
                for order in ORDERS:
                    for seed in SEEDS:
                        name = f"exp3_{dataset}_M{n_train}_B{budget}_N{order}_s{seed}"
                        if name in todo:
                            continue
                        row = {"experiment": name} | run_one(dataset, n_train, budget, order, seed)
                        with RESULTS.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row) + "\n")
                print(f"{dataset:<11s} M={n_train:<6d} B={budget:<4d} N={order} done", flush=True)


if __name__ == "__main__":
    main()
