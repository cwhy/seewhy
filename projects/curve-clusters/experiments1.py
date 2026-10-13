"""exp1 -- Is a cluster better off as a curve than as a ball?

Hypothesis
----------
Spend a fixed memory budget `B = K * N` (times D) two ways: many low-order
curves or few high-order ones. Because order N=1 is a point and N=2 is a
straight segment, the sweep contains its own baselines, and the question
becomes a single reading off one axis:

    best N = 1   balls win. Spend the budget on more clusters. Idea is dead.
    best N = 2   straight sticks win. Use local PCA, not polynomials.
    best N >= 3  bend pays for itself. The idea is worth pursuing.

Predictions, recorded before running so the result can disagree with me:
    spiral2d   best N large   (a single 1-D manifold; curves should crush balls)
    moons2d    best N >= 3
    square2d   best N moderate
    swissroll  best N >= 3
    blobs10d   best N = 1     (negative control: isotropic point clusters)
    gauss10d   best N = 1 or 2 (negative control: no low-dim structure)

If blobs10d does not prefer N=1, the metric is not discriminating and nothing
else in the table can be trusted.

Also sweeps n_passes in {1, 3}: single-pass is the regime we actually care
about, and the gap to 3 passes measures how much the one-pass restriction
costs -- i.e. whether early arbitrary assignments tangle the curves.

Usage:
    python projects/curve-clusters/experiments1.py            # full sweep
    python projects/curve-clusters/experiments1.py --selftest # N=1 == k-means
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

BUDGET = 128                                        # K * N, held fixed
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32)
PASSES = (1, 3)
SEEDS = tuple(range(8))
GRID_PER_ORDER = 16                                 # scan resolution vs order
GRID_MIN = 64

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"


def grid_for(order: int) -> int:
    return max(GRID_MIN, GRID_PER_ORDER * order)


def append_result(row: dict) -> None:
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


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


def run_one(dataset: str, order: int, n_passes: int, seed: int) -> dict:
    train, test = toy.load(dataset, seed=seed)
    k = max(1, BUDGET // order)
    model = CurveClusters(k, order, train.shape[1], grid_for(order))
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    for _ in range(n_passes):
        model.partial_fit(train[rng.permutation(len(train))])
    elapsed = time.time() - t0
    return {
        "dataset": dataset,
        "order": order,
        "n_curves": k,
        "budget": BUDGET,
        "grid": grid_for(order),
        "n_passes": n_passes,
        "seed": seed,
        "dim": int(train.shape[1]),
        "n_params": model.n_params,
        "distortion_train": model.distortion(train),
        "distortion_test": model.distortion(test),
        "time_s": round(elapsed, 3),
    }


def selftest() -> None:
    """Order 1 must be online k-means exactly, not approximately.

    Runs the reference centroid rule alongside the curve machinery on the same
    stream and requires the centres to agree to floating-point tolerance. This
    is the load-bearing check: if it passes, the N=1 column of every table is a
    genuine k-means baseline rather than a lookalike, and the sweep is anchored.
    """
    train, _ = toy.load("blobs10d", seed=0)
    k, d = 16, train.shape[1]

    model = CurveClusters(k, 1, d, 64)
    model.partial_fit(train)

    centres = train[:k].copy()
    counts = np.zeros(k)
    for x in train[k:]:
        j = int(np.argmin(((centres - x) ** 2).sum(1)))
        centres[j] += (x - centres[j]) / (1.0 + counts[j])
        counts[j] += 1

    got = model.values[:, 0, :]
    gap = float(np.abs(got - centres).max())
    same_counts = np.array_equal(model.counts, counts.astype(np.int64))
    print(f"max |curve_N1 - kmeans| = {gap:.3e}   counts match: {same_counts}")
    assert gap < 1e-9, "order-1 curves do not reproduce online k-means"
    assert same_counts, "assignment counts differ from k-means"

    ok = np.allclose(model.values, model.values[:, :1, :])
    assert ok, "order-1 curve is not constant along p"
    print("selftest passed: N=1 is exactly online k-means")


def main() -> None:
    todo = done_names()
    for dataset in toy.DATASETS:
        for order in ORDERS:
            for n_passes in PASSES:
                for seed in SEEDS:
                    name = f"exp1_{dataset}_N{order}_p{n_passes}_s{seed}"
                    if name in todo:
                        continue
                    row = {"experiment": name} | run_one(dataset, order, n_passes, seed)
                    append_result(row)
                    print(
                        f"{name:44s} K={row['n_curves']:3d} "
                        f"test={row['distortion_test']:.4f} "
                        f"({row['time_s']:.1f}s)",
                        flush=True,
                    )


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
