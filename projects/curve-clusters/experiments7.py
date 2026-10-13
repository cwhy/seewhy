"""exp7 -- K and N swept independently.

Every earlier experiment fixed the product K*N and swept N, which hides K. At
budget 128, N=1 gets K=128 and N=2 gets K=64, so "N=2 beats N=1" is really "64
sticks beat 128 points" -- the shape effect and the cluster count move together
and cannot be separated.

This sweeps them independently. The readable projection is then:

    x = stored floats K*N*D      y = held-out distortion      one line per N

K is traced along each line rather than eliminated, so a vertical slice is a
genuine equal-cost comparison and the lowest line at a given cost is the best
choice of N. Nothing is held fixed by fiat and there are no ratios.

The grid is G = 8N throughout (min 32). Grid resolution costs nothing to store,
so restricting it would only handicap high N for no reason -- and exp4 already
covers what happens when addresses are charged.

Real data is capped at a smaller K range for D=784, since cost per point goes
as K*G*D and 784-dimensional data at K=256 is minutes per cell for no extra
insight; the PCA-32 variants carry the real-data signal.

Usage:
    python projects/curve-clusters/experiments7.py
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
from experiments6 import real_data, subspace_data, _standardise

KS = (8, 16, 32, 64, 128, 256)
ORDERS = (1, 2, 3, 4, 6, 8)
SEEDS = tuple(range(4))
N_TRAIN, N_TEST = 2000, 1000

# (tag, loader, max_K) -- max_K caps cost for wide inputs
SPECS = [
    ("mnist_pca32", lambda s: real_data("mnist", 32, s), 256),
    ("fashion_mnist_pca32", lambda s: real_data("fashion_mnist", 32, s), 256),
    ("mnist_raw", lambda s: real_data("mnist", None, s), 64),
    ("sub8_in40", lambda s: subspace_data(8, s), 256),
    ("gauss_40d", lambda s: _gauss40(s), 256),
]

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"


def _gauss40(seed: int):
    rng = np.random.default_rng(seed)
    return _standardise(rng.standard_normal((N_TRAIN, 40)),
                        rng.standard_normal((N_TEST, 40)))


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


def main() -> None:
    todo = done_names()
    for tag, load, max_k in SPECS:
        for seed in SEEDS:
            train = test = None
            for k in KS:
                if k > max_k:
                    continue
                for order in ORDERS:
                    name = f"exp7_{tag}_K{k}_N{order}_s{seed}"
                    if name in todo:
                        continue
                    if train is None:
                        train, test = load(seed)
                    grid = max(32, 8 * order)
                    model = CurveClusters(k, order, train.shape[1], grid)
                    rng = np.random.default_rng(1000 + seed)
                    t0 = time.time()
                    model.partial_fit(train[rng.permutation(len(train))])
                    row = {
                        "experiment": name, "dataset": tag, "n_curves": k,
                        "order": order, "grid": grid, "grid_mode": "indep",
                        "n_passes": 1, "seed": seed, "dim": int(train.shape[1]),
                        "stored_floats": k * order * int(train.shape[1]),
                        "n_params": model.n_params,
                        "distortion_train": model.distortion(train),
                        "distortion_test": model.distortion(test),
                        "time_s": round(time.time() - t0, 3),
                    }
                    with RESULTS.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
            print(f"{tag:<22s} seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
