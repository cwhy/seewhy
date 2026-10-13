"""exp6 -- Real data, and intrinsic dimension swept at fixed ambient dimension.

Why exp5's high-dimensional result does not count
-------------------------------------------------
exp5 found curves beating k-means at matched rate on `gauss_40d`, 6/6 seeds.
But an isotropic Gaussian in 40-D has no structure to exploit: k-means scores
0.9395 there and curves score 0.9229, so both fail and one fails marginally
less. That is not evidence for anything. exp5's other family went to the
opposite extreme, intrinsic d=1. Neither resembles real data, which is high
ambient dimension with LOW intrinsic dimension.

Two things this runs
--------------------
1. `sub{d}_in40` -- a d-dimensional Gaussian subspace randomly rotated into
   ambient D=40, sweeping intrinsic d over 1..40 with the ambient dimension
   pinned. exp5 concluded the curve-vs-k-means flip tracks intrinsic dimension;
   if that is right, the crossing should land near d≈20 here regardless of the
   ambient dimension being fixed at 40.

2. MNIST and Fashion-MNIST, raw 784-D pixels and PCA-32 projections. This is
   the actual target: intrinsic dimension is estimated at ~12-14 for MNIST
   against 784 ambient, which is the regime the synthetic families bracket but
   never occupy. PCA-32 is included because raw pixels put most of their
   variance in a few directions, and the two can disagree.

Both rate framings are run, since exp4/exp5 showed they trend in opposite
directions with dimension and a single framing would be cherry-picking:

    matched   G = N, so K*G = budget for every order -- addresses charged
    generous  G = 16N -- addresses free

Prediction: at MNIST's intrinsic dimension the matched-rate ratio sits near
1.0, i.e. nothing to gain, since exp5's gauss family crossed 1.0 around d=10-20.

Usage:
    python projects/curve-clusters/experiments6.py
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

BUDGET = 128
INTRINSIC = (1, 2, 4, 8, 16, 32, 40)
AMBIENT = 40
ORDERS = (1, 2, 3, 4, 6, 8)
GRID_MODES = ("matched", "generous")
SEEDS = tuple(range(6))
REAL_SEEDS = tuple(range(3))
N_TRAIN, N_TEST = 2000, 1000

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"


def _standardise(train, test):
    mu = train.mean(0)
    train, test = train - mu, test - mu
    scale = np.sqrt((train ** 2).sum(1).mean())
    return train / scale, test / scale


def subspace_data(d_intrinsic: int, seed: int):
    """d-dimensional Gaussian in a random d-subspace of R^AMBIENT."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((AMBIENT, d_intrinsic)))
    draw = lambda n: rng.standard_normal((n, d_intrinsic)) @ q.T
    return _standardise(draw(N_TRAIN), draw(N_TEST))


_REAL_CACHE: dict = {}


def real_data(name: str, n_pca: int | None, seed: int):
    """MNIST / Fashion-MNIST, optionally projected onto its top PCA directions.

    The PCA basis is fitted on the train split only and applied to both, so the
    test split never informs the projection.
    """
    from shared_lib.datasets import load_supervised_1d

    key = (name,)
    if key not in _REAL_CACHE:
        ds = load_supervised_1d(name)
        _REAL_CACHE[key] = (np.asarray(ds.X, dtype=np.float64),
                            np.asarray(ds.X_test, dtype=np.float64))
    X, Xt = _REAL_CACHE[key]
    rng = np.random.default_rng(seed)
    tr = X[rng.choice(len(X), N_TRAIN, replace=False)]
    te = Xt[rng.choice(len(Xt), N_TEST, replace=False)]
    if n_pca is not None:
        mu = tr.mean(0)
        _, _, vt = np.linalg.svd(tr - mu, full_matrices=False)
        basis = vt[:n_pca].T
        tr, te = (tr - mu) @ basis, (te - mu) @ basis
    return _standardise(tr, te)


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


def evaluate(train, test, tag, meta, order, grid_mode, seed) -> dict:
    k = max(1, BUDGET // order)
    grid = order if grid_mode == "matched" else 16 * order
    model = CurveClusters(k, order, train.shape[1], grid)
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    model.partial_fit(train[rng.permutation(len(train))])
    addr = k if order == 1 else k * grid
    return {
        "dataset": tag, "order": order, "n_curves": k, "budget": BUDGET,
        "grid": grid, "grid_mode": grid_mode, "n_passes": 1, "seed": seed,
        "dim": int(train.shape[1]), "addresses": addr,
        "address_bits": round(float(np.log2(addr)), 2),
        "n_params": model.n_params,
        "distortion_train": model.distortion(train),
        "distortion_test": model.distortion(test),
        "time_s": round(time.time() - t0, 3),
        **meta,
    }


def main() -> None:
    todo = done_names()

    for d in INTRINSIC:
        for grid_mode in GRID_MODES:
            for order in ORDERS:
                for seed in SEEDS:
                    name = f"exp6_sub{d}in{AMBIENT}_N{order}_{grid_mode}_s{seed}"
                    if name in todo:
                        continue
                    tr, te = subspace_data(d, seed)
                    row = {"experiment": name} | evaluate(
                        tr, te, f"sub{d}_in{AMBIENT}",
                        {"kind": "subspace", "intrinsic": d, "ambient": AMBIENT},
                        order, grid_mode, seed)
                    with RESULTS.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
            print(f"sub{d}_in{AMBIENT} {grid_mode:<9s} done", flush=True)

    for real, n_pca in [("mnist", None), ("mnist", 32),
                        ("fashion_mnist", None), ("fashion_mnist", 32)]:
        tag = f"{real}" + (f"_pca{n_pca}" if n_pca else "_raw")
        for grid_mode in GRID_MODES:
            for order in ORDERS:
                for seed in REAL_SEEDS:
                    name = f"exp6_{tag}_N{order}_{grid_mode}_s{seed}"
                    if name in todo:
                        continue
                    tr, te = real_data(real, n_pca, seed)
                    row = {"experiment": name} | evaluate(
                        tr, te, tag,
                        {"kind": "real", "intrinsic": None,
                         "ambient": int(tr.shape[1])},
                        order, grid_mode, seed)
                    with RESULTS.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
            print(f"{tag} {grid_mode:<9s} done", flush=True)


if __name__ == "__main__":
    main()
