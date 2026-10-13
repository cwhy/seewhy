"""exp5 -- Does the curve's advantage invert with dimension?

exp4's matched-rate table has one result pointing the opposite way to all the
others. On every low-dimensional set, curves at equal address bits are 1.09-2.06x
WORSE than k-means. On gauss10d they are better -- N=3 gives 0.980+-0.001 and
N=6 gives 0.976+-0.003, on 4 of 4 seeds, improving with order.

A mechanism that would explain it: the continuity constraint costs a fraction of
the space that shrinks like 1/D. In D=2, forcing representation points to lie
along a segment removes half the available directions, which is brutal. In D=40
it removes one direction in forty, which is nearly free. The interpolation
benefit, meanwhile, has no obvious D dependence -- so the sign of the trade
should flip somewhere.

But "high dimension" is two different things and exp4 cannot separate them.
gauss10d has ambient D=10 AND intrinsic d=10. The low-dimensional sets have
small ambient dimension AND small intrinsic dimension. So the inversion could
be driven by either.

This sweeps them independently:

    gauss_{D}      ambient D, intrinsic D    -- no structure at any D
    spiral_in_{D}  ambient D, intrinsic 1    -- a 1-D curve randomly rotated
                                                into D dimensions

If the flip tracks *ambient* D, it is the constraint-cost story above, and it
predicts curves win in any high-dimensional embedding. If it tracks *intrinsic*
d, then spiral_in_40 should still favour k-means, and the gauss10d result is
about structurelessness rather than dimension.

Rate is matched by construction. With budget B = K*N and grid G, the address
count is K*G = (B/N)*G, so G = N gives exactly B addresses for every order --
the same as k-means' K = B. The `generous` mode (G = 16N) is also run, to show
the free-addressing framing alongside.

Usage:
    python projects/curve-clusters/experiments5.py
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
DIMS = (2, 3, 5, 10, 20, 40)
ORDERS = (1, 2, 3, 4, 6, 8)
GRID_MODES = ("matched", "generous")
SEEDS = tuple(range(6))
N_TRAIN, N_TEST = 2000, 1000

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"


def _standardise(train, test):
    mu = train.mean(0)
    train, test = train - mu, test - mu
    scale = np.sqrt((train ** 2).sum(1).mean())
    return train / scale, test / scale


def make_data(kind: str, dim: int, seed: int):
    """gauss: intrinsic = ambient. spiral: intrinsic 1, rotated into ambient."""
    rng = np.random.default_rng(seed)
    if kind == "gauss":
        return _standardise(rng.standard_normal((N_TRAIN, dim)),
                            rng.standard_normal((N_TEST, dim)))
    # one random orthonormal frame, shared by both splits
    q, _ = np.linalg.qr(rng.standard_normal((dim, min(2, dim))))

    def draw(n):
        t = rng.uniform(0.0, 1.0, n)
        r, a = 0.2 + 0.8 * t, 4.0 * np.pi * t
        flat = np.stack([r * np.cos(a), r * np.sin(a)], 1)[:, :q.shape[1]]
        return flat @ q.T + 0.02 * rng.standard_normal((n, dim))

    return _standardise(draw(N_TRAIN), draw(N_TEST))


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


def run_one(kind, dim, order, grid_mode, seed) -> dict:
    train, test = make_data(kind, dim, seed)
    k = max(1, BUDGET // order)
    grid = order if grid_mode == "matched" else 16 * order
    model = CurveClusters(k, order, dim, grid)
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    model.partial_fit(train[rng.permutation(len(train))])
    addr = k if order == 1 else k * grid
    return {
        "dataset": f"{kind}_{dim}d", "kind": kind, "dim": dim, "budget": BUDGET,
        "order": order, "n_curves": k, "grid": grid, "grid_mode": grid_mode,
        "n_passes": 1, "seed": seed, "addresses": addr,
        "address_bits": round(float(np.log2(addr)), 2),
        "n_params": model.n_params,
        "distortion_train": model.distortion(train),
        "distortion_test": model.distortion(test),
        "time_s": round(time.time() - t0, 3),
    }


def main() -> None:
    todo = done_names()
    for kind in ("gauss", "spiral"):
        for dim in DIMS:
            for grid_mode in GRID_MODES:
                for order in ORDERS:
                    for seed in SEEDS:
                        name = f"exp5_{kind}{dim}_N{order}_{grid_mode}_s{seed}"
                        if name in todo:
                            continue
                        row = {"experiment": name} | run_one(kind, dim, order, grid_mode, seed)
                        with RESULTS.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row) + "\n")
                print(f"{kind}_{dim}d {grid_mode:<9s} done", flush=True)


if __name__ == "__main__":
    main()
