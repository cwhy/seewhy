"""exp4 -- Is the curve's advantage partly just more addresses?

The budget K*N counts the model's degrees of freedom: K*N*D stored reals. It
does NOT count addresses. A curve's address p is a real number, so a point can
be located anywhere along it to arbitrary precision -- N bounds how many wiggles
the curve can have, not how finely it can be indexed. The actual address count
is K*G, and exp1-3 used G = max(64, 16N).

That means the "matched budget" comparison quietly handed curves more address
bits than k-means: at B=128, k-means has K=128 addresses (7 bits) while N=3
curves have 42*64 = 2688 (11.4 bits). If those extra bits are doing the work,
the headline result is confounded.

This sweeps G at fixed N and reads the answer directly:

    distortion flat in G     -> extra addresses buy nothing, comparison was fair
    distortion falls with G  -> the advantage is partly refinement, and the
                                honest axis is bits, not stored reals

N=1 is the built-in control: a constant curve is the same at every p, so its
distortion must be *exactly* flat in G. Any variation there is a bug.

The MDL reading matters here. k-means needs K = 2^R centroids to use R address
bits, so its model grows exponentially in the rate. A curve spends address bits
at no model cost, until it hits the transverse error floor set by N and K. If
that asymmetry is real, it is a better argument for curves than anything in
exp1 -- and it is invisible to a sweep that holds K*N fixed.

Usage:
    python projects/curve-clusters/experiments4.py
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

BUDGET = 128
ORDERS = (1, 2, 3, 6)
GRIDS = (2, 4, 8, 16, 32, 64, 128, 256, 512)
SEEDS = tuple(range(4))

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


def run_one(dataset: str, order: int, grid: int, seed: int) -> dict:
    train, test = toy.load(dataset, seed=seed)
    k = max(1, BUDGET // order)
    model = CurveClusters(k, order, train.shape[1], grid)
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    model.partial_fit(train[rng.permutation(len(train))])
    return {
        "dataset": dataset, "budget": BUDGET, "order": order, "n_curves": k,
        "grid": grid, "n_passes": 1, "seed": seed,
        "address_bits": round(float(np.log2(k * grid)), 2),
        "dim": int(train.shape[1]), "n_params": model.n_params,
        "distortion_train": model.distortion(train),
        "distortion_test": model.distortion(test),
        "time_s": round(time.time() - t0, 3),
    }


def main() -> None:
    todo = done_names()
    for dataset in toy.DATASETS:
        for order in ORDERS:
            for grid in GRIDS:
                if grid < order:          # Phi would be rank-deficient
                    continue
                for seed in SEEDS:
                    name = f"exp4_{dataset}_N{order}_G{grid}_s{seed}"
                    if name in todo:
                        continue
                    row = {"experiment": name} | run_one(dataset, order, grid, seed)
                    with RESULTS.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
            print(f"{dataset:<13s} N={order} done", flush=True)


if __name__ == "__main__":
    main()
