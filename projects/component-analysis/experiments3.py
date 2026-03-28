"""
exp3 — Pairwise CCA across all 10 MNIST digit classes.

Computes standard 2-set CCA for every pair (i, j) with i < j — 45 pairs total.
For each pair the result contains:
  - directions_a, directions_b  : (n_components, 784) pixel-space canonical directions
  - correlations                : (n_components,) canonical correlations in [0, 1]

No visualization here; use gen_viz_exp3.py for that.

Saved to cca_pairwise_exp3.pkl for post-hoc analysis.

Usage:
    uv run python projects/component-analysis/scripts/run_experiments.py exp3
"""

import json, time, sys, pickle, logging
from pathlib import Path
from itertools import combinations

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_1d

sys.path.insert(0, str(Path(__file__).parent))
from lib.cca import fit_cca

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME     = "exp3"
N_COMPONENTS = 20
REG          = 1e-4
JSONL = Path(__file__).parent / "results.jsonl"


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    t0 = time.perf_counter()

    logging.info("Loading MNIST...")
    data = load_supervised_1d("mnist")
    X_tr = np.array(data.X,      dtype=np.float64)
    y_tr = np.array(data.y,      dtype=np.int32)
    X_te = np.array(data.X_test, dtype=np.float64)
    y_te = np.array(data.y_test, dtype=np.int32)

    Xs_tr = [X_tr[y_tr == d] for d in range(10)]

    # ── Fit all 45 pairwise CCAs ───────────────────────────────────────────────
    pairs: dict[tuple[int, int], object] = {}
    for i, j in combinations(range(10), 2):
        logging.info(f"  CCA ({i}, {j})  N_a={len(Xs_tr[i])}  N_b={len(Xs_tr[j])}")
        pairs[(i, j)] = fit_cca(
            Xs_tr[i], Xs_tr[j],
            digit_a=i, digit_b=j,
            n_components=N_COMPONENTS,
            reg=REG,
        )
        r = pairs[(i, j)]
        logging.info(
            f"    corr[:3] = {np.round(r.correlations[:3], 4).tolist()}"
            f"  corr_mean = {r.correlations.mean():.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    results_path = Path(__file__).parent / f"cca_pairwise_{EXP_NAME}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({
            "pairs": pairs,
            "X_tr": X_tr, "y_tr": y_tr,
            "X_te": X_te, "y_te": y_te,
        }, f)
    logging.info(f"Saved to {results_path}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    # corr_matrix[i,j] = mean canonical correlation between digit i and j
    corr_matrix = np.zeros((10, 10))
    for (i, j), r in pairs.items():
        corr_matrix[i, j] = corr_matrix[j, i] = float(r.correlations.mean())

    elapsed = time.perf_counter() - t0
    append_result({
        "experiment": EXP_NAME,
        "name": f"Pairwise CCA  n_components={N_COMPONENTS}  reg={REG}",
        "n_pairs": len(pairs),
        "mean_corr_matrix": [[round(corr_matrix[i, j], 4) for j in range(10)] for i in range(10)],
        "top1_corr_by_pair": {
            f"{i}_{j}": round(float(pairs[(i, j)].correlations[0]), 4)
            for i, j in combinations(range(10), 2)
        },
        "time_s": round(elapsed, 1),
    })
    logging.info(f"Done. {elapsed:.1f}s")
    logging.info("Mean canonical correlation matrix:")
    for i in range(10):
        logging.info("  " + "  ".join(f"{corr_matrix[i,j]:.3f}" for j in range(10)))
