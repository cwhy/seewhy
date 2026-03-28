"""
exp5 — Data-dimension PCA of MNIST (per digit).

Standard PCA (exp1) treats each 784-pixel image as a point in ℝ^784 and
finds the principal directions of variation in *feature space*.

Here we flip the matrix: for each digit class d we treat (X_d - mean)^T as
a 784 × N_d matrix and find the principal directions of variation in *data
(sample) space* — i.e. the left singular vectors of X_d.

Each component u_k ∈ ℝ^{N_d} is a weighting over the training images.
To visualise it we show:
  1. The pixel loading image: (X_d - mean)^T u_k ∈ ℝ^{784}  (28×28)
  2. The top-4 and bottom-4 training images ranked by their loading u_k[i].

This reveals *within-class structure*: what distinguishes, say, "thick 3s"
from "thin 3s" in each component.

Usage:
    uv run python projects/component-analysis/scripts/run_experiments.py exp5
"""

import json, time, sys, pickle, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_1d
from shared_lib.html import save_figures_page

sys.path.insert(0, str(Path(__file__).parent))
from lib.viz import save_data_pca_grid, save_variance_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME    = "exp5"
N_COMPONENTS = 20   # left singular vectors to keep per digit
JSONL = Path(__file__).parent / "results.jsonl"


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    t0 = time.perf_counter()

    # ── Load data ─────────────────────────────────────────────────────────────
    logging.info("Loading MNIST...")
    data = load_supervised_1d("mnist")
    X_tr = np.array(data.X,      dtype=np.float64)   # (60000, 784)
    y_tr = np.array(data.y,      dtype=np.int32)
    logging.info(f"Train {X_tr.shape}")

    # ── Per-digit data-dimension PCA ──────────────────────────────────────────
    results_by_digit: dict[int, dict] = {}
    evr_by_digit: dict[int, np.ndarray] = {}

    for d in range(10):
        X_d = X_tr[y_tr == d]              # (N_d, 784)
        N_d = len(X_d)
        mean_d = X_d.mean(axis=0)          # (784,)
        Xc = X_d - mean_d                  # centred

        # Economy SVD: U (N_d, K), s (K,), Vt (K, 784)  where K = min(N_d, 784)
        logging.info(f"  SVD for digit {d} (N={N_d})...")
        U, s, _ = np.linalg.svd(Xc, full_matrices=False)

        # Explained variance ratio in data space
        var = s ** 2 / (784 - 1)   # "observations" = 784 pixels
        total_var = var.sum()
        evr = var / total_var

        results_by_digit[d] = {
            "U":    U[:, :N_COMPONENTS],   # (N_d, N_COMPONENTS)
            "s":    s[:N_COMPONENTS],
            "evr":  evr[:N_COMPONENTS],
            "mean": mean_d,
            "X_d":  X_d,
        }
        evr_by_digit[d] = evr[:N_COMPONENTS]
        logging.info(
            f"    EVR[:1]={evr[0]:.4f}  EVR[:5]={evr[:5].sum():.4f}"
            f"  EVR[:10]={evr[:10].sum():.4f}  EVR[:20]={evr[:20].sum():.4f}"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = Path(__file__).parent / f"data_pca_results_{EXP_NAME}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({"per_digit": results_by_digit,
                     "X_tr": X_tr, "y_tr": y_tr}, f)
    logging.info(f"Saved to {results_path}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    figures = []

    # Cumulative EVR curves (one line per digit)
    # Build a dummy global EVR as the mean across digits for reference
    mean_evr = np.stack([evr_by_digit[d] for d in range(10)]).mean(axis=0)
    url = save_variance_curve(
        mean_evr, EXP_NAME, "data_pca",
        per_class_evr=evr_by_digit,
        title="Data-dimension PCA — explained variance per digit (mean=black)",
    )
    figures.append(("Explained variance (data-dimension PCA)", url))

    # Per-digit component grids
    for d in range(10):
        r = results_by_digit[d]
        url = save_data_pca_grid(
            r["X_d"], r["U"], r["mean"],
            EXP_NAME, f"digit{d}",
            n_show=10, n_exemplars=4,
            title=f"Digit {d} — data-dimension PCA  (top 10 components)",
        )
        figures.append((f"Digit {d} data-PCA components", url))

    page_url = save_figures_page(
        f"{EXP_NAME}_data_pca",
        f"{EXP_NAME} — Data-Dimension PCA per Digit",
        figures,
    )
    logging.info(f"Figures page → {page_url}")

    # ── Log result ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    append_result({
        "experiment": EXP_NAME,
        "name": f"Data-dimension PCA per digit  n_components={N_COMPONENTS}",
        "evr_at_10_by_digit": {
            str(d): round(float(evr_by_digit[d][:10].sum()), 5) for d in range(10)
        },
        "page_url": page_url,
        "time_s": round(elapsed, 1),
    })
    logging.info(f"Done. {elapsed:.1f}s")
