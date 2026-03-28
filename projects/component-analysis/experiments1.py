"""
exp1 — PCA of MNIST: global (all 60 000 images) + per-digit (one fit per class).

For each fit we compute the top N_COMPONENTS principal components and record:
  - explained variance ratio per component
  - cumulative EVR at k = 1, 2, 5, 10, 20, 50 components
  - reconstruction MSE on held-out test set

We also save the full results to pca_results_exp1.pkl so post-hoc visualization
scripts can reuse them without re-running PCA.

Usage:
    uv run python projects/component-analysis/scripts/run_experiments.py exp1
"""

import json, time, sys, pickle, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_1d

sys.path.insert(0, str(Path(__file__).parent))
from lib.pca import fit_pca, project, reconstruction_error
from lib.viz import (
    save_component_images,
    save_projection_2d,
    save_variance_curve,
    save_projection_grid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME    = "exp1"
N_COMPONENTS = 50
JSONL = Path(__file__).parent / "results.jsonl"


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    t0 = time.perf_counter()

    # ── Load data ─────────────────────────────────────────────────────────────
    logging.info("Loading MNIST...")
    data   = load_supervised_1d("mnist")
    X_tr   = np.array(data.X,      dtype=np.float64)   # (60000, 784)
    y_tr   = np.array(data.y,      dtype=np.int32)
    X_te   = np.array(data.X_test, dtype=np.float64)   # (10000, 784)
    y_te   = np.array(data.y_test, dtype=np.int32)
    logging.info(f"Train {X_tr.shape}, Test {X_te.shape}")

    # ── Global PCA ────────────────────────────────────────────────────────────
    logging.info(f"Fitting global PCA (n={N_COMPONENTS})...")
    pca_global = fit_pca(X_tr, N_COMPONENTS)
    recon_global = reconstruction_error(X_te, pca_global)
    logging.info(
        f"  EVR[:1]={pca_global.explained_variance_ratio[0]:.4f}"
        f"  EVR[:10]={pca_global.explained_variance_ratio[:10].sum():.4f}"
        f"  EVR[:50]={pca_global.explained_variance_ratio.sum():.4f}"
        f"  recon_mse(test)={recon_global:.2f}"
    )

    # ── Per-digit PCA ─────────────────────────────────────────────────────────
    pca_per_digit: dict[int, object] = {}
    recon_per_digit: dict[int, float] = {}
    for d in range(10):
        X_d = X_tr[y_tr == d]
        pca_per_digit[d] = fit_pca(X_d, N_COMPONENTS)
        # Eval on test images of that digit
        X_d_te = X_te[y_te == d]
        recon_per_digit[d] = reconstruction_error(X_d_te, pca_per_digit[d])
        logging.info(
            f"  digit {d}: N_tr={len(X_d)}"
            f"  EVR[:10]={pca_per_digit[d].explained_variance_ratio[:10].sum():.4f}"
            f"  recon_mse(test)={recon_per_digit[d]:.2f}"
        )

    # ── Save results for post-hoc viz ─────────────────────────────────────────
    results_path = Path(__file__).parent / f"pca_results_{EXP_NAME}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({
            "global": pca_global,
            "per_digit": pca_per_digit,
            "X_tr": X_tr, "y_tr": y_tr,
            "X_te": X_te, "y_te": y_te,
        }, f)
    logging.info(f"Saved to {results_path}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    save_component_images(pca_global.components, EXP_NAME, "global_pca",
                          n_show=20, title="Global PCA — top 20 components")
    for d in range(10):
        save_component_images(pca_per_digit[d].components, EXP_NAME, f"digit{d}_pca",
                              n_show=10, title=f"Digit {d} PCA — top 10 components")
    save_variance_curve(
        pca_global.explained_variance_ratio, EXP_NAME, "pca",
        per_class_evr={d: pca_per_digit[d].explained_variance_ratio for d in range(10)},
        title="MNIST PCA — explained variance (global=black, per-digit=colour)",
    )
    # 2D scatter using global PCA scores
    scores_te = project(X_te, pca_global)
    save_projection_2d(scores_te, y_te, EXP_NAME, "global_pca_2d",
                       xlabel="PC 1", ylabel="PC 2",
                       title="Global PCA — test set projected onto PC 1 & 2")
    # Per-digit PCA: project each digit's test samples onto that digit's PCA
    per_digit_scores = [project(X_te[y_te == d], pca_per_digit[d]) for d in range(10)]
    per_digit_labels = [np.full(len(project(X_te[y_te == d], pca_per_digit[d])), d) for d in range(10)]
    save_projection_grid(
        per_digit_scores, per_digit_labels,
        titles=[f"Digit {d} PCA" for d in range(10)],
        exp_name=EXP_NAME, tag="per_digit_pca_2d",
    )

    # ── Log result ────────────────────────────────────────────────────────────
    checkpoint_evr = [1, 2, 5, 10, 20, 50]
    elapsed = time.perf_counter() - t0
    append_result({
        "experiment": EXP_NAME,
        "name": f"PCA global + per-digit  n_components={N_COMPONENTS}",
        "global_evr_by_k": {k: round(float(pca_global.explained_variance_ratio[:k].sum()), 5)
                             for k in checkpoint_evr if k <= N_COMPONENTS},
        "global_recon_mse": round(recon_global, 4),
        "per_digit_recon_mse": {str(d): round(recon_per_digit[d], 4) for d in range(10)},
        "time_s": round(elapsed, 1),
    })
    logging.info(f"Done. {elapsed:.1f}s")
