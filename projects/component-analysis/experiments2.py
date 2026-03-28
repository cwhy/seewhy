"""
exp2 — Multiset CCA over 10 MNIST digit datasets.

Treats each digit class as a separate dataset and fits pooled-whitening MAXVAR MCCA:
the resulting canonical directions capture structure that is systematically common
across all digit classes, after removing within-class variance.

Compare to global PCA (exp1): PCA captures directions of maximum overall variance;
MCCA captures directions of maximum *shared* variance across classes.

Metrics logged:
  - eigenvalues (squared singular values from the stacked-whitened SVD)
  - mean pairwise canonical correlation per component
  - MCCA reconstruction MSE on test set (using per-digit canonical projection)

Saves full results to mcca_results_exp2.pkl for post-hoc visualization.

Usage:
    uv run python projects/component-analysis/scripts/run_experiments.py exp2
"""

import json, time, sys, pickle, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_1d

sys.path.insert(0, str(Path(__file__).parent))
from lib.pca import fit_pca, project as pca_project
from lib.cca import fit_mcca, project_class, project_all, canonical_correlations
from lib.viz import (
    save_component_images,
    save_component_comparison,
    save_projection_2d,
    save_projection_grid,
    save_eigenvalue_curve,
    save_canonical_correlation_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME     = "exp2"
N_COMPONENTS = 20
REG          = 1e-4
JSONL = Path(__file__).parent / "results.jsonl"


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    t0 = time.perf_counter()

    # ── Load data ─────────────────────────────────────────────────────────────
    logging.info("Loading MNIST...")
    data = load_supervised_1d("mnist")
    X_tr = np.array(data.X,      dtype=np.float64)
    y_tr = np.array(data.y,      dtype=np.int32)
    X_te = np.array(data.X_test, dtype=np.float64)
    y_te = np.array(data.y_test, dtype=np.int32)

    # ── Split training set by digit ───────────────────────────────────────────
    Xs_tr = [X_tr[y_tr == d] for d in range(10)]
    Xs_te = [X_te[y_te == d] for d in range(10)]
    for d, X in enumerate(Xs_tr):
        logging.info(f"  digit {d}: N_tr={len(X)}")

    # ── Fit MCCA ──────────────────────────────────────────────────────────────
    logging.info(f"Fitting MCCA (n_components={N_COMPONENTS}, reg={REG})...")
    mcca = fit_mcca(Xs_tr, n_components=N_COMPONENTS, reg=REG, seed=0)
    logging.info(f"  Top eigenvalues: {mcca.eigenvalues[:5].tolist()}")

    # Canonical correlations on training projections
    scores_tr = project_all(Xs_tr, mcca)
    mean_corrs = canonical_correlations(scores_tr)
    logging.info(f"  Mean pairwise corr per component: {np.round(mean_corrs, 3).tolist()}")

    # ── Also fit global PCA (for visual comparison) ───────────────────────────
    logging.info("Fitting global PCA for comparison...")
    pca_global = fit_pca(X_tr, N_COMPONENTS)

    # ── Save full results ─────────────────────────────────────────────────────
    results_path = Path(__file__).parent / f"mcca_results_{EXP_NAME}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({
            "mcca": mcca,
            "pca_global": pca_global,
            "Xs_tr": Xs_tr, "Xs_te": Xs_te,
            "X_tr": X_tr, "y_tr": y_tr,
            "X_te": X_te, "y_te": y_te,
        }, f)
    logging.info(f"Saved to {results_path}")

    # ── Visualize ─────────────────────────────────────────────────────────────

    # MCCA canonical images (pixel-space directions)
    save_component_images(
        mcca.components_pixel, EXP_NAME, "mcca_pixel",
        n_show=N_COMPONENTS,
        title=f"MCCA canonical directions in pixel space (top {N_COMPONENTS})",
    )

    # Side-by-side: global PCA vs MCCA pixel components
    save_component_comparison(
        {"Global PCA": pca_global.components, "MCCA (pixel)": mcca.components_pixel},
        EXP_NAME, "pca_vs_mcca", n_show=10,
    )

    # Eigenvalue scree plot
    save_eigenvalue_curve(mcca.eigenvalues, EXP_NAME, "mcca",
                          title="MCCA eigenvalues (squared singular values of stacked whitened data)")

    # 2D scatter: project all test data using MCCA (each class uses its class mean)
    all_scores_te = np.vstack([
        project_class(Xs_te[d], d, mcca) for d in range(10)
    ])
    all_labels_te = np.hstack([np.full(len(Xs_te[d]), d) for d in range(10)])
    save_projection_2d(
        all_scores_te, all_labels_te, EXP_NAME, "mcca_2d",
        xlabel="CCA 1", ylabel="CCA 2",
        title="MCCA — test set projected onto canonical component 1 & 2",
    )

    # Compare: global PCA 2D vs MCCA 2D
    pca_scores_te = pca_project(X_te, pca_global)
    save_projection_grid(
        [pca_scores_te[:, :2], all_scores_te[:, :2]],
        [y_te, all_labels_te],
        titles=["Global PCA (PC1, PC2)", "MCCA (CC1, CC2)"],
        exp_name=EXP_NAME, tag="pca_vs_mcca_2d",
    )

    # Per-digit projections onto MCCA space
    per_digit_mcca_scores = [project_class(Xs_te[d], d, mcca) for d in range(10)]
    per_digit_labels = [np.full(len(Xs_te[d]), d) for d in range(10)]
    save_projection_grid(
        per_digit_mcca_scores, per_digit_labels,
        titles=[f"Digit {d} → MCCA" for d in range(10)],
        exp_name=EXP_NAME, tag="per_digit_mcca_2d",
    )

    # Canonical correlation heatmap: (n_components × K)
    # Rows = canonical components, cols = digits; value = correlation of each digit's
    # score with the global mean score across all digits.
    global_mean_scores = np.vstack(scores_tr).mean(0, keepdims=True)   # rough reference
    corr_rows = []
    for comp_idx in range(N_COMPONENTS):
        row = []
        for d in range(10):
            s_d = scores_tr[d][:, comp_idx]
            s_d = (s_d - s_d.mean()) / (s_d.std() + 1e-12)
            row.append(float(mean_corrs[comp_idx]))  # mean pairwise corr for this component
        corr_rows.append(row)
    corr_array = np.array([[mean_corrs[i]] * 10 for i in range(N_COMPONENTS)])

    # Richer heatmap: for each component, per-digit score std (normalized by global)
    score_std_matrix = np.zeros((N_COMPONENTS, 10))
    for comp_idx in range(N_COMPONENTS):
        global_scores = np.hstack([scores_tr[d][:, comp_idx] for d in range(10)])
        global_std = global_scores.std() + 1e-12
        for d in range(10):
            score_std_matrix[comp_idx, d] = scores_tr[d][:, comp_idx].std() / global_std

    save_canonical_correlation_heatmap(
        score_std_matrix,
        exp_name=EXP_NAME, tag="mcca_score_std",
        row_labels=[f"CC{i+1}" for i in range(N_COMPONENTS)],
        col_labels=[str(d) for d in range(10)],
        title="MCCA per-digit score std / global std (1 = average, >1 = this digit uses component more)",
    )

    # ── Log result ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    append_result({
        "experiment": EXP_NAME,
        "name": f"MCCA 10-class  n_components={N_COMPONENTS}  reg={REG}",
        "eigenvalues": [round(float(v), 4) for v in mcca.eigenvalues],
        "mean_pairwise_corr": [round(float(v), 4) for v in mean_corrs],
        "time_s": round(elapsed, 1),
    })
    logging.info(f"Done. {elapsed:.1f}s")
