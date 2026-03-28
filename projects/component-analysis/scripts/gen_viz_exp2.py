"""
Post-hoc visualization for exp2 (MCCA analysis).

Reads mcca_results_exp2.pkl and produces:
  - SVG figures assembled into a Tailwind HTML page
  - Interactive scatter: test set projected onto CC1/CC2, hover thumbnail
  - Interactive component grid: MCCA pixel-space canonical directions

Usage:
    uv run python projects/component-analysis/scripts/gen_viz_exp2.py
    uv run python projects/component-analysis/scripts/gen_viz_exp2.py exp2
"""

import sys, pickle, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.pca import project as pca_project
from lib.cca import project_class, project_all, canonical_correlations
from lib.viz import (
    save_component_images,
    save_component_comparison,
    save_projection_2d,
    save_projection_grid,
    save_eigenvalue_curve,
    save_canonical_correlation_heatmap,
)
from lib.interactive import save_projection_interactive, save_components_interactive

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.html import save_figures_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME    = sys.argv[1] if len(sys.argv) > 1 else "exp2"
PROJECT_DIR = Path(__file__).parent.parent
results_path = PROJECT_DIR / f"mcca_results_{EXP_NAME}.pkl"

if not results_path.exists():
    print(f"ERROR: {results_path} not found. Run exp2 first.")
    sys.exit(1)

logging.info(f"Loading {results_path}...")
with open(results_path, "rb") as f:
    data = pickle.load(f)

mcca         = data["mcca"]
pca_global   = data["pca_global"]
Xs_tr, Xs_te = data["Xs_tr"], data["Xs_te"]
X_te, y_te   = data["X_te"],  data["y_te"]

N_COMPONENTS = len(mcca.eigenvalues)
figures = []

# ── MCCA canonical images ─────────────────────────────────────────────────────
url = save_component_images(
    mcca.components_pixel, EXP_NAME, "mcca_pixel",
    n_show=N_COMPONENTS,
    title=f"MCCA canonical directions in pixel space (top {N_COMPONENTS})",
)
figures.append((f"MCCA canonical directions (top {N_COMPONENTS})", url))

# ── PCA vs MCCA side-by-side ──────────────────────────────────────────────────
url = save_component_comparison(
    {"Global PCA": pca_global.components,
     "MCCA (pixel)": mcca.components_pixel},
    EXP_NAME, "pca_vs_mcca", n_show=min(10, N_COMPONENTS),
)
figures.append(("Global PCA vs MCCA canonical images (top 10)", url))

# ── Eigenvalue scree ──────────────────────────────────────────────────────────
url = save_eigenvalue_curve(
    mcca.eigenvalues, EXP_NAME, "mcca",
    title="MCCA eigenvalues (squared singular values of stacked whitened data)",
)
figures.append(("MCCA eigenvalues (scree)", url))

# ── 2D scatter: MCCA projections ──────────────────────────────────────────────
all_scores_te = np.vstack([project_class(Xs_te[d], d, mcca) for d in range(10)])
all_labels_te = np.hstack([np.full(len(Xs_te[d]), d) for d in range(10)])

url = save_projection_2d(
    all_scores_te, all_labels_te, EXP_NAME, "mcca_2d",
    xlabel="CC 1", ylabel="CC 2",
    title="MCCA — test set projected onto CC1 & CC2",
)
figures.append(("MCCA 2D projection (CC1 vs CC2)", url))

url = save_projection_2d(
    all_scores_te[:, 1:3], all_labels_te, EXP_NAME, "mcca_cc2cc3",
    xlabel="CC 2", ylabel="CC 3",
    title="MCCA — test set CC2 vs CC3",
)
figures.append(("MCCA 2D projection (CC2 vs CC3)", url))

# ── PCA vs MCCA 2D comparison ─────────────────────────────────────────────────
pca_scores_te = pca_project(X_te, pca_global)
url = save_projection_grid(
    [pca_scores_te[:, :2], all_scores_te[:, :2]],
    [y_te, all_labels_te],
    titles=["Global PCA (PC1, PC2)", "MCCA (CC1, CC2)"],
    exp_name=EXP_NAME, tag="pca_vs_mcca_2d",
)
figures.append(("Global PCA vs MCCA: 2D projections", url))

# ── Per-digit MCCA projections ────────────────────────────────────────────────
per_digit_mcca = [project_class(Xs_te[d], d, mcca) for d in range(10)]
per_digit_lbls = [np.full(len(Xs_te[d]), d) for d in range(10)]
url = save_projection_grid(
    per_digit_mcca, per_digit_lbls,
    titles=[f"Digit {d} → MCCA" for d in range(10)],
    exp_name=EXP_NAME, tag="per_digit_mcca_2d",
)
figures.append(("Per-digit MCCA 2D projections", url))

# ── Score std heatmap ─────────────────────────────────────────────────────────
scores_tr = project_all(Xs_tr, mcca)
score_std_matrix = np.zeros((N_COMPONENTS, 10))
for ci in range(N_COMPONENTS):
    global_std = np.hstack([scores_tr[d][:, ci] for d in range(10)]).std() + 1e-12
    for d in range(10):
        score_std_matrix[ci, d] = scores_tr[d][:, ci].std() / global_std

url = save_canonical_correlation_heatmap(
    score_std_matrix,
    exp_name=EXP_NAME, tag="mcca_score_std",
    row_labels=[f"CC{i+1}" for i in range(N_COMPONENTS)],
    col_labels=[str(d) for d in range(10)],
    title="Per-digit score std / global std (1=average, >1=digit uses this component more)",
)
figures.append(("Per-digit score std per canonical component", url))

# ── Pairwise inter-digit correlation for top 3 components ────────────────────
for ci in [0, 1, 2]:
    pair_corr = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            N = min(len(scores_tr[i]), len(scores_tr[j]))
            si = scores_tr[i][:N, ci]; si = (si - si.mean()) / (si.std() + 1e-12)
            sj = scores_tr[j][:N, ci]; sj = (sj - sj.mean()) / (sj.std() + 1e-12)
            pair_corr[i, j] = (si * sj).mean()
    url = save_canonical_correlation_heatmap(
        pair_corr,
        exp_name=EXP_NAME, tag=f"inter_digit_corr_cc{ci+1}",
        row_labels=[str(d) for d in range(10)],
        col_labels=[str(d) for d in range(10)],
        title=f"Inter-digit Pearson correlation on CC{ci+1} (training scores)",
    )
    figures.append((f"Inter-digit correlation on CC{ci+1}", url))

# ── Assembled HTML page ───────────────────────────────────────────────────────
page_url = save_figures_page(
    f"{EXP_NAME}_mcca_analysis",
    f"{EXP_NAME} — MCCA Analysis",
    figures,
)
logging.info(f"Figures page → {page_url}")

# ── Interactive scatter (MCCA, all test data) ─────────────────────────────────
# Reconstruct test images array in class order to match all_scores_te / all_labels_te
X_te_ordered = np.vstack([Xs_te[d] for d in range(10)])
interactive_url = save_projection_interactive(
    all_scores_te[:, :2], all_labels_te, EXP_NAME, "mcca",
    images=X_te_ordered,
    n_max=8000,
    xlabel="CC 1", ylabel="CC 2",
    title=f"{EXP_NAME} — MCCA interactive (test set)",
    component_images=mcca.components_pixel[:N_COMPONENTS],
    component_labels=[f"CC {i+1}" for i in range(N_COMPONENTS)],
)
logging.info(f"Interactive scatter → {interactive_url}")

# ── Interactive component grid (MCCA canonical images) ───────────────────────
comp_meta = [{"eigenvalue": round(float(mcca.eigenvalues[i]), 2)}
             for i in range(N_COMPONENTS)]
comp_url = save_components_interactive(
    mcca.components_pixel, EXP_NAME, "mcca",
    labels=[f"CC {i+1}" for i in range(N_COMPONENTS)],
    metadata=comp_meta,
    title=f"{EXP_NAME} — MCCA canonical directions in pixel space (click to enlarge)",
)
logging.info(f"Interactive components → {comp_url}")

print(f"\nFigures page : {page_url}")
print(f"Interactive  : {interactive_url}")
print(f"Components   : {comp_url}")
