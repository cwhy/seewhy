"""
Post-hoc visualization for exp1 (PCA analysis).

Reads pca_results_exp1.pkl and produces:
  - SVG figures assembled into a Tailwind HTML page (figures_page)
  - Interactive scatter: test set projected onto PC1/PC2, hover thumbnail
  - Interactive component grid: all 50 global + per-digit components clickable

Usage:
    uv run python projects/component-analysis/scripts/gen_viz_exp1.py
    uv run python projects/component-analysis/scripts/gen_viz_exp1.py exp1
"""

import sys, pickle, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.pca import project, reconstruction_error
from lib.viz import (
    save_component_images,
    save_component_comparison,
    save_projection_2d,
    save_projection_grid,
    save_variance_curve,
)
from lib.interactive import save_projection_interactive, save_components_interactive

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.html import save_figures_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME    = sys.argv[1] if len(sys.argv) > 1 else "exp1"
PROJECT_DIR = Path(__file__).parent.parent
results_path = PROJECT_DIR / f"pca_results_{EXP_NAME}.pkl"

if not results_path.exists():
    print(f"ERROR: {results_path} not found. Run exp1 first.")
    sys.exit(1)

logging.info(f"Loading {results_path}...")
with open(results_path, "rb") as f:
    data = pickle.load(f)

pca_global    = data["global"]
pca_per_digit = data["per_digit"]
X_tr, y_tr   = data["X_tr"], data["y_tr"]
X_te, y_te   = data["X_te"], data["y_te"]

figures = []   # (caption, url) pairs for the assembled HTML page

# ── Global PCA ────────────────────────────────────────────────────────────────
url = save_component_images(pca_global.components, EXP_NAME, "global_pca",
                             n_show=20, title="Global PCA — top 20 components")
figures.append(("Global PCA components (top 20)", url))

url = save_variance_curve(
    pca_global.explained_variance_ratio, EXP_NAME, "pca",
    per_class_evr={d: pca_per_digit[d].explained_variance_ratio for d in range(10)},
    title="MNIST PCA — explained variance (global=black, per-digit=colour)",
)
figures.append(("Explained variance (global + per-digit)", url))

scores_te_global = project(X_te, pca_global)
url = save_projection_2d(scores_te_global, y_te, EXP_NAME, "global_pca_2d",
                          xlabel="PC 1", ylabel="PC 2",
                          title="Global PCA — test set PC1 vs PC2")
figures.append(("Global PCA 2D projection (PC1 vs PC2)", url))

url = save_projection_2d(scores_te_global[:, 1:3], y_te, EXP_NAME, "global_pca_pc2pc3",
                          xlabel="PC 2", ylabel="PC 3",
                          title="Global PCA — test set PC2 vs PC3")
figures.append(("Global PCA 2D projection (PC2 vs PC3)", url))

# ── Per-digit PCA ─────────────────────────────────────────────────────────────
url = save_component_comparison(
    {f"Digit {d}": pca_per_digit[d].components for d in range(10)},
    EXP_NAME, "all_digits_components", n_show=10,
)
figures.append(("Per-digit PCA components (top 10 each)", url))

url = save_component_comparison(
    {"Global": pca_global.components,
     **{f"Digit {d}": pca_per_digit[d].components for d in range(5)}},
    EXP_NAME, "global_vs_digits_0to4", n_show=10,
)
figures.append(("Global PCA vs digit 0–4 components", url))

url = save_component_comparison(
    {"Global": pca_global.components,
     **{f"Digit {d}": pca_per_digit[d].components for d in range(5, 10)}},
    EXP_NAME, "global_vs_digits_5to9", n_show=10,
)
figures.append(("Global PCA vs digit 5–9 components", url))

# Per-digit own-basis projections
per_digit_scores_own = [project(X_te[y_te == d], pca_per_digit[d]) for d in range(10)]
per_digit_labels     = [np.full(len(s), d) for d, s in enumerate(per_digit_scores_own)]
url = save_projection_grid(
    per_digit_scores_own, per_digit_labels,
    titles=[f"Digit {d} PCA" for d in range(10)],
    exp_name=EXP_NAME, tag="per_digit_pca_own_basis",
)
figures.append(("Per-digit PCA (own basis, PC1 vs PC2)", url))

# Cross-projections: project all test data onto digit 0's, 5's basis
for ref_d in [0, 5]:
    cross_scores = [project(X_te[y_te == d], pca_per_digit[ref_d]) for d in range(10)]
    combined     = np.vstack([s[:, :2] for s in cross_scores])
    combined_lbl = np.hstack([np.full(len(X_te[y_te == d]), d) for d in range(10)])
    url = save_projection_2d(combined, combined_lbl, EXP_NAME,
                              f"cross_proj_basis{ref_d}",
                              xlabel="PC 1", ylabel="PC 2",
                              title=f"All digits → digit-{ref_d} PCA basis")
    figures.append((f"All digits projected onto digit-{ref_d} PCA basis", url))

# ── Assembled HTML page ────────────────────────────────────────────────────────
page_url = save_figures_page(
    f"{EXP_NAME}_pca_analysis",
    f"{EXP_NAME} — PCA Analysis",
    figures,
)
logging.info(f"Figures page → {page_url}")

# ── Interactive scatter (global PCA) ──────────────────────────────────────────
interactive_url = save_projection_interactive(
    scores_te_global[:, :2], y_te, EXP_NAME, "global_pca",
    images=X_te,
    n_max=8000,
    xlabel="PC 1", ylabel="PC 2",
    title=f"{EXP_NAME} — Global PCA interactive (test set)",
    component_images=pca_global.components[:20],
    component_labels=[f"PC {i+1}" for i in range(20)],
)
logging.info(f"Interactive scatter → {interactive_url}")

# ── Interactive component grid (global PCA, all 50) ───────────────────────────
n_comp = len(pca_global.components)
comp_meta = [{"evr": round(float(pca_global.explained_variance_ratio[i]), 5),
              "cumulative_evr": round(float(pca_global.explained_variance_ratio[:i+1].sum()), 5)}
             for i in range(n_comp)]
comp_url = save_components_interactive(
    pca_global.components, EXP_NAME, "global_pca",
    labels=[f"PC {i+1}" for i in range(n_comp)],
    metadata=comp_meta,
    title=f"{EXP_NAME} — Global PCA components (click to enlarge)",
)
logging.info(f"Interactive components → {comp_url}")

print(f"\nFigures page : {page_url}")
print(f"Interactive  : {interactive_url}")
print(f"Components   : {comp_url}")
