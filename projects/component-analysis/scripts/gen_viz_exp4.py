"""
Post-hoc visualization for exp4 (ICA analysis).

Usage:
    uv run python projects/component-analysis/scripts/gen_viz_exp4.py
    uv run python projects/component-analysis/scripts/gen_viz_exp4.py exp4
"""

import sys, pickle, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ica import project
from lib.pca import fit_pca, project as pca_project
from lib.viz import (
    save_component_images,
    save_component_comparison,
    save_projection_2d,
    save_projection_grid,
)
from lib.interactive import save_projection_interactive, save_components_interactive
from shared_lib.html import save_figures_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME    = sys.argv[1] if len(sys.argv) > 1 else "exp4"
PROJECT_DIR = Path(__file__).parent.parent
results_path = PROJECT_DIR / f"ica_results_{EXP_NAME}.pkl"

if not results_path.exists():
    print(f"ERROR: {results_path} not found. Run exp4 first.")
    sys.exit(1)

logging.info(f"Loading {results_path}...")
with open(results_path, "rb") as f:
    data = pickle.load(f)

ica_global    = data["global"]
ica_per_digit = data["per_digit"]
X_tr, y_tr   = data["X_tr"], data["y_tr"]
X_te, y_te   = data["X_te"], data["y_te"]

figures = []

# ── Global ICA mixing atoms ───────────────────────────────────────────────────
url = save_component_images(ica_global.mixing, EXP_NAME, "global_ica_atoms",
                             n_show=50, title="Global ICA — mixing atoms (pixel-space)")
figures.append(("Global ICA mixing atoms (pixel-space)", url))

url = save_component_images(ica_global.components, EXP_NAME, "global_ica_filters",
                             n_show=50, title="Global ICA — unmixing filters")
figures.append(("Global ICA unmixing filters", url))

# ── PCA vs ICA side by side ───────────────────────────────────────────────────
pca_global = fit_pca(X_tr, 50)
url = save_component_comparison(
    {"Global PCA": pca_global.components,
     "Global ICA (atoms)": ica_global.mixing},
    EXP_NAME, "pca_vs_ica", n_show=10,
)
figures.append(("Global PCA vs ICA mixing atoms (top 10)", url))

# ── Per-digit ICA atoms ───────────────────────────────────────────────────────
url = save_component_comparison(
    {f"Digit {d}": ica_per_digit[d].mixing for d in range(10)},
    EXP_NAME, "all_digits_ica_atoms", n_show=10,
)
figures.append(("Per-digit ICA atoms (top 10 each)", url))

# ── 2D projections ────────────────────────────────────────────────────────────
scores_te_global = project(X_te, ica_global)
url = save_projection_2d(scores_te_global, y_te, EXP_NAME, "global_ica_2d",
                          xlabel="IC 1", ylabel="IC 2",
                          title="Global ICA — test set projected onto IC1 & IC2")
figures.append(("Global ICA 2D projection (IC1 vs IC2)", url))

url = save_projection_2d(scores_te_global[:, 1:3], y_te, EXP_NAME, "global_ica_ic2ic3",
                          xlabel="IC 2", ylabel="IC 3",
                          title="Global ICA — IC2 vs IC3")
figures.append(("Global ICA 2D projection (IC2 vs IC3)", url))

# Compare PCA vs ICA 2D projections
pca_scores_te = pca_project(X_te, pca_global)
url = save_projection_grid(
    [pca_scores_te[:, :2], scores_te_global[:, :2]],
    [y_te, y_te],
    titles=["Global PCA (PC1, PC2)", "Global ICA (IC1, IC2)"],
    exp_name=EXP_NAME, tag="pca_vs_ica_2d",
)
figures.append(("Global PCA vs ICA: 2D projections", url))

# Per-digit own-basis projections
per_digit_scores = [project(X_te[y_te == d], ica_per_digit[d]) for d in range(10)]
per_digit_labels = [np.full(len(s), d) for d, s in enumerate(per_digit_scores)]
url = save_projection_grid(
    per_digit_scores, per_digit_labels,
    titles=[f"Digit {d} ICA" for d in range(10)],
    exp_name=EXP_NAME, tag="per_digit_ica_2d",
)
figures.append(("Per-digit ICA 2D projections (own basis)", url))

# ── Assembled HTML page ───────────────────────────────────────────────────────
page_url = save_figures_page(
    f"{EXP_NAME}_ica_analysis",
    f"{EXP_NAME} — ICA Analysis",
    figures,
)
logging.info(f"Figures page → {page_url}")

# ── Interactive scatter ───────────────────────────────────────────────────────
interactive_url = save_projection_interactive(
    scores_te_global[:, :2], y_te, EXP_NAME, "global_ica",
    images=X_te,
    n_max=8000,
    xlabel="IC 1", ylabel="IC 2",
    title=f"{EXP_NAME} — Global ICA interactive (test set)",
    component_images=ica_global.mixing[:20],
    component_labels=[f"IC {i+1} (atom)" for i in range(20)],
)
logging.info(f"Interactive scatter → {interactive_url}")

# ── Interactive component grid ────────────────────────────────────────────────
comp_url = save_components_interactive(
    ica_global.mixing, EXP_NAME, "global_ica_atoms",
    labels=[f"IC {i+1}" for i in range(len(ica_global.mixing))],
    title=f"{EXP_NAME} — Global ICA mixing atoms (click to enlarge)",
)
logging.info(f"Interactive components → {comp_url}")

print(f"\nFigures page : {page_url}")
print(f"Interactive  : {interactive_url}")
print(f"Components   : {comp_url}")
