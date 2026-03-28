"""
Compare feature-space PCA (exp1) vs data-space PCA (exp5) per digit.

Feature PCA components: right singular vectors V_k ∈ ℝ^784 (eigenimages).
Data PCA pixel loadings: (X_d - mean)^T u_k ∈ ℝ^784 — how each pixel
  correlates with the k-th data-space component direction.

For each digit: one figure with two rows (feature / data) and N_SHOW columns.
"""

import sys, pickle, logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared_lib.html import save_figures_page

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
from lib.viz import save_component_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_SHOW = 10

# ── Load ──────────────────────────────────────────────────────────────────────
logging.info("Loading exp1 (feature PCA)...")
with open(PROJECT / "pca_results_exp1.pkl", "rb") as f:
    exp1 = pickle.load(f)

logging.info("Loading exp5 (data PCA)...")
with open(PROJECT / "data_pca_results_exp5.pkl", "rb") as f:
    exp5 = pickle.load(f)

# ── Build comparison figures ──────────────────────────────────────────────────
figures = []

for d in range(10):
    # Feature PCA: top N_SHOW components (rows of Vt, shape (50, 784))
    feature_comps = exp1["per_digit"][d].components[:N_SHOW]   # (N_SHOW, 784)

    # Data PCA: pixel loadings = (X_d - mean)^T @ U[:, :N_SHOW]  → (784, N_SHOW)
    r = exp5["per_digit"][d]
    Xc = r["X_d"] - r["mean"]
    pixel_loadings = Xc.T @ r["U"][:, :N_SHOW]                # (784, N_SHOW)
    data_comps = pixel_loadings.T                               # (N_SHOW, 784)

    url = save_component_comparison(
        {"Feature PCA (exp1)": feature_comps,
         "Data PCA (exp5)":    data_comps},
        exp_name="cmp",
        tag=f"digit{d}",
        n_show=N_SHOW,
    )
    figures.append((f"Digit {d}", url))

page_url = save_figures_page(
    "cmp_feature_vs_data_pca",
    "Feature PCA vs Data PCA — per digit",
    figures,
)
logging.info(f"Page → {page_url}")
print(page_url)
