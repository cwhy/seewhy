"""
Regenerate all visualizations for exp5 (data-dimension PCA per digit).

Reads data_pca_results_exp5.pkl from the project root and re-renders
all figures, uploading to R2 / local outputs/.

Usage:
    uv run python projects/component-analysis/scripts/gen_viz_exp5.py
"""

import sys, pickle, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.html import save_figures_page

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.viz import save_data_pca_grid, save_variance_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp5"
PROJECT_DIR = Path(__file__).parent.parent
PKL = PROJECT_DIR / f"data_pca_results_{EXP_NAME}.pkl"


def main():
    logging.info(f"Loading {PKL}...")
    with open(PKL, "rb") as f:
        data = pickle.load(f)

    per_digit = data["per_digit"]
    figures = []

    # EVR curves
    evr_by_digit = {d: per_digit[d]["evr"] for d in range(10)}
    mean_evr = np.stack(list(evr_by_digit.values())).mean(axis=0)
    url = save_variance_curve(
        mean_evr, EXP_NAME, "data_pca",
        per_class_evr=evr_by_digit,
        title="Data-dimension PCA — explained variance per digit (mean=black)",
    )
    figures.append(("Explained variance (data-dimension PCA)", url))

    # Per-digit grids
    for d in range(10):
        r = per_digit[d]
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
    logging.info(f"Page → {page_url}")
    print(page_url)


if __name__ == "__main__":
    main()
