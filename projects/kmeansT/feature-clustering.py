"""
kmeansT: K-means clustering of features (pixels) rather than samples.

For MNIST 28×28, each of the 784 pixels is a point described by its values
across N training samples — a vector in R^N. K-means clusters these 784
feature vectors to discover which pixels co-activate across examples.

Varying:
  - K ∈ {10, 16, 32, 48, 64}
  - Several sample subsets: random draws of different sizes + class-filtered

Output: SVG grid — rows = K, columns = subset — each cell is a 28×28 image
colored by cluster assignment.
"""

import os, sys
import colorsys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_media
from shared_lib.html import save_html
from lib.kmeans import fit as kmeans_fit


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def distinct_colors(n: int) -> list[str]:
    """Generate n visually distinct hex colors via HSV with varied lightness."""
    cols = []
    for i in range(n):
        h = i / n
        s = 0.65 + 0.20 * (i % 3 == 0)
        v = 0.55 + 0.30 * (i % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return cols


# ---------------------------------------------------------------------------
# SVG builder
# ---------------------------------------------------------------------------

CELL = 12       # px per pixel-cell inside a panel
IMG_H = 28
IMG_W = 28
GRID_GAP = 24   # gap between panels
LABEL_LEFT = 70
LABEL_TOP = 72


def build_svg(
    assignments_grid: list[list[np.ndarray]],
    K_values: list[int],
    col_labels: list[str],
    title: str = "kmeansT — pixel feature clustering",
) -> str:
    n_rows = len(K_values)
    n_cols = len(col_labels)

    panel_w = IMG_W * CELL
    panel_h = IMG_H * CELL

    svg_w = LABEL_LEFT + n_cols * panel_w + (n_cols - 1) * GRID_GAP + 40
    svg_h = 30 + LABEL_TOP + n_rows * panel_h + (n_rows - 1) * GRID_GAP + 50

    lines: list[str] = []
    a = lines.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">')
    a(f'  <rect width="{svg_w}" height="{svg_h}" fill="#12121e"/>')
    a(  '  <style>text{font-family:monospace;fill:#c8c8d8}</style>')

    # title
    a(f'  <text x="{svg_w//2}" y="22" text-anchor="middle" font-size="13" font-weight="bold">{title}</text>')

    # column headers
    for col, label in enumerate(col_labels):
        cx = LABEL_LEFT + col * (panel_w + GRID_GAP) + panel_w // 2
        a(f'  <text x="{cx}" y="{30 + LABEL_TOP - 14}" text-anchor="middle" font-size="10">{label}</text>')

    # row headers + pixel grids
    for row, K in enumerate(K_values):
        panel_top = 30 + LABEL_TOP + row * (panel_h + GRID_GAP)
        ry = panel_top + panel_h // 2 + 4
        a(f'  <text x="4" y="{ry}" text-anchor="start" font-size="12" font-weight="bold">K={K}</text>')

        colors = distinct_colors(K)

        for col in range(n_cols):
            assignments = assignments_grid[row][col]   # (784,)
            px0 = LABEL_LEFT + col * (panel_w + GRID_GAP)
            py0 = panel_top

            # panel background
            a(f'  <rect x="{px0}" y="{py0}" width="{panel_w}" height="{panel_h}" fill="#0a0a18" rx="2"/>')

            # pixel cells
            for i in range(IMG_H):
                for j in range(IMG_W):
                    k = int(assignments[i * IMG_W + j])
                    px = px0 + j * CELL
                    py = py0 + i * CELL
                    a(f'  <rect x="{px}" y="{py}" width="{CELL}" height="{CELL}" fill="{colors[k]}"/>')

    # --- legend row at bottom (one per K row) ---
    legend_y0 = 30 + LABEL_TOP + n_rows * (panel_h + GRID_GAP) + 4
    for row, K in enumerate(K_values):
        colors = distinct_colors(K)
        lx = LABEL_LEFT + (row * (svg_w - LABEL_LEFT - 40)) // n_rows
        ly = legend_y0
        a(f'  <text x="{lx}" y="{ly + 12}" font-size="9" fill="#888">K={K}: </text>')
        for ki in range(K):
            bx = lx + 30 + ki * 10
            a(f'  <rect x="{bx}" y="{ly}" width="8" height="14" fill="{colors[ki]}" rx="1"/>')

    a('</svg>')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Subset definitions
# ---------------------------------------------------------------------------

def make_subsets(X_train: np.ndarray, y_train: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return list of (label, X_subset) where X_subset.T will be clustered."""
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    n = X_train.shape[0]

    subsets = [
        ("rand-500 s0",   X_train[rng0.choice(n, 500,   replace=False)]),
        ("rand-500 s1",   X_train[rng1.choice(n, 500,   replace=False)]),
        ("rand-2000 s2",  X_train[rng2.choice(n, 2000,  replace=False)]),
        ("rand-10000 s0", X_train[rng0.choice(n, 10000, replace=False)]),
        ("cls 0,1,2",     X_train[np.isin(y_train, [0, 1, 2])]),
        ("cls 7,8,9",     X_train[np.isin(y_train, [7, 8, 9])]),
    ]
    return subsets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

K_VALUES = [10, 16, 32, 48, 64]
N_ITER   = 40


def run():
    print("Loading MNIST …")
    data = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples, -1).astype(np.float32) / 255.0
    y_train = np.array(data.y)
    print(f"  X_train: {X_train.shape}")

    subsets = make_subsets(X_train, y_train)
    col_labels = [s[0] for s in subsets]
    n_subsets  = len(subsets)

    print(f"\nK values : {K_VALUES}")
    print(f"Subsets  : {col_labels}")
    print(f"Iteratons: {N_ITER}\n")

    assignments_grid: list[list[np.ndarray]] = []

    for K in K_VALUES:
        row: list[np.ndarray] = []
        for label, X_sub in subsets:
            X_T = X_sub.T                            # (784, n_samples) — cluster pixels
            print(f"  K={K:2d}  subset={label:16s}  X_T={X_T.shape}")
            seed = K * 1000 + hash(label) % 997
            assignments, centroids = kmeans_fit(X_T, K, n_iter=N_ITER, seed=seed)
            counts = np.bincount(assignments, minlength=K)
            print(f"    cluster sizes  min={counts.min():3d}  max={counts.max():3d}  mean={counts.mean():.1f}")
            row.append(assignments)
        assignments_grid.append(row)

    print("\nBuilding SVG …")
    svg = build_svg(
        assignments_grid=assignments_grid,
        K_values=K_VALUES,
        col_labels=col_labels,
    )
    svg_bytes = svg.encode("utf-8")

    # Upload SVG via shared_lib (R2 with local fallback)
    svg_url = save_media("kmeansT_feature_clustering.svg", svg_bytes, "image/svg+xml")
    print(f"SVG    → {svg_url}")

    # Build a minimal HTML page that embeds the SVG inline
    html = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='utf-8'>"
        "<title>kmeansT — pixel feature clustering</title>"
        "<style>body{margin:0;background:#12121e;display:flex;justify-content:center;padding:20px}</style>"
        "</head><body>"
        + svg +
        "</body></html>"
    )
    html_url = save_html("kmeansT_feature_clustering", html)
    print(f"HTML   → {html_url}")

    # also keep a local copy next to the script for quick reference
    local_path = Path(__file__).parent / "latest.svg"
    local_path.write_text(svg)
    print(f"Local  → {local_path}")


if __name__ == "__main__":
    run()
