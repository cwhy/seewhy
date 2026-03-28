"""
Post-hoc visualization for exp3 (pairwise CCA).

Usage:
    uv run python projects/component-analysis/scripts/gen_viz_exp3.py
    uv run python projects/component-analysis/scripts/gen_viz_exp3.py exp3
"""

import sys, pickle, logging
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.cca import project_cca_a, project_cca_b
from lib.interactive import save_projection_interactive, save_components_interactive
from shared_lib.media import save_svg
from shared_lib.html import save_figures_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME    = sys.argv[1] if len(sys.argv) > 1 else "exp3"
PROJECT_DIR = Path(__file__).parent.parent
results_path = PROJECT_DIR / f"cca_pairwise_{EXP_NAME}.pkl"

if not results_path.exists():
    print(f"ERROR: {results_path} not found. Run exp3 first.")
    sys.exit(1)

logging.info(f"Loading {results_path}...")
with open(results_path, "rb") as f:
    data = pickle.load(f)

pairs = data["pairs"]
X_te, y_te = data["X_te"], data["y_te"]
Xs_te = [X_te[y_te == d] for d in range(10)]

figures = []

# ── 1. Mean canonical correlation heatmap (10×10) ────────────────────────────
corr_matrix = np.zeros((10, 10))
for (i, j), r in pairs.items():
    corr_matrix[i, j] = corr_matrix[j, i] = float(r.correlations.mean())

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix, cmap="viridis", vmin=0.55, vmax=0.80)
plt.colorbar(im, ax=ax, label="mean canonical correlation")
ax.set_xticks(range(10)); ax.set_xticklabels(range(10))
ax.set_yticks(range(10)); ax.set_yticklabels(range(10))
for i in range(10):
    for j in range(10):
        if i != j:
            ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if corr_matrix[i,j] < 0.70 else "black")
ax.set_title(f"{EXP_NAME} — mean canonical correlation per digit pair")
plt.tight_layout()
url = save_svg(f"{EXP_NAME}_pairwise_corr_matrix", fig); plt.close(fig)
figures.append(("Mean canonical correlation (10×10)", url))
logging.info(f"  corr matrix → {url}")

# ── 2. Top-1 canonical correlation per pair ───────────────────────────────────
top1 = np.zeros((10, 10))
for (i, j), r in pairs.items():
    top1[i, j] = top1[j, i] = float(r.correlations[0])

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(top1, cmap="viridis", vmin=0.6, vmax=1.0)
plt.colorbar(im, ax=ax, label="CC1 canonical correlation")
ax.set_xticks(range(10)); ax.set_xticklabels(range(10))
ax.set_yticks(range(10)); ax.set_yticklabels(range(10))
for i in range(10):
    for j in range(10):
        if i != j:
            ax.text(j, i, f"{top1[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if top1[i,j] < 0.85 else "black")
ax.set_title(f"{EXP_NAME} — first canonical correlation per digit pair")
plt.tight_layout()
url = save_svg(f"{EXP_NAME}_top1_corr_matrix", fig); plt.close(fig)
figures.append(("First canonical correlation CC1 (10×10)", url))
logging.info(f"  top1 corr matrix → {url}")

# ── 3. Correlation decay curves — one line per pair ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
cmap = plt.cm.tab10
for (i, j), r in pairs.items():
    # colour by the digit that is NOT digit 1 (or by i otherwise)
    c = cmap(i / 9)
    ax.plot(range(1, len(r.correlations) + 1), r.correlations,
            alpha=0.5, linewidth=1, color=c)
# Overlay mean curve
mean_curve = np.mean([r.correlations for r in pairs.values()], axis=0)
ax.plot(range(1, len(mean_curve) + 1), mean_curve,
        color="black", linewidth=2, label="mean over all pairs")
ax.set_xlabel("Canonical component"); ax.set_ylabel("Canonical correlation")
ax.set_title(f"{EXP_NAME} — canonical correlation decay (45 pairs, coloured by digit i)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
url = save_svg(f"{EXP_NAME}_corr_decay", fig); plt.close(fig)
figures.append(("Canonical correlation decay curves", url))
logging.info(f"  decay curves → {url}")

# ── 4. Canonical direction pairs for selected digit pairs ─────────────────────
# Show top-N component images side by side: row A, row B, for a few pairs of interest
N_SHOW = 10
selected_pairs = [(0, 1), (1, 6), (3, 5), (4, 9), (0, 8)]

fig, axes = plt.subplots(len(selected_pairs) * 2, N_SHOW,
                          figsize=(N_SHOW * 1.3, len(selected_pairs) * 2 * 1.4))

def _norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)

for row_pair, (i, j) in enumerate(selected_pairs):
    r = pairs[(i, j)]
    for k in range(N_SHOW):
        ax_a = axes[row_pair * 2,     k]
        ax_b = axes[row_pair * 2 + 1, k]
        ax_a.imshow(_norm(r.directions_a[k]).reshape(28, 28), cmap="gray")
        ax_b.imshow(_norm(r.directions_b[k]).reshape(28, 28), cmap="gray")
        ax_a.axis("off"); ax_b.axis("off")
        if k == 0:
            ax_a.set_ylabel(f"digit {i}", fontsize=8)
            ax_b.set_ylabel(f"digit {j}", fontsize=8)
        if row_pair == 0:
            ax_a.set_title(f"CC{k+1}\nr={r.correlations[k]:.2f}", fontsize=7)

fig.suptitle(f"{EXP_NAME} — canonical direction pairs (selected digit pairs)", fontsize=10)
plt.tight_layout()
url = save_svg(f"{EXP_NAME}_direction_pairs", fig); plt.close(fig)
figures.append(("Canonical direction pairs (selected pairs)", url))
logging.info(f"  direction pairs → {url}")

# ── 5. All 45 pairs: CC1 direction A as a grid ────────────────────────────────
fig, axes = plt.subplots(10, 10, figsize=(14, 14))
for i in range(10):
    for j in range(10):
        ax = axes[i, j]
        if i == j:
            ax.set_facecolor("#111"); ax.axis("off"); continue
        key = (min(i,j), max(i,j))
        r = pairs[key]
        img = r.directions_a[0] if i < j else r.directions_b[0]
        ax.imshow(_norm(img).reshape(28, 28), cmap="gray")
        ax.axis("off")
        if j == 0: ax.set_ylabel(str(i), fontsize=7)
        if i == 0: ax.set_title(str(j), fontsize=7)
fig.suptitle(f"{EXP_NAME} — CC1 direction for each digit pair (row=A, col=B)", fontsize=10)
plt.tight_layout()
url = save_svg(f"{EXP_NAME}_cc1_grid", fig); plt.close(fig)
figures.append(("CC1 canonical directions — full 10×10 grid", url))
logging.info(f"  cc1 grid → {url}")

# ── 6. Interactive scatter for a few pairs ────────────────────────────────────
# Project both digit classes onto their shared CC1/CC2 subspace
interactive_urls = []
for i, j in [(0, 1), (1, 6), (3, 8)]:
    r = pairs[(i, j)]
    scores_a = project_cca_a(Xs_te[i], r)
    scores_b = project_cca_b(Xs_te[j], r)
    scores   = np.vstack([scores_a[:, :2], scores_b[:, :2]])
    labels   = np.hstack([np.full(len(scores_a), i), np.full(len(scores_b), j)])
    images   = np.vstack([Xs_te[i], Xs_te[j]])
    url = save_projection_interactive(
        scores, labels, EXP_NAME, f"cca_{i}vs{j}",
        images=images,
        xlabel="CC 1", ylabel="CC 2",
        title=f"{EXP_NAME} — CCA digit {i} vs {j}  (CC1={r.correlations[0]:.3f})",
        component_images=np.vstack([r.directions_a[:5], r.directions_b[:5]]),
        component_labels=[f"{i}:CC{k+1}" for k in range(5)] + [f"{j}:CC{k+1}" for k in range(5)],
    )
    interactive_urls.append((f"Interactive: digit {i} vs {j}", url))
    logging.info(f"  interactive {i}vs{j} → {url}")

# ── 7. Interactive component grid for a chosen pair ───────────────────────────
r_highlight = pairs[(1, 6)]
comp_images = np.vstack([r_highlight.directions_a, r_highlight.directions_b])
comp_labels = ([f"digit1 CC{k+1}  r={r_highlight.correlations[k]:.3f}" for k in range(len(r_highlight.correlations))] +
               [f"digit6 CC{k+1}  r={r_highlight.correlations[k]:.3f}" for k in range(len(r_highlight.correlations))])
comp_url = save_components_interactive(
    comp_images, EXP_NAME, "pair1_6",
    labels=comp_labels,
    title=f"{EXP_NAME} — CCA canonical directions: digit 1 vs 6",
)
logging.info(f"  interactive components 1vs6 → {comp_url}")

# ── Assemble figures page ─────────────────────────────────────────────────────
all_figures = figures + interactive_urls + [(f"Component grid: 1 vs 6", comp_url)]
page_url = save_figures_page(
    f"{EXP_NAME}_pairwise_cca",
    f"{EXP_NAME} — Pairwise CCA Analysis",
    all_figures,
)
logging.info(f"Figures page → {page_url}")

print(f"\nFigures page : {page_url}")
for caption, url in interactive_urls:
    print(f"{caption}: {url}")
print(f"Components 1v6: {comp_url}")
