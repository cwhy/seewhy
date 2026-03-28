"""
t-SNE scatter: full MNIST data + spherical K-means cluster centers.

t-SNE is run on a sample of MNIST training points + all centroids together,
so centroids land in the natural data embedding.

Each centroid shown as 4 concentric translucent discs (p25/p50/p75/p100 of
2D distances from assigned points in the t-SNE embedding).  Scaled so the
median p50 ring just touches its nearest centroid neighbour.

Usage:
    uv run python projects/online-kmeans/scripts/gen_viz_tsne_clusters.py [exp_name]
    # default: exp5
"""
import colorsys, pickle, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_svg
from shared_lib.datasets import load_supervised_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.manifold import TSNE

EXP_NAME  = sys.argv[1] if len(sys.argv) > 1 else "exp5"
N_SAMPLE  = 8000    # MNIST points to include in t-SNE (+ all centroids)
BASE      = Path(__file__).parent.parent

# ── Load centroids & label probs ───────────────────────────────────────────────

with open(BASE / f"params_{EXP_NAME}.pkl", "rb") as f:
    params = pickle.load(f)
centroids = np.array(params["centroids"])    # (K, 784) unit-norm
N_CODES   = len(centroids)

label_probs    = np.load(BASE / f"label_probs_{EXP_NAME}.npy")
majority_label = label_probs.argmax(axis=1)  # (K,)

# ── Load + normalise MNIST training data ──────────────────────────────────────

data  = load_supervised_image("mnist")
X_raw = data.X.reshape(60000, 784).astype(np.float32)
norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
X_all = X_raw / np.where(norms > 0, norms, 1.0)          # (60000, 784)
Y_all = np.array(data.y).astype(np.int32)                 # (60000,)

# Assign every training point to a centroid
dots_all  = X_all @ centroids.T    # (60000, K)
assign_all = np.argmax(dots_all, axis=1)

# Subsample for t-SNE (stratified by label for balance)
rng = np.random.default_rng(42)
n_per_class = N_SAMPLE // 10
idx_list = []
for d in range(10):
    idx_d = np.where(Y_all == d)[0]
    idx_list.append(rng.choice(idx_d, size=n_per_class, replace=False))
sample_idx  = np.concatenate(idx_list)
X_sample    = X_all[sample_idx]            # (N_SAMPLE, 784)
Y_sample    = Y_all[sample_idx]            # (N_SAMPLE,)
assign_samp = assign_all[sample_idx]       # (N_SAMPLE,)

# Combined array for t-SNE: data first, centroids last
X_combined = np.vstack([X_sample, centroids])   # (N_SAMPLE + K, 784)

print(f"Running t-SNE on {len(X_combined):,} points ({N_SAMPLE} data + {N_CODES} centroids) …")
tsne = TSNE(n_components=2, random_state=42,
            perplexity=40, max_iter=2000, metric="cosine")
pos_all = tsne.fit_transform(X_combined)   # (N_SAMPLE + K, 2)
print("t-SNE done.")

pos_data = pos_all[:N_SAMPLE]              # (N_SAMPLE, 2)  data points
pos_cent = pos_all[N_SAMPLE:]              # (K, 2)          centroids

# ── Per-cluster radius percentiles in 2D t-SNE space ─────────────────────────

PERCENTILES = [25, 50, 75, 100]
radii_2d    = np.zeros((N_CODES, len(PERCENTILES)))

for c in range(N_CODES):
    mask = assign_samp == c
    if mask.sum() > 0:
        diffs = pos_data[mask] - pos_cent[c]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        for i, p in enumerate(PERCENTILES):
            radii_2d[c, i] = np.percentile(dists, p)

# Global scale: find the single factor so no two p25 circles touch
diff_cc  = pos_cent[:, None, :] - pos_cent[None, :, :]
dists_cc = np.sqrt((diff_cc ** 2).sum(axis=2))
np.fill_diagonal(dists_cc, np.inf)
nn_dists = dists_cc.min(axis=1)                          # (K,) nearest centroid dist

scale = np.min(nn_dists * 0.5 / (radii_2d[:, 2] + 1e-9))  # p75 don't touch
radii_scaled = radii_2d * scale                          # (K, 4)

# ── Per-cluster color with hue/sat jitter within each digit ───────────────────

digit_cmap = plt.get_cmap("tab10")

def jitter_color(base_rgba, rng, h_std=0.025, s_std=0.08):
    r, g, b, _ = base_rgba
    h, s, v    = colorsys.rgb_to_hsv(r, g, b)
    h = (h + rng.normal(0, h_std)) % 1.0
    s = float(np.clip(s + rng.normal(0, s_std), 0.3, 1.0))
    return colorsys.hsv_to_rgb(h, s, v)

cluster_colors = []
for c in range(N_CODES):
    base = digit_cmap(int(majority_label[c]))
    cluster_colors.append(jitter_color(base, rng))

# ── Plot ───────────────────────────────────────────────────────────────────────

# alphas: p100 (outermost) → p25 (innermost)
ALPHAS = [0.75, 0.54, 0.09, 0.09]  # indexed as p25, p50, p75, p100

fig, ax = plt.subplots(figsize=(14, 12))

# MNIST data points — tiny, very transparent
for d in range(10):
    mask = Y_sample == d
    base = digit_cmap(d)
    ax.scatter(pos_data[mask, 0], pos_data[mask, 1],
               s=1.5, color=base, alpha=0.12, linewidths=0, rasterized=True)

# Centroid rings + centre dot
for c in range(N_CODES):
    x, y  = pos_cent[c]
    color = cluster_colors[c]
    for i in reversed(range(len(PERCENTILES))):
        r = radii_scaled[c, i]
        circle = Circle((x, y), r, color=color, alpha=ALPHAS[i], linewidth=0)
        ax.add_patch(circle)
    ax.plot(x, y, "o", color=color, markersize=5, alpha=1.0, zorder=10,
            markeredgewidth=0.5, markeredgecolor="white")

# Legend
for d in range(10):
    ax.plot([], [], "o", color=digit_cmap(d), markersize=9, label=str(d))
ax.legend(title="digit", ncol=1, fontsize=9, title_fontsize=9,
          loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

margin = 5.0
ax.set_xlim(pos_all[:, 0].min() - margin, pos_all[:, 0].max() + margin)
ax.set_ylim(pos_all[:, 1].min() - margin, pos_all[:, 1].max() + margin)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    f"{EXP_NAME} — t-SNE  ({N_SAMPLE:,} MNIST + {N_CODES} centroids, cosine)\n"
    "Rings = p25 / p50 / p75 / p100 of 2D distances from assigned points"
    " · scaled so median p50 ≈ touches nearest centroid",
    fontsize=11
)
fig.tight_layout()

url = save_svg(f"{EXP_NAME}_tsne_clusters", fig)
plt.close(fig)
print(url)
