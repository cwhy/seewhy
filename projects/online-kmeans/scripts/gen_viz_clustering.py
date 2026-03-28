"""
Clustering analysis plots for online-kmeans experiments.

Generates SVGs for each plot, then assembles them into a single HTML page.

Plots:
  1. Per-epoch cluster assignment line plot (coloured by majority label)
  2. Inertia curve (mean L2 to nearest centroid, train + eval)
  3. Label accuracy curves (hard majority-vote vs Bayes)
  4. Per-cluster label distribution grid (sorted by total assignments)
  5. Cluster centroid image grid (sorted by majority label)
  6. Centroid cosine similarity heatmap (N_CODES × N_CODES)

Usage:
    uv run python projects/online-kmeans/scripts/gen_viz_clustering.py [exp_name]
    # default exp_name = exp1
"""
import importlib.util, pickle, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_svg
from shared_lib.html import save_figures_page, save_flat_grid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

EXP_NAME = sys.argv[1] if len(sys.argv) > 1 else "exp1"
BASE     = Path(__file__).parent.parent

# ── Load history ──────────────────────────────────────────────────────────────

with open(BASE / f"history_{EXP_NAME}.pkl", "rb") as f:
    history = pickle.load(f)

cluster_usage   = np.array(history["cluster_usage"])    # (N_EPOCHS, N_CODES)
label_acc_hard  = history.get("label_acc_hard", [])     # [(epoch, acc), ...]
label_acc_soft  = history.get("label_acc_soft", [])     # [(epoch, acc), ...]
match_loss      = history.get("match_loss", [])         # [loss per epoch]
match_loss_eval = history.get("match_loss_eval", [])    # [(epoch, loss), ...]

N_EPOCHS, N_CODES = cluster_usage.shape
cum_usage = np.cumsum(cluster_usage, axis=0)

# ── Load label_probs ──────────────────────────────────────────────────────────

label_probs_path = BASE / f"label_probs_{EXP_NAME}.npy"
label_probs      = np.load(label_probs_path)   # (N_CODES, 10)
majority_label   = label_probs.argmax(axis=1)  # (N_CODES,)

figures = []  # [(caption, url), ...]

# ── Plot 1: Per-epoch cluster usage (coloured by majority label) ───────────────

epochs     = np.arange(N_EPOCHS)
digit_cmap = plt.get_cmap("tab10")
sort_by_maj = np.argsort(majority_label, kind="stable")

fig, ax = plt.subplots(figsize=(13, 5))
for c in sort_by_maj:
    lbl = majority_label[c]
    ax.plot(epochs, cluster_usage[:, c], linewidth=1.0, alpha=0.7,
            color=digit_cmap(lbl), label=str(lbl))

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Samples assigned", fontsize=11)
ax.set_title(f"Per-epoch cluster assignment counts — {EXP_NAME}  "
             f"(N_CODES={N_CODES}, coloured by majority label)", fontsize=11)
ax.set_xlim(0, N_EPOCHS - 1)
ax.grid(True, alpha=0.25)

handles, labels_ = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels_):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), [f"digit {k}" for k in seen.keys()],
          title="majority label", ncol=1, fontsize=8, title_fontsize=8,
          loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
fig.tight_layout()
figures.append(("Per-epoch cluster usage", save_svg(f"{EXP_NAME}_cluster_usage_lineplot", fig)))
plt.close(fig)

# ── Plot 2: Inertia curve ─────────────────────────────────────────────────────

if match_loss:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(match_loss)), match_loss, linewidth=1.5,
            color="#555", alpha=0.7, label="train inertia (per epoch)")
    if match_loss_eval:
        eval_epochs, eval_losses = zip(*match_loss_eval)
        ax.plot(eval_epochs, eval_losses, "o", color="#e05c2a",
                markersize=5, label="eval inertia")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean L2 to nearest centroid", fontsize=11)
    ax.set_title(f"Inertia — {EXP_NAME}", fontsize=11)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    figures.append(("Inertia curve", save_svg(f"{EXP_NAME}_loss_curve", fig)))
    plt.close(fig)

# ── Plot 3: Label accuracy curves ─────────────────────────────────────────────

if label_acc_hard and label_acc_soft:
    epochs_h, accs_h = zip(*label_acc_hard)
    epochs_s, accs_s = zip(*label_acc_soft)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_h, [a * 100 for a in accs_h], "o-", color="#e05c2a",
            linewidth=2, markersize=4, label="Hard (majority-vote label per cluster)")
    ax.plot(epochs_s, [a * 100 for a in accs_s], "s-", color="#2a7ae0",
            linewidth=2, markersize=4,
            label=r"Bayes ($\mathrm{argmax}_y \sum_c P(c|x)\,P(y|c)$)")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Label accuracy (%)", fontsize=11)
    ax.set_title(f"Cluster label accuracy over training — {EXP_NAME}\n"
                 "Hard: majority-vote per cluster  |  Bayes: soft P(c|x) weighted label prediction",
                 fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.annotate(f"{accs_h[-1]*100:.1f}%", xy=(epochs_h[-1], accs_h[-1]*100),
                xytext=(8, -12), textcoords="offset points", fontsize=8, color="#e05c2a")
    ax.annotate(f"{accs_s[-1]*100:.1f}%", xy=(epochs_s[-1], accs_s[-1]*100),
                xytext=(8, 4), textcoords="offset points", fontsize=8, color="#2a7ae0")
    fig.tight_layout()
    figures.append(("Label accuracy (hard + Bayes)", save_svg(f"{EXP_NAME}_label_accuracy", fig)))
    plt.close(fig)

# ── Load params (centroids) ────────────────────────────────────────────────────

total_assigned   = cum_usage[-1]
sort_by_count    = np.argsort(total_assigned)[::-1]
sort_by_maj_count = np.lexsort((-total_assigned, majority_label))

params_path = BASE / f"params_{EXP_NAME}.pkl"
imgs        = None   # (N_CODES, 28, 28) or None

if params_path.exists():
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    raw = np.array(params["centroids"]).reshape(N_CODES, -1)  # (K, 784)
    c_min = raw.min(1, keepdims=True)
    c_max = raw.max(1, keepdims=True)
    imgs = ((raw - c_min) / np.where(c_max - c_min > 0, c_max - c_min, 1.0) * 255.0).reshape(N_CODES, 28, 28)

# ── Plot 4: Per-cluster label distribution ────────────────────────────────────

def _cluster_dist_svg(c):
    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.bar(range(10), label_probs[c] * 100, color="#2a7ae0", width=0.7)
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(0, 100)
    ax.set_xticks(range(10)); ax.tick_params(labelsize=7)
    ax.set_title(f"c={c:x}  n={total_assigned[c]:.0f}  maj={majority_label[c]}",
                 fontsize=8, fontfamily="monospace")
    ax.yaxis.set_visible(False)
    if imgs is not None:
        ax.imshow(255 - imgs[c], cmap="gray", vmin=0, vmax=255,
                  extent=[-0.5, 9.5, 0, 100], aspect="auto", alpha=0.5, interpolation="bilinear")
    fig.tight_layout(pad=0.3)
    url = save_svg(f"{EXP_NAME}_cluster_{c:03d}_dist", fig)
    plt.close(fig)
    return url

cluster_svg_urls = [_cluster_dist_svg(c) for c in sort_by_count]
label_dist_detail_url = save_flat_grid(
    f"{EXP_NAME}_cluster_label_dist_detail",
    f"{EXP_NAME} — P(label | cluster) sorted by assignment count",
    [(f"c={sort_by_count[i]:x}  n={total_assigned[sort_by_count[i]]:.0f}  maj={majority_label[sort_by_count[i]]}",
      url) for i, url in enumerate(cluster_svg_urls)],
)

n_cols = 10
n_rows = (N_CODES + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 1.7))
for i, ax in enumerate(axes.flat):
    if i < N_CODES:
        c = sort_by_count[i]
        ax.bar(range(10), label_probs[c] * 100, color="#2a7ae0", width=0.7)
        ax.set_xlim(-0.5, 9.5); ax.set_ylim(0, 100); ax.set_xticks(range(10))
        ax.tick_params(labelsize=5)
        ax.set_title(f"c={c:x}  n={total_assigned[c]:.0f}  maj={majority_label[c]}",
                     fontsize=6, fontfamily="monospace")
        ax.yaxis.set_visible(False)
        if imgs is not None:
            ax.imshow(255 - imgs[c], cmap="gray", vmin=0, vmax=255,
                      extent=[-0.5, 9.5, 0, 100], aspect="auto", alpha=0.5, interpolation="bilinear")
    else:
        ax.set_visible(False)
fig.suptitle(f"{EXP_NAME} — P(label | cluster) for all {N_CODES} clusters "
             "(sorted by total assignments, most → least)", fontsize=10)
fig.tight_layout()
figures.append(("Label distribution per cluster",
                save_svg(f"{EXP_NAME}_cluster_label_dist", fig),
                label_dist_detail_url))
plt.close(fig)

# ── Plot 5: Centroid image grid ────────────────────────────────────────────────

if imgs is not None:
    n_cols_r = 10
    n_rows_r = (N_CODES + n_cols_r - 1) // n_cols_r
    fig, axes = plt.subplots(n_rows_r, n_cols_r, figsize=(n_cols_r * 1.6, n_rows_r * 1.8))
    for i, ax in enumerate(axes.flat):
        if i < N_CODES:
            c = sort_by_maj_count[i]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"c={c:x}  maj={majority_label[c]}  n={total_assigned[c]:.0f}",
                         fontsize=6, fontfamily="monospace")
            ax.axis("off")
        else:
            ax.set_visible(False)

    fig.suptitle(f"{EXP_NAME} — cluster centroids "
                 "(sorted by majority label, then assignment count)", fontsize=10)
    fig.tight_layout()
    figures.append(("Centroid images", save_svg(f"{EXP_NAME}_cluster_centroids", fig)))
    plt.close(fig)

# ── Plot 6: Centroid cosine similarity heatmap ────────────────────────────────

if imgs is not None:
    p   = imgs.reshape(N_CODES, -1).astype(np.float32)
    p   = p - p.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    p_n = p / nrm

    order = sort_by_maj_count
    corr  = p_n[order] @ p_n[order].T

    maj_sorted = majority_label[order]
    boundaries = np.where(np.diff(maj_sorted))[0] + 1

    size_in = max(6, N_CODES / 20)
    fig, ax = plt.subplots(figsize=(size_in + 1.5, size_in))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="cosine similarity")

    for b in boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=0.6, alpha=0.7)
        ax.axvline(b - 0.5, color="black", linewidth=0.6, alpha=0.7)

    prev = 0
    for b in np.append(boundaries, N_CODES):
        mid = (prev + b) / 2
        lbl = maj_sorted[prev]
        ax.text(mid, -1.5, str(lbl), ha="center", va="bottom", fontsize=7, color="black")
        ax.text(-1.5, mid, str(lbl), ha="right",  va="center", fontsize=7, color="black")
        prev = b

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{EXP_NAME} — centroid cosine similarity  "
                 f"(clusters sorted by majority label)", fontsize=10)
    fig.tight_layout()
    figures.append(("Centroid correlation", save_svg(f"{EXP_NAME}_centroid_corr", fig)))
    plt.close(fig)

# ── Assemble HTML page ────────────────────────────────────────────────────────

page_url = save_figures_page(f"{EXP_NAME}_clustering", f"{EXP_NAME} — clustering analysis", figures)
print(page_url)
