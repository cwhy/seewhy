"""
Clustering analysis plots for exp17b / exp17c / future clustering experiments.

Generates:
  1. Cumulative cluster assignment line plot (colored by majority label)
  2. Label accuracy curves (hard majority-vote vs Bayes)
  3. Per-cluster label distribution grid (sorted by total assignments)
  4. Cluster prototype reconstruction grid (if params file exists)

Usage:
    uv run python projects/imle-gram/scripts/gen_viz_clustering.py [exp_name]
    # default exp_name = exp17b
"""
import importlib.util, pickle, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import jax, jax.numpy as jnp

EXP_NAME = sys.argv[1] if len(sys.argv) > 1 else "exp17b"
BASE     = Path(__file__).parent.parent

# ── Load history ──────────────────────────────────────────────────────────────

with open(BASE / f"history_{EXP_NAME}.pkl", "rb") as f:
    history = pickle.load(f)

cluster_usage  = np.array(history["cluster_usage"])   # (N_EPOCHS, N_CODES)
label_acc_hard = history["label_acc_hard"]            # [(epoch, acc), ...]
label_acc_soft = history["label_acc_soft"]            # [(epoch, acc), ...]

N_EPOCHS, N_CODES = cluster_usage.shape
cum_usage = np.cumsum(cluster_usage, axis=0)          # accumulated totals

# ── Load label_probs (needed for coloring + distribution plot) ────────────────

label_probs_path = BASE / f"label_probs_{EXP_NAME}.npy"
label_probs      = np.load(label_probs_path)          # (N_CODES, 10)
majority_label   = label_probs.argmax(axis=1)         # (N_CODES,)

# ── Plot 1: Cumulative cluster usage (colored by majority label) ──────────────

epochs     = np.arange(N_EPOCHS)
digit_cmap = plt.get_cmap("tab10")
sort_by_maj = np.argsort(majority_label, kind="stable")

fig, ax = plt.subplots(figsize=(13, 5))
for c in sort_by_maj:
    lbl = majority_label[c]
    ax.plot(epochs, cum_usage[:, c], linewidth=1.0, alpha=0.7,
            color=digit_cmap(lbl), label=str(lbl))

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Accumulated samples assigned", fontsize=11)
ax.set_title(f"Cumulative cluster assignment counts — {EXP_NAME}  "
             f"(N_CODES={N_CODES}, colored by majority label)", fontsize=11)
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
url1 = save_matplotlib_figure(f"{EXP_NAME}_cluster_usage_lineplot", fig, dpi=130)
print(f"cluster usage line plot  {url1}")
plt.close(fig)

# ── Plot 2: Label accuracy curves ─────────────────────────────────────────────

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
url2 = save_matplotlib_figure(f"{EXP_NAME}_label_accuracy", fig, dpi=130)
print(f"label accuracy curves    {url2}")
plt.close(fig)

# ── Load params + reconstructions if available ────────────────────────────────

total_assigned = cum_usage[-1]
sort_by_count  = np.argsort(total_assigned)[::-1]

params_path = BASE / f"params_{EXP_NAME}.pkl"
imgs        = None   # (N_CODES, 28, 28) or None

if params_path.exists():
    exp_num = EXP_NAME.replace("exp", "")
    spec    = importlib.util.spec_from_file_location(EXP_NAME, BASE / f"experiments{exp_num}.py")
    mod     = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with open(params_path, "rb") as f:
        params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

    imgs = mod.decode_prototypes(params).reshape(N_CODES, 28, 28)

# ── Plot 3: Per-cluster label distribution (sorted by total assignments) ──────
# Overlays an inverted-colour 50%-transparent reconstruction when params exist.

n_cols = 10
n_rows = (N_CODES + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 1.7))
for i, ax in enumerate(axes.flat):
    if i < N_CODES:
        c = sort_by_count[i]
        ax.bar(range(10), label_probs[c] * 100, color="#2a7ae0", width=0.7)
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(10))
        ax.tick_params(labelsize=5)
        ax.set_title(f"c={c:x}  n={total_assigned[c]:.0f}  maj={majority_label[c]}",
                     fontsize=6, fontfamily="monospace")
        ax.yaxis.set_visible(False)
        if imgs is not None:
            ax.imshow(255 - imgs[c], cmap="gray", vmin=0, vmax=255,
                      extent=[-0.5, 9.5, 0, 100], aspect="auto",
                      alpha=0.5, interpolation="bilinear")
    else:
        ax.set_visible(False)

fig.suptitle(f"{EXP_NAME} — P(label | cluster) for all {N_CODES} clusters "
             "(sorted by total assignments, most → least)", fontsize=10)
fig.tight_layout()
url3 = save_matplotlib_figure(f"{EXP_NAME}_cluster_label_dist", fig, dpi=120)
print(f"cluster label dist       {url3}")
plt.close(fig)

# ── Plot 4: Cluster prototype reconstruction grid ─────────────────────────────

if imgs is not None:
    n_cols_r = 10
    n_rows_r = (N_CODES + n_cols_r - 1) // n_cols_r
    fig, axes = plt.subplots(n_rows_r, n_cols_r, figsize=(n_cols_r * 1.6, n_rows_r * 1.8))
    for i, ax in enumerate(axes.flat):
        if i < N_CODES:
            c = sort_by_count[i]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"c={c:x}  n={total_assigned[c]:.0f}  maj={majority_label[c]}",
                         fontsize=6, fontfamily="monospace")
            ax.axis("off")
        else:
            ax.set_visible(False)

    fig.suptitle(f"{EXP_NAME} — cluster prototype reconstructions "
                 "(sorted by total assignments, most → least)", fontsize=10)
    fig.tight_layout()
    url4 = save_matplotlib_figure(f"{EXP_NAME}_cluster_reconstructions", fig, dpi=120)
    print(f"cluster reconstructions  {url4}")
    plt.close(fig)
else:
    print(f"(skipping reconstructions — {params_path.name} not found)")
