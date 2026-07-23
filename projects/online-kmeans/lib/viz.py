"""
Shared visualization utilities for online-kmeans experiments.

K-means has no decoder — prototypes are centroids directly.
All functions save to R2 via save_matplotlib_figure and return the URL.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure


def save_centroid_grid(centroids, exp_name, tag, n_cols=10):
    """Display all cluster centroids as 28×28 images in a grid."""
    n_codes = len(centroids)
    n_rows = (n_codes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.5))
    for i, ax in enumerate(axes.flat):
        if i < n_codes:
            ax.imshow(centroids[i].reshape(28, 28), cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest")
            ax.set_title(str(i), fontsize=6)
            ax.axis("off")
        else:
            ax.set_visible(False)
    fig.suptitle(f"{exp_name} centroids — {tag}", fontsize=10)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_centroids_{tag}", fig, format="png", dpi=150)
    plt.close(fig)
    logging.info(f"  centroid grid   → {url}")
    return url


def save_learning_curves(history, exp_name, subtitle=""):
    """Plot inertia (train + eval) and label accuracy over epochs."""
    match_loss = history.get("match_loss", [])
    match_loss_eval = history.get("match_loss_eval", [])
    label_acc_hard = history.get("label_acc_hard", [])
    label_acc_soft = history.get("label_acc_soft", [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if match_loss:
        epochs = range(len(match_loss))
        axes[0].plot(epochs, match_loss, label="train inertia", color="#555", alpha=0.8)
    if match_loss_eval:
        eval_e = [e for e, _ in match_loss_eval]
        eval_v = [v for _, v in match_loss_eval]
        axes[0].plot(eval_e, eval_v, label="eval inertia", linestyle="--", color="#e05c2a")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean L2 to centroid")
    axes[0].set_title("Inertia (L2 to nearest centroid)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if label_acc_hard:
        acc_e = [e for e, _ in label_acc_hard]
        acc_v = [a * 100 for _, a in label_acc_hard]
        axes[1].plot(acc_e, acc_v, color="#e05c2a", marker="o", markersize=3, label="hard")
    if label_acc_soft:
        acc_e = [e for e, _ in label_acc_soft]
        acc_v = [a * 100 for _, a in label_acc_soft]
        axes[1].plot(acc_e, acc_v, color="#2a7ae0", marker="s", markersize=3,
                     linestyle="--", label="Bayes")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Label accuracy (%)")
    axes[1].set_title("Cluster label accuracy (eval)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))

    fig.suptitle(f"{exp_name} — {subtitle}", fontsize=12)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_learning_curves", fig, format="svg")
    plt.close(fig)
    logging.info(f"  learning curves → {url}")
    return url
