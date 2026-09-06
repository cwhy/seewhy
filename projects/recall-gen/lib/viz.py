"""Matplotlib helpers for Recall-Gen. Pixel figures only — PNG, dpi=150."""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

from . import domains


def _tile(ax, v, domain, mask=None, hidden=False, title=None):
    domains.draw(ax, v, domain, mask=mask, hidden_shading=hidden)
    if title:
        ax.set_title(title, fontsize=7)


def completion_grid(name: str, rows: list[dict], mask, n_show=6,
                    domain: str = "mnist") -> str:
    """One block per condition. Columns: query input, model completion, truth,
    look-up baseline. `rows` is a list of dicts with those keys plus `label`.

    The model column is a COMPOSITE — the query's visible coordinates with the
    model's prediction pasted into the hole. The head emits every coordinate but
    the loss only ever scores the hidden ones, so the model's visible half is
    unconstrained noise and showing it raw makes every completion look broken.
    """
    mask = np.asarray(mask)
    ncols = 4
    scale = 1.05 if domains.get(domain).kind == "image" else 1.35
    fig, axes = plt.subplots(len(rows) * n_show, ncols,
                             figsize=(ncols * scale, len(rows) * n_show * scale))
    for r, row in enumerate(rows):
        for j in range(n_show):
            i = r * n_show + j
            truth = np.asarray(row["qry"][j])
            comp = domains.composite(truth, row["pred"][j], mask)
            _tile(axes[i, 0], truth, domain, mask=mask, hidden=True,
                  title=f"{row['label']}: input" if j == 0 else None)
            _tile(axes[i, 1], comp, domain, title="model" if j == 0 else None)
            _tile(axes[i, 2], truth, domain, title="truth" if j == 0 else None)
            _tile(axes[i, 3], row["nn"][j], domain,
                  title="look-up" if j == 0 else None)
        axes[r * n_show, 0].set_ylabel(row["label"], fontsize=7)
    fig.tight_layout(pad=0.2)
    url = save_matplotlib_figure(name, fig, format="png", dpi=150)
    plt.close(fig)
    return url


def learning_curves(name: str, history: dict, baselines: dict) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    steps = history["step"]
    for cond, ys in history["nmse"].items():
        ax[0].plot(steps, ys, label=cond)
    for cond, v in baselines.items():
        ax[0].axhline(v, ls=":", lw=0.8, color="grey")
    ax[0].axhline(1.0, ls="--", lw=1, color="k")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("MSE / mean-image MSE")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7); ax[0].set_title("normalised MSE")

    for cond, ys in history["id_acc"].items():
        ax[1].plot(steps, ys, label=cond)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("identification accuracy")
    ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=7); ax[1].set_title("retrieval identification")
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url
