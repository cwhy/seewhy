"""Visualisation for sparse-attn-emergence. SVG for curves, per the workflow."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared_lib.media import save_matplotlib_figure


def save_seed_curves(name, steps, loss2, acc2, plateau, thresh):
    """Per-seed second-half loss and accuracy. loss2/acc2: (n_seeds, n_points)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i in range(loss2.shape[0]):
        axes[0].plot(steps, loss2[i], lw=0.8, alpha=0.75)
        axes[1].plot(steps, acc2[i], lw=0.8, alpha=0.75)

    axes[0].axhline(plateau, ls="--", c="k", lw=1, label=f"plateau = {plateau:.3f}")
    axes[0].set(xlabel="step", ylabel="second-half CE (nats)",
                title=f"loss2 per seed (n={loss2.shape[0]})")
    axes[0].legend(fontsize=8)

    axes[1].axhline(thresh, ls="--", c="k", lw=1, label=f"threshold = {thresh}")
    axes[1].set(xlabel="step", ylabel="second-half accuracy", title="acc2 per seed",
                ylim=(0.45, 1.02))
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def save_tstar_hist(name, tstars, total_steps, thresh):
    """Histogram of time-to-emergence; censored (never-emerged) seeds annotated."""
    hit = [t for t in tstars if t is not None]
    n_cens = len(tstars) - len(hit)

    fig, ax = plt.subplots(figsize=(6, 4))
    if hit:
        ax.hist(hit, bins=min(16, max(3, len(hit))), color="#4a7ebb", edgecolor="k", lw=0.5)
    ax.set(xlabel=f"step of first acc2 > {thresh}", ylabel="seeds",
           xlim=(0, total_steps),
           title=f"time-to-emergence — {len(hit)}/{len(tstars)} emerged, {n_cens} censored")
    if hit:
        ax.axvline(float(np.median(hit)), ls="--", c="r", lw=1,
                   label=f"median {np.median(hit):.0f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def save_mechanism_panel(name, dsteps, iou_max, ent_min, loss2_at_dsteps, sparsity):
    """Attention-support IoU and min-head entropy alongside loss, all per seed.

    Arrays are (n_seeds, n_diag_points). If the mechanism story holds, the IoU
    rise and the entropy drop sit at the same step as each seed's loss drop.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(iou_max.shape[0]):
        axes[0].plot(dsteps, iou_max[i], lw=0.9, alpha=0.75)
        axes[1].plot(dsteps, ent_min[i], lw=0.9, alpha=0.75)
        axes[2].plot(dsteps, loss2_at_dsteps[i], lw=0.9, alpha=0.75)

    axes[0].set(xlabel="step", ylabel=f"IoU(top-{sparsity} attention, row support)",
                title="best head vs ground-truth support", ylim=(-0.02, 1.02))
    axes[1].set(xlabel="step", ylabel="attention entropy (nats)",
                title="most-peaked head")
    axes[2].set(xlabel="step", ylabel="second-half CE (nats)", title="loss2 (same steps)")

    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url
