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


def save_sweep_panels(name, s_values, S_values, solve, median_t, n_seeds):
    """Difficulty surface over (context length S, sparsity s).

    solve / median_t: (n_S, n_s) with NaN for cells outside a row's s-grid. The line
    panel is what actually shows H2 — a heatmap makes non-monotonicity in s hard to
    read off, three lines make it obvious.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    x = np.arange(len(s_values))

    im0 = axes[0].imshow(solve, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set(title=f"solve rate (acc2 > 0.95 within budget, n={n_seeds})",
                xlabel="sparsity s", ylabel="state size S")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    with np.errstate(invalid="ignore"):
        logt = np.log10(median_t)
    im1 = axes[1].imshow(logt, cmap="magma_r", aspect="auto")
    axes[1].set(title="median time-to-emergence, log10 steps (blank = never)",
                xlabel="sparsity s", ylabel="state size S")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes[:2]:
        ax.set_xticks(x, [str(v) for v in s_values])
        ax.set_yticks(np.arange(len(S_values)), [str(v) for v in S_values])

    for i, S in enumerate(S_values):
        m = ~np.isnan(solve[i])
        axes[2].plot(np.array(s_values, float)[m], solve[i][m], "o-", lw=1.4, ms=4, label=f"S={S}")
    axes[2].set(xlabel="sparsity s (log scale)", ylabel="solve rate", xscale="log",
                ylim=(-0.04, 1.04), title="both extremes easy, middle hard?")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def save_ablation_panel(name, base, best, worst, plateau):
    """Per-seed loss with no ablation, with the best-aligned head removed, and with the
    worst-aligned head removed. Log scale — the effect spans orders of magnitude."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(base))
    w = 0.27
    ax.bar(x - w, np.maximum(base, 1e-6), w, label="intact", color="#4a7ebb")
    ax.bar(x, np.maximum(best, 1e-6), w, label="best-aligned head removed", color="#c0504d")
    ax.bar(x + w, np.maximum(worst, 1e-6), w, label="worst-aligned head removed",
           color="#9bbb59")
    ax.axhline(plateau, ls="--", c="k", lw=1, label=f"ln 2 = {plateau:.3f} (no knowledge)")
    ax.set(xlabel="seed", ylabel="second-half CE (nats, log)", yscale="log",
           title="knocking out one head", xticks=x)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def save_attention_maps(name, A, early, final, seed, head, S):
    """Ground-truth support of A beside the same head's attention before and after the
    jump. early/final are (L, L) for the chosen seed and head; rows are restricted to the
    second-half queries and columns to the first half, which is where the support lives."""
    qpos = S - 1 + np.arange(S)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, m, t in zip(axes,
                        [A, early[np.ix_(qpos, np.arange(S))], final[np.ix_(qpos, np.arange(S))]],
                        [f"ground truth: support of A",
                         f"attention, pre-jump (seed {seed}, head {head})",
                         f"attention, end of training"]):
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        ax.set(xlabel="key position (first half)", ylabel="row i of A / query S+i", title=t)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="png", dpi=150)
    plt.close(fig)
    return url


def save_search_space_panel(name, cells):
    """Test the search-space account of difficulty across every sweep cell.

    If difficulty is driven by HOW MANY candidate supports exist per row — C(S,s), maximal
    at s=S/2 and 1 at both extremes — then cells from different S should collapse onto one
    curve against log10 C(S,s). If instead difficulty tracks s itself (XOR arity) or S
    (context length), the S groups will separate.

    cells: list of dicts with S, s, comb, final_loss2, median_t_star, solve_rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    colors = {8: "#4a7ebb", 16: "#c0504d", 32: "#674ea7"}

    for S in sorted({c["S"] for c in cells}):
        grp = sorted((c for c in cells if c["S"] == S), key=lambda c: c["comb"])
        x = [np.log10(c["comb"]) for c in grp]
        axes[0].plot(x, [c["final_loss2"] for c in grp], "o-", color=colors.get(S), lw=1.2,
                     ms=5, label=f"S={S}", alpha=0.85)
        solved = [c for c in grp if c["median_t_star"]]
        if solved:
            axes[1].plot([np.log10(c["comb"]) for c in solved],
                         [c["median_t_star"] for c in solved], "o-", color=colors.get(S),
                         lw=1.2, ms=5, label=f"S={S}", alpha=0.85)

    axes[0].axhline(np.log(2), ls="--", c="k", lw=1, label="ln 2 (no learning)")
    axes[0].set(xlabel="log10 C(S, s) — candidate supports per row",
                ylabel="median final loss2 (nats)",
                title="difficulty vs size of the pattern search space")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].set(xlabel="log10 C(S, s) — candidate supports per row",
                ylabel="median t* (steps, log)", yscale="log",
                title="time-to-emergence, cells that solved")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

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
