"""Figures for the "two senses of generalisation" report.

The project's other figures answer "what happened". These answer one question:
*does training on recall produce generalisation?* — which needs the two things
the word can mean to be shown side by side, because that ambiguity is what makes
the result read as contradictory.

Four figures:
  1. two_senses   the hero, pixels: on the SAME unseen digits, retrieval is
                  perfect and completion is worth nothing
  2. where_it_sits  where the recall-trained model lands among the reference
                  strategies on condition D
  3. sweep        gain and completion error against context size, one axis
  4. digit_split  identification accuracy and completion error as novelty
                  increases, as two panels (different units — never one axis)

Usage:
    uv run --no-sync python projects/recall-gen/scripts/gen_viz_twosenses.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

import jax.numpy as jnp
from lib.core import Cfg, predict, row_mask, SIDE, PIX
from lib import evalsets
from lib.train import Run, build_pools
from shared_lib.media import save_matplotlib_figure

# Validated categorical palette (dataviz reference instance, light surface).
# scripts/validate_palette.js "#2a78d6,#eb6834" --mode light -> all checks pass.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID = "#e3e2df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def rows():
    out = {}
    for line in (PROJECT / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["experiment"], r)
    return out


# ── figure 1: the hero, in pixels ─────────────────────────────────────────────

def two_senses(n=5, pcts=(10, 30, 50, 70, 90)):
    """exp8 (trained on digits 0-4) shown on digits 5-9 it has never seen.

    Top block: the answer is in the context -> the model reproduces it exactly.
    Bottom block: the answer is absent.

    Columns are chosen at FIXED PERCENTILES of the per-sample error within each
    block, not taken in file order. This is not fussiness: the first six samples
    of condition D score 0.88 0.67 0.81 1.04 0.60 0.59 against a median of 1.03,
    so showing them in order flatters the model badly. Each column is labelled
    with its own error.

    The absent block also carries an "average digit" row — the thing that scores
    exactly 1.0 — because the interesting fact is not that the model's output is
    ugly. It is that the output is *sharp and confident and wrong*, and averages
    out no better than a blur.
    """
    rn = Run(exp_name="exp8", name="", M=16, Q=4, mask_rows=14,
             train_digits=(0, 1, 2, 3, 4), held_digits=(5, 6, 7, 8, 9),
             cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20))
    pools, _ = build_pools(rn)
    mask_np = row_mask(rn.mask_rows)
    mask = jnp.array(mask_np)
    mean_img = pools["train"].mean(0)
    ev = evalsets.build(pools, mask_np, rn.M, rn.Q, 512, mean_img,
                        conditions=__import__("experiments8").SPLIT_CONDITIONS)

    with open(PROJECT / "params_exp8.pkl", "rb") as f:
        p = jnp_tree(pickle.load(f))

    hid = mask_np > 0.5
    blocks = []
    for cond, title, show_mean in (
            ("B_novel_present", "the answer IS in the context", False),
            ("D_novel_absent", "the answer is NOT in the context", True)):
        es = ev[cond]
        pred = np.asarray(predict(p, es.ctx[:256], es.qry[:256], mask, rn.cfg))[:, 0]
        truth = np.asarray(es.qry[:256, 0])
        err = ((pred - truth) ** 2)[:, hid].mean(1) / \
              ((mean_img[None] - truth) ** 2)[:, hid].mean(1)
        order = np.argsort(err)
        idx = [order[int(round(q / 100 * (len(order) - 1)))] for q in pcts]
        blocks.append((title, truth[idx], pred[idx], err[idx], show_mean))

    # An explicit spacer row between the blocks. Without it the second block's
    # title and column labels are drawn on top of the first block's last row —
    # annotations are anchored to their axes, so hspace cannot fix it.
    heights = [1, 1, 1, 0.62, 1, 1, 1, 1]
    fig = plt.figure(figsize=(n * 1.15, sum(heights) * 1.20))
    gs = fig.add_gridspec(len(heights), n, height_ratios=heights,
                          hspace=0.10, wspace=0.06,
                          top=0.935, bottom=0.015, left=0.155, right=0.99)
    row = 0
    for title, truth, pred, err, show_mean in blocks:
        rows_here = ["input", "model", "truth"] + (["average\ndigit"] if show_mean else [])
        for j in range(n):
            vis = truth[j] * (1 - mask_np) + 0.55 * mask_np
            comp = truth[j] * (1 - mask_np) + pred[j] * mask_np
            avg = truth[j] * (1 - mask_np) + mean_img * mask_np
            imgs = [vis, comp, truth[j]] + ([avg] if show_mean else [])
            for r, img in enumerate(imgs):
                ax = fig.add_subplot(gs[row + r, j])
                ax.imshow(img.reshape(SIDE, SIDE), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if j == 0:
                    ax.set_ylabel(rows_here[r], fontsize=8, color=INK2, rotation=0,
                                  ha="right", va="center", labelpad=8)
                if r == 0:
                    ax.set_title(f"error {err[j]:.2f}", fontsize=8, color=MUTED, pad=6)
                if r == 0 and j == 0:
                    ax.annotate(title, xy=(0, 1.62), xycoords="axes fraction",
                                fontsize=10.5, color=INK, fontweight="bold",
                                ha="left", va="bottom")
        row += len(rows_here) + 1        # +1 for the spacer

    url = save_matplotlib_figure("recallgen_two_senses", fig, format="png", dpi=170)
    plt.close(fig)
    return url


def jnp_tree(p):
    import jax
    return jax.tree_util.tree_map(jnp.asarray, p)


# ── figure 2: where the model sits among the reference strategies ─────────────

def where_it_sits(R):
    base = R["baselines_M16_r14"]["baselines"]["D_novel_absent"]
    items = [
        ("copy the closest context image", base["n_nn1"], "reference"),
        ("best soft look-up from context", base["n_knn"], "reference"),
        ("predict the average digit", 1.000, "reference"),
        ("recall-trained model", R["exp1"]["final"]["D_novel_absent"]["nmse"], "model"),
        ("linear regression, no context", base["n_ridge"], "reference"),
        ("trained to complete (best)", min(R["exp2"]["history"]["nmse"]["D_novel_absent"]),
         "reference"),
    ]
    items.sort(key=lambda t: -t[1])
    labels = [t[0] for t in items]
    vals = [t[1] for t in items]
    cols = [ORANGE if t[2] == "model" else BLUE for t in items]

    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    y = np.arange(len(items))
    ax.barh(y, vals, height=0.62, color=cols, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(0, 1.72)
    ax.set_xlabel("error completing an unseen digit  (1.0 = predict the average digit)")
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.3f}", xy=(v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    for lab, yi in zip(labels, y):
        if lab == "recall-trained model":
            ax.get_yticklabels()[yi].set_color(INK)
            ax.get_yticklabels()[yi].set_fontweight("bold")
    fig.tight_layout()
    url = save_matplotlib_figure("recallgen_where_it_sits", fig, format="png", dpi=170)
    plt.close(fig)
    return url


# ── figure 3: the sweep, both series in the same units, one axis ─────────────

def sweep(R):
    exps = {4: "exp6", 16: "exp1", 64: "exp4", 256: "exp5"}
    Ms = sorted(exps)
    gain, comp = [], []
    for m in Ms:
        f = R[exps[m]]["final"]
        gain.append(f["D_novel_absent"]["nmse"] - f["B_novel_present"]["nmse"])
        comp.append(f["D_novel_absent"]["nmse"])

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    x = np.arange(len(Ms))
    ax.plot(x, gain, color=BLUE, lw=2, marker="o", ms=8, zorder=3,
            label="how much the answer being present is worth")
    ax.plot(x, comp, color=ORANGE, lw=2, marker="o", ms=8, zorder=3,
            label="error completing an unseen digit")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("no better than ignoring the input", xy=(0.985, 1.0),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=MUTED, ha="right")
    ax.annotate("still retrieving", xy=(x[0], gain[0]), xytext=(10, -26),
                textcoords="offset points", fontsize=9, color=BLUE,
                fontweight="bold", ha="left")
    ax.annotate("not retrieving at all", xy=(x[-1], gain[-1]), xytext=(-8, 20),
                textcoords="offset points", fontsize=9, color=BLUE,
                fontweight="bold", ha="right")
    # Endpoints only — the two series sit on top of each other at M=4 and M=16,
    # and a number on every point is noise when the story is the divergence.
    for xi, v, dy in ((x[0], gain[0], -16), (x[-1], gain[-1], -16)):
        ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    for xi, v in ((x[0], comp[0]), (x[-1], comp[-1])):
        ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel("context images per episode")
    ax.set_ylabel("normalised MSE")
    ax.set_ylim(-0.09, 1.14)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.03, 0.06))
    fig.tight_layout()
    url = save_matplotlib_figure("recallgen_sweep_two_senses", fig, format="png", dpi=170)
    plt.close(fig)
    return url


# ── figure 4: the digit split, two panels (two units, never one axis) ────────

def digit_split(R):
    f8 = R["exp8"]["final"]
    xs = ["nothing", "the images", "the images\nand the classes"]
    ident = [f8["A_seen_present"]["id_acc"], f8["E_same_present"]["id_acc"],
             f8["B_novel_present"]["id_acc"]]
    comp = [f8["C_seen_absent"]["nmse"], f8["F_same_absent"]["nmse"],
            f8["D_novel_absent"]["nmse"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    x = np.arange(3)

    ax = axes[0]
    ax.bar(x, ident, width=0.55, color=BLUE, zorder=3)
    ax.axhline(1 / 16, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("identification accuracy\n(dashed line = chance, 0.063)")
    ax.set_title("finding it  (higher is better)", fontsize=9.5, color=INK, loc="left")
    for xi, v in zip(x, ident):
        ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)

    ax = axes[1]
    ax.bar(x, comp, width=0.55, color=ORANGE, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("completion error\n(dashed line = predict the average digit)")
    ax.set_title("completing it  (lower is better)", fontsize=9.5, color=INK, loc="left")
    for xi, v in zip(x, comp):
        ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=8.5)
        ax.set_xlabel("what the model has never seen before")
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    url = save_matplotlib_figure("recallgen_digit_split", fig, format="png", dpi=170)
    plt.close(fig)
    return url


if __name__ == "__main__":
    R = rows()
    print("two_senses    ", two_senses())
    print("where_it_sits ", where_it_sits(R))
    print("sweep         ", sweep(R))
    print("digit_split   ", digit_split(R))
