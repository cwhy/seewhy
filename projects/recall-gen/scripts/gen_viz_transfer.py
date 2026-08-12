"""Figures for the transfer evaluations (scripts/eval_transfer.py).

  1. length_transfer   the same three models run at context sizes they never
                       trained at. Two panels, because identification accuracy
                       and completion error are different units.
  2. dataset_transfer  the scope test: how far the learned retrieval mechanism
                       travels off MNIST. A strip of sample images from each
                       pool sits above the bars, because "shuffled" and "noise"
                       mean nothing as words.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/gen_viz_transfer.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

from lib.core import PIX, SIDE
from lib.train import Run, build_pools
from eval_transfer import synthetic_pools
from shared_lib.media import save_matplotlib_figure

# dataviz reference palette, light surface. validate_palette.js on
# "#2a78d6,#eb6834,#1baf7a" passes every check; the aqua slot carries a contrast
# WARN, which direct-labelling the lines discharges.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e3e2df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

MODEL_STYLE = {
    "recall_M16":    ("trained on recall, 16 in context", BLUE),
    "recall_M256":   ("trained on recall, 256 in context", ORANGE),
    "complete_best": ("trained to complete", AQUA),
}


def rows():
    out = {}
    for line in (PROJECT / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["experiment"], r)
    return out


def length_transfer(R, name="recallgen_length_transfer"):
    """Three panels, because the first one on its own is misleading.

    Identification accuracy has a chance level of 1/M, which moves by a factor
    of 64 across this axis: 0.25 at M=4 and 0.0039 at M=256. An earlier version
    of this figure drew a single chance line and made the fall from 1.000 to
    0.322 look like a collapse, when 0.322 against a chance of 0.0039 is 82x
    chance. The chance curve is now plotted as its own series on a log axis, and
    `gain` — which needs no chance level — sits beside it as the metric that
    actually says how much retrieval is left.
    """
    t = R["transfer_length"]
    Ms = [int(m) for m in t["lengths"]]
    x = np.arange(len(Ms))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for mk, (label, colour) in MODEL_STYLE.items():
        met = lambda M, c, k: t["transfer"][mk][str(M)]["metrics"][c][k]
        idb = [met(M, "B_novel_present", "id_acc") for M in Ms]
        gain = [t["transfer"][mk][str(M)]["gain"] for M in Ms]
        d = [met(M, "D_novel_absent", "nmse") for M in Ms]
        j = Ms.index(t["trained_at"][mk])
        for ax, ys in zip(axes, (idb, gain, d)):
            ax.plot(x, ys, color=colour, lw=2, marker="o", ms=7, zorder=3,
                    label=label)
            ax.plot([x[j]], [ys[j]], marker="o", ms=13, mfc="none", mec=colour,
                    mew=2, zorder=4)

    ax = axes[0]
    ax.plot(x, [1 / m for m in Ms], color=MUTED, lw=1.4, ls=(0, (4, 3)),
            marker="o", ms=5, zorder=2, label="chance (1/M)")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 2.0)
    ax.set_ylabel("identification accuracy (log scale)")
    ax.set_title("does the output land on the right\nspecific context image?",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=7.8)

    ax = axes[1]
    ax.axhline(0.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("zero = the context is not being read", xy=(0.98, 0.0),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=MUTED, ha="right")
    ax.set_ylim(-0.12, 0.95)
    ax.set_ylabel("gain — what the answer\nbeing present is worth")
    ax.set_title("is it retrieving at all?", fontsize=9.5, color=INK, loc="left")

    ax = axes[2]
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("predict the average digit", xy=(0.98, 1.0),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=MUTED, ha="right")
    ax.set_ylim(0.3, 1.12)
    ax.set_ylabel("completion error, answer absent")
    ax.set_title("and how well does it complete?", fontsize=9.5, color=INK,
                 loc="left")
    ax.annotate("ring = the size each\nmodel trained at", xy=(0.03, 0.05),
                xycoords="axes fraction", fontsize=8, color=MUTED)

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([str(m) for m in Ms])
        ax.set_xlabel("context images at TEST time")
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="png", dpi=170)
    plt.close(fig)
    return url


def dataset_transfer(R, name="recallgen_dataset_transfer"):
    t = R["transfer_dataset"]["transfer"]["recall_M16"]
    tags = ["mnist", "fashion", "shuffled", "noise"]
    nice = ["held-out\nMNIST", "Fashion-\nMNIST", "MNIST, pixels\npermuted",
            "random\nfields"]
    idb = [t[k]["metrics"]["B_novel_present"]["id_acc"] for k in tags]

    pools, _ = build_pools(Run(exp_name="", name="", M=16))
    fashion, perm, noise = synthetic_pools()
    samples = {"mnist": pools["held"][3], "fashion": fashion[3],
               "shuffled": pools["held"][3][perm], "noise": noise[3]}

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, len(tags), height_ratios=[1.0, 2.4],
                          hspace=0.30, wspace=0.22,
                          top=0.99, bottom=0.13, left=0.115, right=0.99)
    for j, k in enumerate(tags):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(samples[k].reshape(SIDE, SIDE), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    xs = np.arange(len(tags))
    ax.bar(xs, idb, width=0.55, color=BLUE, zorder=3)
    ax.axhline(1 / 16, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    for xi, v in zip(xs, idb):
        ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(nice, fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("identification accuracy\n(dashed line = chance, 0.063)")
    ax.set_xlabel("what the 16 context images are")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    url = save_matplotlib_figure(name, fig, format="png", dpi=170)
    plt.close(fig)
    return url


# ── pixel figures: what the transfer numbers actually look like ───────────────

import pickle
import jax
import jax.numpy as jnp
from lib.core import Cfg, predict, row_mask
from lib import evalsets
from eval_transfer import MODELS, CFG, load

MASK = row_mask(14)
HID = MASK > 0.5


def _panel(ax, img, mask=MASK):
    ax.imshow(np.asarray(img).reshape(SIDE, SIDE), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _cols_at_percentiles(err, k, pcts=(15, 40, 65, 90)):
    order = np.argsort(err)
    return [order[int(round(q / 100 * (len(order) - 1)))] for q in pcts[:k]]


def retrieval_across_pools(n=4, name="recallgen_retrieval_pools"):
    """What identification accuracies of 1.000 / 0.651 / 0.116 / 0.222 look like.

    Condition B throughout: the query's true image IS one of the 16 in the
    context, so a working retriever reproduces it exactly. One recall-trained
    model (trained on MNIST at M=16), four pools.
    """
    base = Run(exp_name="", name="", M=16, mask_rows=14, cfg=CFG)
    mnist_pools, _ = build_pools(base)
    fashion, perm, noise = synthetic_pools()
    variants = {
        "held-out MNIST": mnist_pools["held"],
        "Fashion-MNIST": fashion,
        "MNIST, pixels permuted": mnist_pools["held"][:, perm],
        "random fields": noise,
    }
    p = load(MODELS["recall_M16"])

    blocks = []
    for title, pool in variants.items():
        pools = {"train": mnist_pools["train"], "held": pool, "held_same": pool}
        mean_img = pool.mean(0)
        ev = evalsets.build(pools, MASK, 16, 4, 64, mean_img, seed=999)
        es = ev["B_novel_present"]
        pred = np.asarray(predict(p, es.ctx[:64], es.qry[:64], jnp.array(MASK),
                                  CFG))[:, 0]
        truth = np.asarray(es.qry[:64, 0])
        err = ((pred - truth) ** 2)[:, HID].mean(1) / \
              ((mean_img[None] - truth) ** 2)[:, HID].mean(1)
        blocks.append((title, truth, pred, err))

    ncol = 3 * len(blocks) + (len(blocks) - 1)          # 3 per pool + 1 gap
    widths = []
    for i in range(len(blocks)):
        widths += [1, 1, 1] + ([0.42] if i < len(blocks) - 1 else [])
    fig = plt.figure(figsize=(sum(widths) * 0.86, n * 0.94 + 0.7))
    gs = fig.add_gridspec(n, ncol, width_ratios=widths, hspace=0.06, wspace=0.06,
                          top=0.86, bottom=0.01, left=0.005, right=0.995)

    for b, (title, truth, pred, err) in enumerate(blocks):
        c0 = b * 4
        idx = _cols_at_percentiles(err, n)
        for r, t in enumerate(idx):
            imgs = [truth[t] * (1 - MASK) + 0.55 * MASK,
                    truth[t] * (1 - MASK) + pred[t] * MASK,
                    truth[t]]
            for c, img in enumerate(imgs):
                ax = fig.add_subplot(gs[r, c0 + c])
                _panel(ax, img)
                if r == 0 and b == 0:
                    ax.set_title(["input", "model", "truth"][c], fontsize=7.5,
                                 color=MUTED, pad=3)
                elif r == 0:
                    ax.set_title(["input", "model", "truth"][c], fontsize=7.5,
                                 color=MUTED, pad=3)
        fig.add_subplot(gs[0, c0 + 1]).set_axis_off()
        fig.text((gs[0, c0].get_position(fig).x0 + gs[0, c0 + 2].get_position(fig).x1) / 2,
                 0.945, title, ha="center", va="bottom", fontsize=9.5,
                 color=INK, fontweight="bold")
    url = save_matplotlib_figure(name, fig, format="png", dpi=170)
    plt.close(fig)
    return url


def completion_across_lengths(n=5, seed=11, name="recallgen_completion_lengths"):
    """The SAME five queries completed by the SAME model at four context sizes.

    Only the number of context images changes between rows; the model, the
    queries and the mask are identical. The answer is in none of the contexts.
    """
    base = Run(exp_name="", name="", M=16, mask_rows=14, cfg=CFG)
    pools, _ = build_pools(base)
    held = pools["held"]
    mean_img = pools["train"].mean(0)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(held), 64 + 256, replace=False)
    targets, ctx_pool = held[pick[:64]], held[pick[64:]]
    p = load(MODELS["recall_M16"])

    rows_out = []
    for M in (4, 16, 64, 256):
        ctx = jnp.array(np.broadcast_to(ctx_pool[:M], (64, M, PIX)))
        pred = np.concatenate([
            np.asarray(predict(p, ctx[i:i + 8], jnp.array(targets[i:i + 8])[:, None, :],
                               jnp.array(MASK), CFG))[:, 0]
            for i in range(0, 64, 8)])
        err = ((pred - targets) ** 2)[:, HID].mean(1) / \
              ((mean_img[None] - targets) ** 2)[:, HID].mean(1)
        rows_out.append((M, pred, err))

    avg = np.mean([e for _, _, e in rows_out], axis=0)
    idx = [np.argsort(avg)[int(round(q / 100 * (len(avg) - 1)))]
           for q in (10, 30, 50, 70, 90)][:n]

    labels = ["input", "truth"] + [f"{M} in context" for M, _, _ in rows_out]
    fig, axes = plt.subplots(len(labels), n, figsize=(n * 1.18, len(labels) * 1.24))
    for j, t in enumerate(idx):
        truth = targets[t]
        for r, lab in enumerate(labels):
            if lab == "input":
                img = truth * (1 - MASK) + 0.55 * MASK
            elif lab == "truth":
                img = truth
            else:
                _, pred, err = rows_out[r - 2]
                img = truth * (1 - MASK) + pred[t] * MASK
                axes[r, j].set_xlabel(f"{err[t]:.2f}", fontsize=7.5, color=MUTED,
                                      labelpad=1)
            _panel(axes[r, j], img)
            if j == 0:
                axes[r, j].set_ylabel(lab, fontsize=8, color=INK2, rotation=0,
                                      ha="right", va="center", labelpad=8)
    fig.suptitle("one model, one set of queries, four context sizes\n"
                 "(the answer is in none of them)",
                 fontsize=10, color=INK, fontweight="bold", x=0.30, ha="left",
                 y=0.995)
    fig.subplots_adjust(hspace=0.30, wspace=0.06, top=0.90, bottom=0.01,
                        left=0.27, right=0.99)
    url = save_matplotlib_figure(name, fig, format="png", dpi=170)
    plt.close(fig)
    return url


if __name__ == "__main__":
    R = rows()
    print("length_transfer  ", length_transfer(R))
    print("dataset_transfer ", dataset_transfer(R))
    print("retrieval_pools  ", retrieval_across_pools())
    print("completion_lens  ", completion_across_lengths())
