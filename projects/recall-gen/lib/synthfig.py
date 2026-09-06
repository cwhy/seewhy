"""Figures and shared prose for the synthetic-prior reports (16, 17, 18).

The primary comparison is the one every report in this project makes: three
networks, identical in size and shape, differing only in the episodes they were
trained on — recall, completion, frozen. The synthetic prior is what they were
trained ON; it is not itself the comparison.

A secondary section asks whether the prior's simplex mode is doing the work,
which is a question about the prior rather than about the networks.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT.parents[1]))

from . import domains, splitfig
from shared_lib.media import save_matplotlib_figure

# The three arms, all trained on the same prior with the same sampler and seed.
#
# These are the d_model=512 runs, not the d_model=256 ones they replace. At 256
# the recall arm identifies at 0.167 on held-out worlds of its own prior, and a
# comparison of training objectives run where no objective produces an ability
# is not a comparison of training objectives. At 512 the same arm reaches 0.540.
RECALL, COMPLETE, FROZEN = "exp49", "exp50", "exp51"
ARMS = [("recall-trained", RECALL, splitfig.BLUE),
        ("completion-trained", COMPLETE, splitfig.ORANGE),
        ("frozen layers", FROZEN, splitfig.GREEN)]
# The same three at half the width, kept for the capacity comparison.
NARROW = {"exp49": "exp45", "exp50": "exp46", "exp51": "exp47"}
# The prior ablation, both recall-trained, both under the earlier sampler.
SIMPLEX, CONT = "exp43", "exp44"
BASE = {SIMPLEX: "synth", CONT: "synth_cont"}
BASE.update({e: "synth" for e in
             ("exp45", "exp46", "exp47", "exp48", "exp49", "exp50", "exp51")})
CHANCE = 1.0 / 16
EVAL_SPREAD = 0.015          # measured over three independent 512-episode draws

PRETTY = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "chess": "chess"}


def near_binary(domain: str, n: int = 4000) -> float:
    """Share of valid coordinates within 0.1 of 0 or 1."""
    _, _, Xte, _ = domains.raw_pools(domain)
    v = Xte[:n][:, domains.valid_vector(domain) > 0.5].ravel()
    return float(((v < 0.1) | (v > 0.9)).mean())


def terms(target: str, den: float) -> str:
    """The glossary each report carries, so none of them needs another open.

    Written out rather than cross-referenced because a reader arriving at report
    18 should not have to read reports 12 through 17 to know what "completion
    training" or "the soft look-up" means.
    """
    p = PRETTY[target]
    return f"""## The terms used here

**An episode.** The network is shown sixteen complete items, one per token, then
a seventeenth with part of it erased. It has to produce the erased part. For a
{p} item the erased part is {"the queenside, files a to d" if target == "chess" else "the bottom fourteen of twenty-eight rows"}.

**The context** is the sixteen items. **The query** is the seventeenth. Two kinds
of episode matter throughout, and they are the two columns of every figure below.
In one the query is a copy of one of the sixteen, so the answer is *present* and
the network can succeed by finding it. In the other the query is not among them,
so the answer is *absent* and the missing part has to be worked out.

**The three networks** are identical in size and shape — 14.95M numbers, four
layers, d_model 512 — and differ only in the episodes they were trained on.

- **Recall-trained**: its answer was always one of the sixteen. Copying always
  worked, so it never had to learn to predict anything.
- **Completion-trained**: its answer was never among the sixteen. Copying was
  never available, so it could only ever predict.
- **Frozen layers**: trained like the recall network, but its four mixing layers
  keep their random starting values forever. Only the input embedding and the
  output head learn — a small fraction of the total.

**The synthetic prior** is what all three were trained on instead of real data.
Each episode samples a fresh **world** — a random low-dimensional generative
model — and draws its sixteen items and its query from that one world. The
network therefore cannot memorise any particular world; it has to work out what
this world is from the sixteen items in front of it. That is the whole idea, and
it is what TabPFN does with tables.

A world is drawn in one of two modes. A **continuous** world produces items whose
coordinates take any value in [0, 1]. A **simplex** world puts exactly one active
coordinate in each group of thirteen, so its items are strictly binary. Forty per
cent of worlds are simplex. That number matters later, because a chess board in
this project *is* sixty-four groups of thirteen with one active coordinate each.

**Three bands** appear in every figure, increasing in novelty downward: items from
worlds seen during training, items from worlds never seen, and real {p} — which
the network has never been shown in any form.

The two synthetic bands are scored on **single-world episodes**, matching how
these networks were trained: all sixteen context items come from one world, which
is what makes "infer this world" a question at all. The real {p} band is scored
on unrelated items, because a real dataset has no world structure to respect.
Scoring the synthetic bands the second way asks a question the networks were
never trained for and reads 1.35 where their own task reads 0.41; an earlier
version of these reports did exactly that.

**Normalised error** is squared error over the erased coordinates, divided by the
error of a fixed reference so that 1.0 means "no better than that reference".
Which reference is stated on each figure. For {p} the reference used here is the
average real {p} item, whose raw error is {den:.4f}.

**Identification accuracy** asks which of the sixteen context items the network's
output most resembles, measured on the erased coordinates only — so a network
that merely copies the visible part cannot score. It is 1/16 = 0.063 at chance.
Its ceiling is 1.000 unless two context items share an erased half, in which case
even a perfect answer can lose the tie-break; the ceiling is reported where it
is not 1.000.

**Two references involve no trained network at all.** *Ridge* is a linear map from
the visible coordinates to the erased ones, fitted on the training pool and
applied blind — it never looks at the context. *Soft look-up* is a
similarity-weighted blend of the sixteen context items, which is the shape of
computation linear attention can actually perform, and is therefore the bar for
whether the context is being used at all."""


def _cell(rows, exp, target, cond, key, ctx="iid"):
    return rows[f"{exp}_stdeval_{BASE[exp]}_to_{target}"]["scores"][ctx][cond][key]


def grid_for(name, target, cfg, mask_rows, tile=0.80, lab_w=1.30):
    """The six-block figure: three novelty bands down, present/absent across."""
    nets = [(lab, splitfig.load_params(e)) for lab, e, _ in ARMS]
    dom = f"{BASE[RECALL]}_to_{target}"
    tr, hd = domains.get(dom).split
    p = PRETTY[target]
    shape_note = ("drawn as 28x28 pictures because that is the shape these 832 "
                  "coordinates are being read in" if target != "chess" else
                  "drawn as boards, the shape these 832 coordinates are read in")
    return splitfig.grid(
        name, dom, mask_rows, tr, hd, cfg, nets,
        ["the prior\nworlds seen in training",
         "the prior\nworlds never seen",
         f"real {p}\nnever seen at all"],
        headline=f"Three networks, one synthetic prior. None was ever shown a {p} item.",
        sub=(f"The prior's own items are {shape_note}; they are not pictures of anything.\n"
             "RED is that network's score over all 512 episodes, against the average TRAINING item."),
        legend=[("recall-trained: during training its answer was ALWAYS one of the sixteen context items, so copying always worked.",
                 splitfig.BLUE),
                ("completion-trained: during training its answer was NEVER in the context, so it could only ever predict.",
                 splitfig.ORANGE),
                ("frozen layers: trained like the recall network, but its four mixing layers keep their random starting values.",
                 splitfig.GREEN)],
        ctx_mode="iid", Q=4, tile=tile, lab_w=lab_w)


def bars_for(name, target, ident, completion, refs):
    """The three arms on one dataset: finding versus predicting.

    `ident` is {(exp, band): id_acc} over bands "prior" and "real"; `completion`
    is {exp: value} already divided by the average real item.
    """
    p = PRETTY[target]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    w = 0.26
    xl = np.arange(2)

    for j, (lab, exp, col) in enumerate(ARMS):
        off = (j - 1) * w
        axes[0].bar(xl + off, [ident[(exp, "prior")], ident[(exp, "real")]],
                    w * 0.9, label=lab, color=col,
                    yerr=EVAL_SPREAD / 2, error_kw=dict(lw=1, capsize=3, ecolor="#444"))
        axes[1].bar([off], [completion[exp]], w * 0.9, label=lab, color=col)

    axes[0].axhline(CHANCE, color="#888", ls="--", lw=1.2)
    axes[0].text(-0.45, CHANCE + 0.012, "chance (1 of 16)", fontsize=7.5, color="#666")
    axes[0].set_xticks(xl)
    axes[0].set_xticklabels(["fresh worlds\nfrom the prior", f"real {p}"], fontsize=9)
    axes[0].set_ylim(0, max(0.4, max(ident.values()) * 1.35))
    axes[0].set_ylabel("identification accuracy", fontsize=9.5)
    axes[0].set_title("Finding an item that IS in the context", fontsize=11)
    axes[0].legend(fontsize=8.2, loc="upper left")

    # Horizontal lines, not markers on one bar: both references apply to all
    # three arms equally, and a marker plotted at x=0 lands on whichever bar
    # happens to be in the middle.
    axes[1].axhline(refs["ridge"], color="#222", ls=":", lw=1.6,
                    label="ridge (never sees the context)")
    axes[1].axhline(refs["soft"], color="#222", ls="-.", lw=1.6,
                    label="soft look-up (context only, no training)")
    axes[1].axhline(1.0, color="#888", ls="--", lw=1.2)
    axes[1].text(-0.47, 1.02, "no better than the average real item",
                 fontsize=7.5, color="#666",
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    axes[1].set_xticks([0.0]); axes[1].set_xticklabels([f"real {p}"], fontsize=9)
    axes[1].set_xlim(-0.5, 0.5)
    top = max(list(completion.values()) + [refs["ridge"], 1.0]) * 1.35
    axes[1].set_ylim(0, top)
    axes[1].set_ylabel("error / error of the average real item", fontsize=9.5)
    axes[1].set_title("Predicting an item that is NOT in the context", fontsize=11)
    axes[1].legend(fontsize=8.2, loc="upper right")

    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def scaling_fig(name, runs, title="Identification against training compute"):
    """Identification on the prior's own worlds, per run.

    `runs` is [(label, steps, d_model, id_train, id_held, colour)].

    Bars rather than a curve against steps: the two 48 000-step runs differ in
    width, not budget, so on a steps axis they land on the same x and cover each
    other. Bars also make the pair that matters — worlds seen against worlds
    never seen — adjacent within each run.

    Read the ENDPOINTS. The optimiser decays its learning rate on a cosine to a
    tenth of the peak over whatever `steps` is set to, so every run flattens over
    its own last decile whatever it has converged to; the 12 000-step run
    flattened at 0.49 and the 48 000-step run at 0.62. That is the schedule, not
    the task.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(len(runs))
    w = 0.36
    for k, (lab, steps, dm, tr, hd, col) in enumerate(runs):
        ax.bar(k - w / 2, tr, w * 0.92, color=col,
               label="worlds seen in training" if k == 0 else None)
        ax.bar(k + w / 2, hd, w * 0.92, color=col, alpha=0.5, hatch="//",
               edgecolor="white",
               label="worlds never seen" if k == 0 else None)
        ax.text(k, max(tr, hd) + 0.03, f"{hd:.2f}", ha="center", fontsize=8.5,
                color="#333")
    ax.axhline(CHANCE, color="#888", ls="--", lw=1.1)
    ax.text(-0.45, CHANCE + 0.015, "chance (1 of 16)", fontsize=7.5, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace(", ", "\n") for lab, *_ in runs], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("identification accuracy", fontsize=9.5)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.4, loc="upper left")
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def capacity_fig(name, runs, refs, chance_lab=True):
    """One recall network at four sizes: what scales and what does not.

    `runs` is [(label, colour, {band: id}, {band: nmse}, absent_nmse)] in
    increasing order of compute. Left panel is finding, split by novelty band;
    right panel is predicting, with the references that bound it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    x = np.arange(len(runs))
    bands = [("worlds seen in training", "A", 1.0),
             ("worlds never seen", "E", 0.62),
             ("real MNIST", "B", 0.34)]
    w = 0.26
    for j, (blab, key, alpha) in enumerate(bands):
        axes[0].bar(x + (j - 1) * w, [r[2][key] for r in runs], w * 0.9,
                    color=[r[1] for r in runs], alpha=alpha, label=blab,
                    edgecolor="white", linewidth=0.6)
    axes[0].axhline(CHANCE, color="#888", ls="--", lw=1.1)
    if chance_lab:
        axes[0].text(-0.45, CHANCE + 0.015, "chance (1 of 16)", fontsize=7.5,
                     color="#666")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("identification accuracy", fontsize=9.5)
    axes[0].set_title("Finding: climbs with capacity", fontsize=11)
    axes[0].legend(fontsize=8.2, loc="upper left")

    axes[1].plot(x, [r[4] for r in runs], "-o", color="#c1121f", lw=2, ms=8,
                 zorder=4, label="the recall network")
    axes[1].axhline(refs["ridge"], color="#222", ls=":", lw=1.6,
                    label="ridge (never sees the context)")
    axes[1].axhline(refs["soft"], color="#222", ls="-.", lw=1.6,
                    label="soft look-up (context only, no training)")
    axes[1].axhline(1.0, color="#888", ls="--", lw=1.2)
    axes[1].text(-0.45, 1.02, "no better than the average real item",
                 fontsize=7.5, color="#666",
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    axes[1].set_ylim(0, max([r[4] for r in runs] + [refs["ridge"], 1.1]) * 1.25)
    axes[1].set_ylabel("error / error of the average real item", fontsize=9.5)
    axes[1].set_title("Predicting: does not", fontsize=11)
    axes[1].legend(fontsize=8.2, loc="lower right")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([r[0].replace(", ", "\n") for r in runs], fontsize=9)
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def context_fig(name, per_ctx, refs_by_ctx, margins):
    """The same network under two kinds of context.

    `per_ctx` is {ctx: (id_B, nmse_D)}; `margins` is {ctx: median nearest rival}.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ctxs = ["iid", "knn"]
    names = ["sixteen unrelated\nimages",
             "the query's sixteen\nnearest neighbours"]
    cols = [splitfig.BLUE, splitfig.GREEN]
    x = np.arange(2)
    axes[0].bar(x, [per_ctx[c][0] for c in ctxs], 0.5, color=cols)
    for k, c in enumerate(ctxs):
        axes[0].text(k, per_ctx[c][0] + 0.012,
                     f"{per_ctx[c][0]:.3f}\nrival at {margins[c]:.2f}",
                     ha="center", fontsize=8.2)
    axes[0].axhline(CHANCE, color="#888", ls="--", lw=1.1)
    axes[0].set_ylim(0, max(per_ctx[c][0] for c in ctxs) * 1.6)
    axes[0].set_ylabel("identification accuracy", fontsize=9.5)
    axes[0].set_title("A closer context is harder to name", fontsize=11)

    axes[1].bar(x, [per_ctx[c][1] for c in ctxs], 0.5, color=cols)
    for k, c in enumerate(ctxs):
        axes[1].plot([k - 0.25, k + 0.25], [refs_by_ctx[c]] * 2, "k-.", lw=1.6,
                     label="soft look-up ceiling" if k == 0 else None)
        axes[1].text(k, per_ctx[c][1] + 0.02, f"{per_ctx[c][1]:.3f}",
                     ha="center", fontsize=8.5)
    axes[1].axhline(1.0, color="#888", ls="--", lw=1.2)
    axes[1].set_ylim(0, max(per_ctx[c][1] for c in ctxs) * 1.4)
    axes[1].set_ylabel("error / error of the average real item", fontsize=9.5)
    axes[1].set_title("...and easier to reconstruct", fontsize=11)
    axes[1].legend(fontsize=8.2, loc="lower right")

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def ablation_fig(name, points, highlight=None):
    """Secondary: does the prior's simplex mode carry the transfer?

    `points` is [(label, near_binary, simplex advantage)]. Three datasets is
    three points; this shows an ordering, not a fit, and the reports say so.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for lab, x, y in points:
        col = splitfig.BLUE if lab == highlight else "#8a8a8a"
        ax.scatter([x], [y], s=150 if lab == highlight else 90, color=col,
                   zorder=4, edgecolor="#222", linewidth=0.8)
        ax.annotate(f"{lab}\n{y:+.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 16 if y >= 0 else -30), ha="center", fontsize=9,
                    fontweight="bold" if lab == highlight else "normal", color=col)
    ax.axhline(0.0, color="#888", ls="--", lw=1.2)
    ax.text(0.515, 0.015, "the simplex mode makes no difference",
            fontsize=7.8, color="#666")
    ax.set_xlabel("share of the dataset's coordinates within 0.1 of 0 or 1\n"
                  "(a simplex world is exactly 1.00; a continuous world, 0.20)",
                  fontsize=9.5)
    ax.set_ylabel("gain in identification from the simplex mode", fontsize=9.5)
    ax.set_xlim(0.5, 1.06)
    ax.set_ylim(-0.16, 0.78)
    ax.set_title("A recall network transfers as far as the prior resembles the target",
                 fontsize=11)
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url
