"""Report 12: train on digits 0-4, test on 5-9.

Everything here is evaluation-only. exp8 (recall training) and exp9 (completion
training) were both trained on MNIST digits 0-4 and already exist; this script
re-scores nothing and only draws what those checkpoints do.

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_12.py
"""
import json
import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.core import Cfg, row_mask
from lib import evalsets
from lib.train import Run, build_pools, make_eval
from shared_lib.media import save_matplotlib_figure
from shared_lib.report import save_report

REPORT_MD_PATH = PROJECT_DIR / "reports" / "12-new-digits.md"
PROJ = "recall-gen"
CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20)

rows = {}
for line in open(PROJECT_DIR / "results.jsonl"):
    r = json.loads(line)
    rows[r["experiment"]] = r

SPLIT = {
    "A_seen_present":  ("train",     True),
    "B_novel_present": ("held",      True),
    "C_seen_absent":   ("train",     False),
    "D_novel_absent":  ("held",      False),
    "E_same_present":  ("held_same", True),
    "F_same_absent":   ("held_same", False),
}
# (row label, present condition, absent condition) — increasing novelty downward
BANDS = [
    ("digits 0-4\nimages seen in training", "A_seen_present", "C_seen_absent"),
    ("digits 0-4\nimages never seen",       "E_same_present", "F_same_absent"),
    ("digits 5-9\nnever seen at all",       "B_novel_present", "D_novel_absent"),
]


def _load(name):
    with open(PROJECT_DIR / f"params_{name}.pkl", "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def _soft_lookup(ctx, qry, mask_j, tau):
    vis = 1.0 - mask_j
    d = (((qry[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1) / vis.sum()
    return jnp.einsum("eqm,emp->eqp", jax.nn.softmax(-d / tau, -1), ctx)


def build():
    rn = Run(exp_name="", name="", M=16, Q=4, mask_rows=14, cfg=CFG,
             train_digits=(0, 1, 2, 3, 4), held_digits=(5, 6, 7, 8, 9),
             conditions=SPLIT)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    ev = evalsets.build(pools, mask, 16, 4, 512, pools["train"].mean(0),
                        conditions=SPLIT, labels=labels)
    return rn, mask, ev


def fig_grid():
    """Six blocks: three levels of novelty down, answer present/absent across.

    Panel numbers are NORMALISED error: the panel's squared error over the hidden
    pixels, divided by the error of drawing the average training image for that
    same pool. Raw per-pixel error would not be comparable down the figure, since
    each pool has its own normaliser. 1.00 means the panel is no better than the
    average digit.

    Columns inside a block are three episodes at fixed percentiles (15th, 50th,
    85th) of the blend look-up's own per-episode error on the ABSENT condition of
    that same novelty band. That ranking uses no trained network, and because the
    present and absent conditions of a band share their queries, a column is the
    same query in both blocks of its row. Predicted panels composite the true
    visible half back over the predicted hidden half and carry their own squared
    error on the hidden pixels.
    """
    rn, mask, ev = build()
    mask_j = jnp.array(mask)
    eval_fn = make_eval(rn, mask_j)
    nets = [("recall-trained", _load("exp8")), ("completion-trained", _load("exp9"))]
    PCT = [0.15, 0.50, 0.85]

    TILE, LAB_W, COL_GAP, ROW_GAP, ERR_H = 0.86, 1.30, 0.09, 0.05, 0.17
    BLOCK_W = LAB_W + 3 * TILE + 2 * COL_GAP
    BLOCK_TITLE = 0.30
    BLOCK_H = BLOCK_TITLE + 3 * TILE + 2 * ROW_GAP + 3 * ERR_H
    BAND_LAB = 1.55                       # left margin holding the novelty label
    GAP_X, GAP_Y = 0.55, 0.34
    TOP = 0.92
    FIG_W = BAND_LAB + 2 * BLOCK_W + GAP_X + 0.25
    FIG_H = TOP + 3 * BLOCK_H + 2 * GAP_Y + 0.15

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def txt(x, y, s, ha="center", **kw):
        fig.text(x / FIG_W, 1.0 - y / FIG_H, s, ha=ha, va="top", **kw)

    def tile(x, y, edge):
        ax = fig.add_axes([x / FIG_W, 1.0 - (y + TILE) / FIG_H,
                           TILE / FIG_W, TILE / FIG_H])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(edge); sp.set_linewidth(1.0)
        return ax

    txt(FIG_W / 2, 0.16, "Trained on digits 0-4. Every block is the same two networks.",
        fontsize=12.5)
    txt(FIG_W / 2, 0.46,
        "Numbers are normalised error: 1.00 is no better than drawing the average digit. "
        "Columns are fixed difficulty percentiles, ranked without any network.",
        fontsize=8.8, color="#555")

    for b, (band_label, cond_p, cond_a) in enumerate(BANDS):
        y0 = TOP + b * (BLOCK_H + GAP_Y)
        txt(BAND_LAB - 0.18, y0 + BLOCK_H / 2 - 0.30, band_label, ha="right",
            fontsize=10.5, fontweight="bold")

        es_a = ev[cond_a]
        blend = np.asarray(_soft_lookup(es_a.ctx, es_a.qry, mask_j, 0.03))
        per = (((blend - np.asarray(es_a.qry)) ** 2) * mask)[:, 0, :].sum(-1) / mask.sum()
        order = np.argsort(per)
        cols = [int(order[int(p * (len(order) - 1))]) for p in PCT]

        for k, (cond, head) in enumerate([(cond_p, "answer IS in the context"),
                                          (cond_a, "answer is NOT in the context")]):
            x0 = BAND_LAB + k * (BLOCK_W + GAP_X)
            es = ev[cond]
            if b == 0:
                txt(x0 + LAB_W + (3 * TILE + 2 * COL_GAP) / 2, y0 - 0.30, head,
                    fontsize=10.5, fontweight="bold")
            for r, (rlabel, params) in enumerate([("true image", None)] + nets):
                y = y0 + BLOCK_TITLE + r * (TILE + ROW_GAP + ERR_H)
                fig.text((x0 + LAB_W - 0.10) / FIG_W, 1.0 - (y + TILE / 2) / FIG_H,
                         rlabel, ha="right", va="center", fontsize=8.6)
                for c, i in enumerate(cols):
                    x = x0 + LAB_W + c * (TILE + COL_GAP)
                    truth = np.asarray(es.qry[i, 0])
                    if params is None:
                        img, lab, edge = truth, "", "#c1121f"
                    else:
                        pred, _ = eval_fn(params, es.ctx[i:i + 1], es.qry[i:i + 1])
                        pred = np.asarray(pred[0, 0])
                        img = truth * (1 - mask) + pred * mask
                        raw = float((((pred - truth) ** 2) * mask).sum() / mask.sum())
                        lab = f"{raw / es.mse_mean:.2f}"
                        edge = "#3a3936"
                    ax = tile(x, y, edge)
                    ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                    if lab:
                        txt(x + TILE / 2, y + TILE + 0.015, lab, fontsize=7.4)

    url = save_matplotlib_figure(f"{PROJ}_r12_digit_split_grid", fig, format="png", dpi=150)
    plt.close(fig)
    return url


def fig_bars():
    """The same six blocks as numbers, with the two references that bound them."""
    r8, r9 = rows["exp8_sharedq"]["final"], rows["exp9_sharedq"]["final"]
    bl = rows["baselines_M16_r14_split"]["baselines"]
    labels = ["0-4\nseen images", "0-4\nnew images", "5-9\nnew digits"]
    order_p = ["A_seen_present", "E_same_present", "B_novel_present"]
    order_a = ["C_seen_absent", "F_same_absent", "D_novel_absent"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    for ax, order, title in [(axes[0], order_p, "answer IS in the context"),
                             (axes[1], order_a, "answer is NOT in the context")]:
        x = np.arange(3)
        ax.bar(x - 0.19, [r8[c]["nmse"] for c in order], 0.36,
               label="recall-trained", color="#2f6fbf")
        ax.bar(x + 0.19, [r9[c]["nmse"] for c in order], 0.36,
               label="completion-trained", color="#e07a3c")
        ax.plot(x, [bl[c]["n_ridge"] for c in order], "k^", ms=7,
                label="ridge (ignores the context)")
        ax.axhline(1.0, color="#888", ls="--", lw=1.2)
        ax.text(2.42, 1.02, "no better than\nthe average digit", fontsize=7.5,
                color="#666", ha="right", va="bottom")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.35)
    axes[0].set_ylabel("normalised error (lower is better)", fontsize=9.5)
    axes[0].legend(fontsize=8.4, loc="upper left")
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_r12_digit_split_bars", fig, format="svg")
    plt.close(fig)
    return url


def main():
    url_grid = fig_grid()
    url_bars = fig_bars()
    print("grid:", url_grid)
    print("bars:", url_bars)

    md = f"""# Retrieval crosses to new digits. Prediction does not.

A network was trained on MNIST digits 0 to 4 only. It never saw a 5, 6, 7, 8 or
9 during training. Not as an example, not as a question, not as an answer.

Then it was tested on those five unseen digits.

It finds them perfectly. Asked to reproduce an image sitting in its context, it
picks the right one every time, even when that image is a 9. Its error is
**0.043**, where 0.0 is perfect and 1.0 means no better than drawing the average
digit.

It cannot predict them at all. Asked to fill in the missing half of an unseen
digit when the answer is *not* in its context, it scores **0.999**. That is the
average-digit score. It has learned nothing about what a 9 looks like.

Both numbers come from the same network, on the same unseen images, in the same
evaluation run. Only the question differs.

![Completions across three levels of novelty]({url_grid})

## The task

Each example is a small episode. The network is shown sixteen complete MNIST
images, one per token. Then it is shown a seventeenth image with its bottom half
erased. It has to produce the missing 392 pixels.

Error is mean squared error over those hidden pixels. It is divided by the error
of always drawing the average training image. So 1.0 is the do-nothing score.
Below 1.0 is better than nothing. Above 1.0 is worse.

Two kinds of episode matter, and they are the two columns of the figure above.

In the first kind, the answer is already there. The seventeenth image is a copy
of one of the sixteen. The network can succeed by finding it and copying it.

In the second kind, the answer is absent. None of the sixteen is the query. The
missing half has to be worked out from what the visible half suggests.

## Why split the digits

Held-out images are a weak test. A held-out 3 is a new image, but the network has
seen thousands of 3s. Whatever it knows about the shape of a 3 still applies.

Splitting by digit class removes that. Train on 0 to 4, test on 5 to 9, and the
test images are not merely new. Their whole category is new.

The figure has three rows for this reason. The top row is images seen in
training. The middle row is new images of digits 0 to 4 — new pictures, familiar
shapes. The bottom row is digits 5 to 9, which are new in every sense.

Reading down a column shows how much each kind of novelty costs.

## What the pictures show

Look at the left column of the figure, where the answer is present. All three
rows look the same. The recall-trained network reproduces the query almost
exactly, whether it is a familiar 3 or an unseen 9.

This is not a small effect. Identification accuracy — does the output most
resemble the correct one of the sixteen context images — is **1.000** on unseen
digits. Chance is 1/16, or 0.063.

Now look at the right column, where the answer is absent. The top two rows are
recognisable attempts. The bottom row is not. On unseen digits the network
produces a shape that belongs to no particular digit.

The middle row is the control that makes this readable. Those are new images too.
If new images alone broke the network, that row would fail as well. It does not:
**0.684** against the top row's 0.696. Novel images cost essentially nothing.

Novel classes cost everything: **0.999**.

One more thing is visible in the figure, in the completion-trained row. Its two
columns match, panel for panel, to within rounding. Same picture, same number,
whether the answer is present or absent.

That network ignores its context. It was never rewarded for reading it, so it
does not. Whatever it draws, it draws from the visible half alone.

![The same six blocks as numbers]({url_bars})

## Worse than ignoring the context

A linear model was fitted to predict hidden pixels from visible ones. It sees no
context at all. It is just a fixed map, trained on digits 0 to 4.

On unseen digits with the answer absent it scores **0.851**. The trained network
scores 0.999. The network with sixteen images to look at does worse than a linear
map with none.

So the network has not merely failed to learn a general prior. It has learned a
prior that applies to five digits and does not extend.

## The other training signal fails differently

A second network was trained on the same split, but on the opposite objective.
Its answer was never in the context. It could only ever predict.

That network handles new images of familiar digits reasonably: **0.642**. On
unseen digits it scores **1.221** — worse than drawing the average digit.

It also loses the ability to find things. Identification accuracy on unseen
digits drops to **0.370**, against the recall-trained network's 1.000.

So neither objective produces knowledge that crosses a class boundary. One keeps
its finding ability and loses its predicting ability. The other loses both.

## What this means

Finding an image and understanding an image are different capabilities here, and
only one of them transfers.

Finding works by matching pixels. A visible top half is compared against sixteen
candidates, and the closest one wins. Nothing in that operation needs to know
what digit it is looking at. It works on a 9 for the same reason it works on a 3.

Predicting needs something else. To finish an unseen bottom half the network
needs to know how that kind of shape continues. That knowledge was built from
digits 0 to 4. It does not stretch to 5 to 9.

The clean version of the result: **perfect retrieval and chance-level prediction,
on the same images, in the same run.**

## What this does not establish

The split is one particular split. Digits 0 to 4 may be unusually poor
preparation for 5 to 9. A rotation of which digits are held out would say whether
the effect is about class novelty in general.

Ten classes is a small vocabulary. A dataset with more classes would show whether
prediction transfers once training covers enough of the space.

Both networks here were trained on unrelated context images. Reports on the
nearest-neighbour version of this task show the context can be made informative,
and that is untested under a class split.

## Sources

`results.jsonl` rows `exp8_sharedq` and `exp9_sharedq` (recall-trained and
completion-trained, digits 0-4, shared-query evaluation), and
`baselines_M16_r14_split` for the linear and average-image references. Figures
generated by `scripts/gen_report_12.py`. No training was run for this report.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_12", md))


if __name__ == "__main__":
    main()
