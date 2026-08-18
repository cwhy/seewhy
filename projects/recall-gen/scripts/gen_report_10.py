"""Generates Report 10: which metric decides (a self-contained reader-facing report).

Numbers sourced from results.jsonl rows metrics_beyond_mse, baselines_M16_r14_knn_Q1,
exp20, exp24. Figures adapted from gen_report_06.py / gen_report_09.py conventions:
no experiment identifiers or project-internal vocabulary in captions, model-free
column selection for the pixel figure.

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_10.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).parent.parent))              # projects/recall-gen

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_lib.media import save_matplotlib_figure
from shared_lib.report import save_report

from lib.core import row_mask, Cfg, predict, masked_mse
from lib import evalsets
from lib.train import Run, build_pools
from baselines import _soft_lookup   # scripts/baselines.py — same directory

REPORT_MD_PATH = Path(__file__).parent.parent / "reports" / "10-which-metric-decides.md"
RESULTS = Path(__file__).parent.parent / "results.jsonl"
PROJECT = Path(__file__).parent.parent

rows = {}
for line in open(RESULTS):
    r = json.loads(line)
    rows[r["experiment"]] = r

PROJ = "recall-gen"
CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17)

M = rows["metrics_beyond_mse"]["metrics"]

TASK_DIAGRAM_URL = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_task_diagram.svg"


# ── Figure 1: the inversion — two temperatures, three metrics ──────────────
def fig_inversion():
    labels = ["squared error\n(lower better)", "realism\n(lower better)",
              "digit identity\n(higher better)"]
    sharp = [M["lookup_tau0.003"]["nmse"], M["lookup_tau0.003"]["realism"],
             M["lookup_tau0.003"]["nn_label_acc"]]
    blend = [M["lookup_tau0.03"]["nmse"], M["lookup_tau0.03"]["realism"],
             M["lookup_tau0.03"]["nn_label_acc"]]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    titles = ["squared error", "realism", "digit identity"]
    ylabels = ["normalised MSE\n(0=perfect, 1=mean image)",
               "distance to nearest\nreal hidden half", "fraction matching\ntrue label"]
    better = ["lower better", "lower better", "higher better"]
    for i, ax in enumerate(axes):
        xs = [0, 1]
        ys = [sharp[i], blend[i]]
        colors = ["C3", "C0"]
        ax.plot(xs, ys, "-", color="grey", lw=1.2, zorder=1)
        ax.scatter(xs, ys, color=colors, s=90, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5)
        ax.set_xticks([0, 1], ["tau=0.003\n(near-copy)", "tau=0.03\n(blend)"], fontsize=8)
        ax.set_title(titles[i], fontsize=10)
        ax.set_ylabel(ylabels[i], fontsize=8)
        ax.text(0.5, -0.32, better[i], transform=ax.transAxes, ha="center",
                fontsize=7.5, color="grey", style="italic")
        pad = (max(ys) - min(ys)) * 0.5 if max(ys) != min(ys) else max(ys) * 0.1
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_xlim(-0.4, 1.4)
    fig.suptitle("Same computation, one parameter changed: squared error picks the\n"
                 "opposite temperature from realism and digit identity", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    url = save_matplotlib_figure(f"{PROJ}_r10_inversion", fig, format="svg")
    plt.close(fig)
    return url


# ── Figure 2: realism vs squared error, every row in the table ─────────────
def fig_frontier():
    order = [
        ("true image (ceiling)", "true_image", "*", "k", 140),
        ("mean image (blur floor)", "mean_image", "X", "grey", 110),
        ("look-up tau=0.003 (near-copy)", "lookup_tau0.003", "o", "C3", 90),
        ("look-up tau=0.03 (blend)", "lookup_tau0.03", "o", "C0", 90),
        ("trained network, best", "exp20_best", "s", "C1", 90),
        ("trained network, end", "exp20_final", "s", "C1", 90),
        ("frozen network, best", "exp24_best", "^", "C2", 90),
        ("frozen network, end", "exp24_final", "^", "C2", 90),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for label, key, marker, color, size in order:
        x = M[key]["nmse"]
        y = M[key]["realism"]
        ax.scatter([x], [y], marker=marker, color=color, s=size, zorder=5,
                   edgecolor="k", linewidth=0.5, label=label)
    ax.set_xlabel("squared error (normalised, 0=perfect, 1=mean image)")
    ax.set_ylabel("realism (distance to nearest real hidden half, lower=more real)")
    ax.set_title("No trained network reaches the realism of the model-free\n"
                 "near-copy look-up, whatever its squared error", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_r10_frontier", fig, format="svg")
    plt.close(fig)
    return url


# ── Figure 3: completions in pixels, six rows, columns at fixed percentiles ─
def _per_sample_hidden_mse(pred, tgt, mask):
    return np.asarray((((pred - tgt) ** 2) * mask).sum(-1)[:, 0] / mask.sum())


def _composite(truth, hidden_pred, mask):
    return truth * (1 - mask) + hidden_pred * mask


def _per_sample_realism(hidden_pred, ref_h, ref_sq, n_hid):
    """hidden_pred: (C,392) -> per-sample nearest-real-neighbour distance / n_hid."""
    hp = np.asarray(hidden_pred)
    d = ((hp ** 2).sum(-1)[:, None] + np.asarray(ref_sq)[None, :]
         - 2.0 * (hp @ np.asarray(ref_h).T))
    return d.min(-1) / n_hid


def _load(name: str):
    with open(PROJECT / f"params_{name}.pkl", "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def fig_completions():
    rn = Run(exp_name="report10_completion", name="report10_completion",
             M=16, Q=1, mask_rows=14, cfg=CFG)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    mask_j = jnp.array(mask)
    hid = mask > 0.5
    n_hid = int(hid.sum())
    mean_img = pools["train"].mean(0)

    knn_ev = evalsets.build(pools, mask, 16, 1, 512, mean_img,
                            ctx_mode="knn", labels=labels)["D_novel_absent"]

    ref_h = jnp.array(pools["train"][:, hid])
    ref_sq = (ref_h ** 2).sum(-1)

    # Column choice: fixed percentiles of the blend look-up's (tau=0.03) own
    # per-sample error, model-free — same rule as report 6/9, with the same
    # tau this report already names "blend" so the figure caption is exact.
    soft_pred_full = np.asarray(_soft_lookup(knn_ev.ctx, knn_ev.qry, mask_j, 0.03))
    soft_err = _per_sample_hidden_mse(jnp.array(soft_pred_full), knn_ev.qry, mask_j)
    order = np.argsort(soft_err)
    pcts = [5, 23, 41, 59, 77, 95]
    cols = order[[int(round(p / 100 * (len(order) - 1))) for p in pcts]]

    truth = np.asarray(knn_ev.qry[cols, 0])
    ctx_c = knn_ev.ctx[cols]
    qry_c = knn_ev.qry[cols]
    mean_row = np.broadcast_to(mean_img, truth.shape)

    near_pred = np.asarray(_soft_lookup(ctx_c, qry_c, mask_j, 0.003))[:, 0]
    blend_pred = np.asarray(_soft_lookup(ctx_c, qry_c, mask_j, 0.03))[:, 0]

    p_trained = _load("exp20")
    p_frozen = _load("exp24_best")
    pred_trained = np.asarray(predict(p_trained, ctx_c, qry_c, mask_j, CFG)[:, 0])
    pred_frozen = np.asarray(predict(p_frozen, ctx_c, qry_c, mask_j, CFG)[:, 0])

    row_specs = [
        ("true image", truth, None),
        ("mean image", truth, mean_row),
        ("near-copy\nlook-up (tau=.003)", truth, near_pred),
        ("blend\nlook-up (tau=.03)", truth, blend_pred),
        ("trained network\n(end of training)", truth, pred_trained),
        ("frozen network\n(best)", truth, pred_frozen),
    ]

    ncols = len(pcts)
    nrows = len(row_specs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.85))
    for r, (label, tr, hp) in enumerate(row_specs):
        for c in range(ncols):
            ax = axes[r, c]
            t = tr[c]
            if hp is None:
                img = t.reshape(28, 28)
                err_txt = real_txt = ""
            else:
                comp = _composite(t, hp[c], mask)
                img = comp.reshape(28, 28)
                se = float((((hp[c] - t) ** 2) * mask).sum() / mask.sum())
                re = float(_per_sample_realism(hp[c][hid][None, :], ref_h, ref_sq, n_hid)[0])
                err_txt = f"e={se:.2f}"
                real_txt = f"r={re:.3f}"
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if err_txt:
                ax.text(0.03, 0.03, err_txt, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=6, color="lime",
                        bbox=dict(boxstyle="square,pad=0.05", fc="black", alpha=0.65, lw=0))
                ax.text(0.97, 0.03, real_txt, transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6, color="cyan",
                        bbox=dict(boxstyle="square,pad=0.05", fc="black", alpha=0.65, lw=0))
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5)
            if r == 0:
                ax.set_title(f"p{pcts[c]}", fontsize=7.5)
    fig.suptitle("Squared error (green, e) and realism (cyan, r) disagree about\n"
                 "which row is best", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3)
    url = save_matplotlib_figure(f"{PROJ}_r10_completions_v2", fig, format="png", dpi=150)
    plt.close(fig)
    return url


def main():
    url_inversion = fig_inversion()
    url_frontier = fig_frontier()
    url_completions = fig_completions()
    print("fig_inversion:", url_inversion)
    print("fig_frontier:", url_frontier)
    print("fig_completions:", url_completions)

    md = f"""# Which completion is "better" depends on which metric is asked, and the two reasonable metrics disagree

Squared error is minimised by the conditional mean, so it rewards a blurred
average over plausible completions of a missing image region. A metric that
asks whether the completion looks like a real digit, or like the *right*
digit, rewards a sharp copy of one plausible context image instead. These are
not two qualities of one good solution — they are opposite ends of a single
knob, and the objective a model is scored on decides which end it lands at.
The clearest demonstration needs no trained model at all: a weighted average
over 16 candidate images, with the averaging weight set by one temperature,
inverts its own ranking depending only on which metric reads the result.

| temperature | squared error (lower better) | realism (lower better) | digit identity (higher better) |
|---|---|---|---|
| tau=0.003 (near-copy) | 0.672 | 0.0154 | 0.805 |
| tau=0.03 (blend) | 0.553 | 0.0180 | 0.756 |

The blend wins on squared error (0.553 vs 0.672); the near-copy wins on
realism (0.0154 vs 0.0180) and digit identity (0.805 vs 0.756). Same
computation, one parameter, and the ranking inverts.

## Setup

Each example is a sequence of 16 MNIST images, 28x28 pixels flattened to 784
(the context), followed by a query image with its bottom 14 rows hidden (392
of 784 pixels scored). The task is to fill in the hidden half; the context is
built from the query's 16 nearest neighbours by pixel distance on the visible
half, and results below are restricted to the case where the query's true
image is *not* one of the 16 — the only case where "completion quality" means
anything, since when the answer is present the best move is simply to find it.

![the task: 16 context images write into a fixed-size state, the query reads it, and only the greyed region is scored]({TASK_DIAGRAM_URL})

Two computations are scored throughout. One is a plain weighted average of the
16 context images — no learned parameters — with a temperature `tau`
controlling how concentrated the weighting is:

```
d_i    = || visible(query) - visible(context_i) ||^2 / n_visible_pixels
w_i    = softmax(-d / tau)_i
output = sum_i  w_i * context_i
```

At low temperature (tau=0.003) almost all the weight lands on the single
closest-matching context image, so the output is nearly a copy of it. At
higher temperature (tau=0.03) the weight spreads across several near
neighbours and the output is their blend. The other is a trained recurrent
network (4-layer linear-attention / delta-rule architecture, 4.03M
parameters) that writes the 16 context images into a fixed-size state and
reads a prediction from it after the query is presented, either left free to
train throughout, or trained only in its input and output layers with the
recurrent layers frozen at random initialisation.

Four metrics are computed on the same 512 held-out episodes:

- **squared error** — masked MSE over the 392 hidden pixels, divided by the
  masked MSE of always predicting the training-set mean image, so 1.0 = no
  better than the average digit and 0.0 = perfect.
- **realism** — per-pixel distance from the predicted hidden half to the
  *nearest real hidden half* in the training pool. No classifier, nothing
  tuned on the models being judged: it asks "does this look like some real
  digit's bottom half", not "is it the right one".
- **digit identity** — the label of that nearest real training image,
  compared against the query's true label. Answers the question realism
  cannot: whether the completion resembles the right digit, not just any
  digit.
- **classifier** — an independently trained MLP (97.99% held-out accuracy)
  classifying the composited completion (true visible half + predicted hidden
  half) against the true label.

## Full comparison, 512 held-out episodes

| | squared error | realism | digit identity | classifier |
|---|---|---|---|---|
| true image (ceiling) | 0.000 | 0.0161 | 0.869 | 0.988 |
| mean image (blur floor) | 1.000 | 0.0360 | 0.113 | 0.641 |
| look-up, tau=0.003 (near-copy) | 0.672 | 0.0154 | 0.805 | 0.912 |
| look-up, tau=0.03 (blend) | 0.553 | 0.0180 | 0.756 | 0.918 |
| trained network, best checkpoint | 0.505 | 0.0169 | 0.729 | 0.930 |
| trained network, end of training | 0.666 | 0.0172 | 0.758 | 0.908 |
| frozen-layer network, best checkpoint | 0.471 | 0.0166 | 0.744 | 0.930 |
| frozen-layer network, end of training | 0.474 | 0.0165 | 0.738 | 0.928 |

![the inversion: the two look-up temperatures on all three metrics]({url_inversion})

Three things follow from this table.

**The metric decides the winner.** The two look-up rows swap places between
squared error and the other two columns, as already shown above.

**No trained network beats the model-free near-copy on realism or digit
identity.** Every trained-network row scores worse realism than 0.0154 and
worse digit identity than 0.805 — and 0.805 is close to the 0.869 ceiling set
by the true image itself. On the metrics that do not reward blur, the best
strategy measured here is "copy the nearest context image", and every network
in this table underperforms it. This holds across all four network rows, best
and final checkpoints of both the trained and the frozen-layer models — it is
not one unlucky checkpoint.

**The frozen-layer network still beats the fully-trained one.** Comparing
best checkpoints, frozen beats trained on squared error (0.471 vs 0.505),
realism (0.0166 vs 0.0169) and the classifier (0.930 vs 0.930, tied);
comparing end-of-training checkpoints, frozen beats trained on squared error
(0.474 vs 0.666), realism (0.0165 vs 0.0172) and the classifier (0.928 vs
0.908), losing narrowly only on digit identity (0.738 vs 0.758). So that
comparison is not an artefact of which metric is used.

![realism against squared error, every row in the table]({url_frontier})

Two entries need a second look because they run against the intuitive
ordering. The near-copy look-up scores *better* realism (0.0154) than the
true image itself (0.0161) — not a measurement error: a copied training image
is, by construction, close to a training image, whereas a held-out query
image's own nearest training neighbour sits slightly further away on average.
The same effect caps digit identity below 1.0 even for the true image
(0.869): the nearest training image by hidden-half distance sometimes carries
a different label than the query, so "the closest match" and "the same digit"
are not always the same training image.

## What the classifier column does not add

The classifier spans a narrow range, 0.908 to 0.930, across every network
row, and this narrowness is not model quality — it is the instrument. The
true visible half already carries most of the class signal on its own: even
the mean-image row, with no information about the hidden half at all, scores
0.641. Realism and digit identity, which need no classifier and are not
trained on anything being judged, are the informative columns here; the
classifier corroborates at best.

## What this looks like in pixels

![six completions of the same six queries, chosen at fixed percentiles of a model-free difficulty measure, true visible half composited back in, each labelled with its squared error (green, e) and realism (cyan, r)]({url_completions})

Columns are six queries at fixed percentiles (p5 through p95) of the blend
look-up's own per-sample error — a model-free difficulty ordering, so no
network's own error picked which examples are shown. Rows: the true image;
the mean image; the near-copy look-up; the blend look-up; the fully-trained
network at the end of training; the frozen-layer network at its best
checkpoint. No single completion is barred from scoring well on both labels —
several panels do — but the two metrics rank the *strategies* differently:
averaged over the full 512-episode set (the table above), blend wins squared
error and near-copy wins realism and digit identity, and that aggregate
disagreement is what the per-panel labels here are showing example by
example, not a per-image trade-off.

## What this changes

A single-number comparison of completion quality on this task is not
meaningful without saying which metric produced it — the ranking of two
otherwise identical computations inverts depending on whether squared error
or realism/digit-identity is read. The temperature of the look-up computation
is better understood as trading between two real objectives than as a knob
toward a generically "better" model, and the same applies to any property of
a trained network's output that moves it toward or away from a copy of one
context image. Nothing here says which of these metrics should be preferred
for this task — that depends on what the completion is being produced for,
and is not answered by the data in this report.

## Sources

`results.jsonl` rows `metrics_beyond_mse`, `baselines_M16_r14_knn_Q1`,
`exp20`, `exp24`. Metric computation: `projects/recall-gen/scripts/metrics_beyond_mse.py`.
"""

    word_count = len(md.split())
    print("WORD COUNT (approx, includes headers/tables):", word_count)

    report_url = save_report(f"{PROJ}_report_10", md)
    print("REPORT:", report_url)

    REPORT_MD_PATH.write_text(md)
    print("LOCAL FILE:", REPORT_MD_PATH)


if __name__ == "__main__":
    main()
