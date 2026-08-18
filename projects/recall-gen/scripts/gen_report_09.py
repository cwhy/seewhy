"""Generates Report 9: what recall training learns (a self-contained reader-facing report).

Numbers sourced from results.jsonl rows exp1, exp20, exp24, exp26, exp27, exp28,
baselines_M16_r14_knn_Q1, and the derived rows effective_tau2, ctx_ablation2.
Figures adapted from gen_report_08.py, recaptioned for a reader with no project
context: no experiment identifiers or project-internal vocabulary in captions.

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_09.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))   # repo root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_lib.media import save_matplotlib_figure, save_media
from shared_lib.report import save_report

REPORT_MD_PATH = Path(__file__).parent.parent / "reports" / "09-what-recall-training-learns.md"
RESULTS = Path(__file__).parent.parent / "results.jsonl"

rows = {}
for line in open(RESULTS):
    r = json.loads(line)
    rows[r["experiment"]] = r

PROJ = "recall-gen"
TAUS_PLOT = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


# ── Figure 1: the temperature trade-off, model-free ─────────────────────────
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    b = rows["baselines_M16_r14_knn_Q1"]["baselines"]
    present = b["A_seen_present"]
    absent = b["D_novel_absent"]
    p_vals = [present["mse_knn_by_tau"][str(t)] / present["mse_mean"] for t in TAUS_PLOT]
    a_vals = [absent["mse_knn_by_tau"][str(t)] / absent["mse_mean"] for t in TAUS_PLOT]

    ax.plot(TAUS_PLOT, p_vals, "o-", color="C0", lw=1.8,
            label="answer present in context (copying)")
    ax.plot(TAUS_PLOT, a_vals, "o-", color="C1", lw=1.8,
            label="answer absent from context (prediction)")
    ax.set_xscale("log")

    pi = int(np.argmin(p_vals))
    ai = int(np.argmin(a_vals))
    ax.scatter([TAUS_PLOT[pi]], [p_vals[pi]], color="C0", zorder=5, s=80,
               edgecolor="k", linewidth=0.8)
    ax.scatter([TAUS_PLOT[ai]], [a_vals[ai]], color="C1", zorder=5, s=80,
               edgecolor="k", linewidth=0.8)
    ax.annotate(f"best: {p_vals[pi]:.3f} at tau={TAUS_PLOT[pi]}",
                (TAUS_PLOT[pi], p_vals[pi]), textcoords="offset points",
                xytext=(6, -12), fontsize=8.5, color="C0")
    ax.annotate(f"best: {a_vals[ai]:.3f} at tau={TAUS_PLOT[ai]}",
                (TAUS_PLOT[ai], a_vals[ai]), textcoords="offset points",
                xytext=(6, 8), fontsize=8.5, color="C1")
    ax.set_xlabel("temperature of the softmax similarity weighting (tau)")
    ax.set_ylabel("error, normalised to predicting the average image (1.0)")
    ax.set_title("Putting more weight on the single nearest context image helps\n"
                 "copying and hurts prediction — with no trained model involved", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="center right")
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_r9_tradeoff", fig, format="svg")
    plt.close(fig)
    return url


# ── Figure 2: fitted temperature vs held-out error, trained vs frozen ───────
def fig_tau_vs_d():
    et = rows["effective_tau2"]["effective_tau"]

    def taustar(mk):
        return et[mk]["B_novel_present"]["tau_star"]

    trained = [
        ("fully trained, mid-training", taustar("exp20_best"), 0.505, "C0", "o"),
        ("fully trained, end of training", taustar("exp20_final"), 0.666, "C0", "o"),
        ("fully trained, unrelated-image context", taustar("exp1_final"), 0.843, "C0", "o"),
    ]
    frozen = [
        ("frozen layers, mid-training", taustar("exp24_best"), 0.471, "C1", "s"),
        ("frozen layers, end of training", taustar("exp24_final"), 0.474, "C1", "s"),
        ("frozen layers, unrelated-image context", taustar("exp26_final"), 0.778, "C1", "s"),
        ("frozen layers, distant neighbours (64)", taustar("exp27_final"), 0.509, "C1", "s"),
        ("frozen layers, distant neighbours (512)", taustar("exp28_final"), 0.616, "C1", "s"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, tau, d, color, marker in trained + frozen:
        ax.scatter([tau], [d], color=color, s=75, zorder=5, marker=marker)

    x0, y0 = taustar("exp20_best"), 0.505
    x1, y1 = taustar("exp20_final"), 0.666
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
    ax.annotate("same model, later in training:\noutput 57x more concentrated on one image,\nerror rises 0.505 -> 0.666",
                xy=(1.3e-3, 0.585), fontsize=8)

    ax.annotate("all five frozen-layer checkpoints:\ntemperature never moves",
                xy=(0.032, 0.50), xytext=(0.045, 0.46), fontsize=8, color="C1",
                arrowprops=dict(arrowstyle="-", color="C1", lw=0.7))

    ax.set_xscale("log")
    ax.set_xlim(2e-4, 0.4)
    ax.set_ylim(0.40, 0.90)
    ax.set_xlabel("fitted temperature (lower = more output weight on one context image)")
    ax.set_ylabel("error on held-out prediction (answer absent from context)")
    ax.set_title("Every model that keeps training its context-reading layers puts more\n"
                 "weight on one image and gets worse at prediction; frozen ones cannot move", fontsize=10.5)
    ax.scatter([], [], color="C0", marker="o", label="context-reading layers trained")
    ax.scatter([], [], color="C1", marker="s", label="context-reading layers frozen")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_r9_tau_vs_error", fig, format="svg")
    plt.close(fig)
    return url


# ── Figure 3: distance geometry concentrates the weighting by itself ────────
def fig_geometry():
    cats = ["nearest\nneighbour", "64th-nearest\nneighbour", "512th-nearest\nneighbour", "unrelated\nimages"]
    idb = [0.294921875, 0.52734375, 0.888671875, 0.98779296875]
    degrad = [0.47360331965307195 - 0.4712455593161424,
              0.5093294482230349 - 0.502035564998649,
              0.6157017586047248 - 0.5216484950645995,
              0.7776823437356903 - 0.5695741578723181]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    x = np.arange(len(cats))
    colors = ["C0", "C0", "C0", "C3"]

    ax = axes[0]
    bars = ax.bar(x, idb, color=colors)
    for b, v in zip(bars, idb):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, cats, fontsize=8)
    ax.set_ylabel("exact-match accuracy on the context item")
    ax.set_title("Moving the nearest distractor further away\nraises exact-match accuracy,\n"
                 "with the context-reading layers frozen throughout", fontsize=9)
    ax.set_ylim(0, 1.08)

    ax = axes[1]
    bars = ax.bar(x, degrad, color=colors)
    for b, v in zip(bars, degrad):
        ax.annotate(f"+{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, cats, fontsize=8)
    ax.set_ylabel("loss in held-out prediction accuracy\nover training (higher = worse)")
    ax.set_title("...and the same ordering degrades prediction,\n"
                 "even though the model's temperature never changes", fontsize=9)
    ax.set_ylim(0, 0.23)

    fig.suptitle("Context geometry alone can concentrate a fixed-temperature weighting "
                 "on one item", fontsize=10.5)
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_r9_geometry", fig, format="svg")
    plt.close(fig)
    return url


# ── Setup diagram: the task, drawn as an SVG (no plotting library) ─────────
def make_task_diagram_svg() -> str:
    """16 context images write into a fixed-size state; the query reads it;
    only the greyed-out bottom half of the query is hidden from the network.
    Flat grey means exactly one thing in this figure: "hidden from the
    network" — it is used only on the query tile, never on the predicted
    tile (which shows an approximate, visibly imperfect completion) or the
    true-image tile. Colour convention matches the paper's architecture
    figure: blue = writes (context), orange = reads only (query)."""
    c_ctx = "#2a78d6"
    c_qry = "#eb6834"
    c_ink = "#0b0b0b"
    c_mut = "#6b6a66"
    c_pan = "#f2f1ee"
    c_state = "#3a3936"
    c_hidden = "#b9b7b1"

    def tile(cx, cy, col, top_strokes=None, bottom_strokes=None, masked=False):
        """One 52x52 image tile. `masked=True` fills the bottom half flat
        grey (hidden from the network) and draws no bottom strokes. Otherwise
        `bottom_strokes` (if given) are drawn on the same panel background as
        the top half, so a tile with content never looks like a hidden one."""
        s = (f'<rect x="{cx-26}" y="{cy-26}" width="52" height="52" rx="4" '
             f'fill="{c_pan}" stroke="{col}" stroke-width="2"/>')
        for (x1, y1, x2, y2) in (top_strokes or []):
            s += (f'<line x1="{cx+x1}" y1="{cy+y1}" x2="{cx+x2}" y2="{cy+y2}" '
                  f'stroke="#3a3936" stroke-width="2.4" stroke-linecap="round"/>')
        if masked:
            s += (f'<rect x="{cx-26}" y="{cy}" width="52" height="26" '
                  f'fill="{c_hidden}" stroke="{col}" stroke-width="2"/>')
        else:
            for (x1, y1, x2, y2) in (bottom_strokes or []):
                s += (f'<line x1="{cx+x1}" y1="{cy+y1}" x2="{cx+x2}" y2="{cy+y2}" '
                      f'stroke="#3a3936" stroke-width="2.4" stroke-linecap="round"/>')
        return s

    def label_block(cx, y0, lines, line_height=19, anchor="middle"):
        """Multi-line caption, one anchor point, fixed line spacing — never
        hand-placed per line, so lines cannot be made to collide by editing
        one of them without the others."""
        s = []
        for i, (text, color, size) in enumerate(lines):
            s.append(f'<text x="{cx}" y="{y0 + i * line_height}" font-size="{size}" '
                      f'fill="{color}" text-anchor="{anchor}">{text}</text>')
        return "".join(s)

    parts = []
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 360" '
                  'font-family="Helvetica,Arial,sans-serif">')
    parts.append('<rect x="0" y="0" width="980" height="360" fill="white"/>')

    # ── row 1: context images -> state -> query ─────────────────────────────
    row1_cy = 90   # tile centres; tile bottom edge is row1_cy + 26 = 116

    # context tiles, four drawn + ellipsis, each with a couple of pen strokes
    ctx_x = [60, 130, 200, 270]
    ctx_strokes = [
        [(-14, -14, 10, 12), (10, -14, -10, 14)],
        [(-14, 10, 14, -10), (0, -16, 0, 16)],
        [(-14, -10, 14, 10), (-14, 12, 14, -12)],
        [(0, -16, 0, 16), (-14, 0, 14, 0)],
    ]
    for x, strokes in zip(ctx_x, ctx_strokes):
        parts.append(tile(x, row1_cy, c_ctx, top_strokes=strokes))
    parts.append(f'<text x="330" y="{row1_cy - 2}" font-size="20" fill="{c_mut}">&#8943;</text>')
    # one anchor, two lines, starting well clear of the tile bottom (116)
    parts.append(label_block(165, 150, [
        ("16 context images, complete", c_ink, 13.5),
        ("these WRITE into the state", c_ctx, 12),
    ]))

    # arrow context -> state
    parts.append(f'<line x1="300" y1="{row1_cy}" x2="400" y2="{row1_cy}" '
                 f'stroke="{c_ctx}" stroke-width="2.5" marker-end="url(#arrC)"/>')

    # state box
    parts.append(f'<rect x="410" y="{row1_cy-55}" width="180" height="90" rx="8" '
                 f'fill="white" stroke="{c_state}" stroke-width="2"/>')
    parts.append(label_block(500, row1_cy - 20, [
        ("state S", c_ink, 16),
    ], anchor="middle").replace("<text", '<text font-weight="bold"', 1))
    parts.append(label_block(500, row1_cy + 2, [
        ("4 heads &#215; 64 &#215; 64", c_mut, 12.5),
        ("= 16,384 numbers, fixed size", c_ink, 12.5),
    ], line_height=18))
    # note sits below the box (box bottom = row1_cy+35 = 125)
    parts.append(label_block(500, 150, [
        ("(16 images &#215; 784 pixels = 12,544", c_mut, 12),
        ("numbers of context content)", c_mut, 12),
    ], line_height=16))

    # arrow state -> query
    parts.append(f'<line x1="590" y1="{row1_cy}" x2="690" y2="{row1_cy}" '
                 f'stroke="{c_qry}" stroke-width="2.5" marker-end="url(#arrQ)"/>')
    parts.append(f'<text x="640" y="{row1_cy-10}" font-size="12.5" fill="{c_qry}" '
                 f'text-anchor="middle">read</text>')

    # query tile: bottom half genuinely hidden from the network -> flat grey
    query_top_strokes = [(-14, -14, 10, 12), (10, -14, -10, 14)]
    parts.append(tile(730, row1_cy, c_qry, top_strokes=query_top_strokes, masked=True))
    parts.append(label_block(730, 150, [
        ("query, bottom half hidden", c_ink, 13.5),
        ("only READS the state", c_qry, 12),
    ]))

    # ── row 2: predicted vs. true, below ────────────────────────────────────
    py = 260
    parts.append(f'<line x1="730" y1="{row1_cy+26}" x2="730" y2="{py-26}" '
                 f'stroke="{c_mut}" stroke-width="1.5" stroke-dasharray="4,3"/>')

    # predicted tile: SAME visible top as the query, but the bottom half is
    # filled with an approximate, visibly imperfect completion — never grey.
    pred_bottom_strokes = [(2, 3, -9, 19), (-9, 19, 7, 21)]
    parts.append(tile(700, py, c_qry, top_strokes=query_top_strokes,
                       bottom_strokes=pred_bottom_strokes))
    parts.append(label_block(700, py + 40, [("predicted", c_ink, 13)]))
    parts.append(label_block(700, py + 58, [("(model output, imperfect)", c_mut, 10.5)]))

    parts.append(f'<text x="760" y="{py+4}" font-size="20" fill="{c_mut}" '
                 f'text-anchor="middle">vs</text>')

    # true-image tile: same visible top, plus the actual bottom-half strokes;
    # a red outline marks the region that is scored.
    true_bottom_strokes = [(0, 2, -8, 22), (-8, 22, 10, 18)]
    parts.append(tile(820, py, c_qry, top_strokes=query_top_strokes,
                       bottom_strokes=true_bottom_strokes))
    parts.append(f'<rect x="794" y="{py}" width="52" height="26" fill="none" '
                 f'stroke="#c1121f" stroke-width="2.5"/>')
    parts.append(label_block(820, py + 40, [("true image", c_ink, 13)]))

    parts.append(label_block(900, py - 10, [
        ("scored region", "#c1121f", 12),
    ], anchor="start"))
    parts.append(f'<line x1="898" y1="{py-6}" x2="846" y2="{py+13}" '
                 f'stroke="#c1121f" stroke-width="1.2"/>')

    parts.append('<defs>'
                  f'<marker id="arrC" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
                  f'<path d="M0,0 L8,4 L0,8 Z" fill="{c_ctx}"/></marker>'
                  f'<marker id="arrQ" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
                  f'<path d="M0,0 L8,4 L0,8 Z" fill="{c_qry}"/></marker>'
                  '</defs>')

    parts.append('</svg>')
    return "\n".join(parts)


def main():
    url_tradeoff = fig_tradeoff()
    url_tau_d = fig_tau_vs_d()
    url_geometry = fig_geometry()
    url_diagram = save_media(f"{PROJ}_r9_task_diagram.svg", make_task_diagram_svg().encode("utf-8"), "image/svg+xml")
    print("fig_tradeoff:", url_tradeoff)
    print("fig_tau_vs_error:", url_tau_d)
    print("fig_geometry:", url_geometry)
    print("fig_diagram:", url_diagram)

    md = f"""# Generalisation appears only where the network is kept from resembling a copy of its single closest match

Two computations appear throughout this report. One is a trained recurrent
network (architecture below). The other is a **reference computation** with no
learned parameters at all — a plain weighted average of the 16 context images,
weight set by a temperature — used both as a baseline score and as a ruler for
describing the network. Only the reference computation literally contains a
temperature: at low temperature almost all its weight lands on the single
closest-matching context image, so its output is a copy of that image; at high
temperature its weight spreads across several near neighbours and its output
is their blend. The network has no such mechanism built into it and so has no
temperature of its own — but it can be *described* by one: take a checkpoint's
own output and ask which temperature of the reference computation would have
produced something closest to it. That fitted value is what "the network's
temperature" means everywhere below.

Left free to train, the fitted value describing the network keeps falling —
its output increasingly resembles a copy of the single closest-matching
context image — because the training signal only ever rewards exact copying,
never blending. The cost is generalisation: on held-out predictions where the
answer is not in the context, the fully-trained network's best result is
**0.505** normalised error (0.0 would be perfect, 1.0 is no better than
predicting the average image), degrading to 0.666 by the end of training as
its output moves further toward a copy of one image. A version of the same
network with its recurrent layers left at random initialisation — only the
input and output layers trained — never comes to resemble a copy of a single
image, and reaches **0.471**: better than the fully-trained network ever gets,
better than the best the reference computation can do at any fixed temperature
over the same context (0.552), and better than a ridge regression baseline
that ignores the context entirely (0.631).

## Setup

Each example is a sequence of 16 MNIST images, 28x28 pixels flattened to 784
(the context), followed by a query image with its bottom 14 rows hidden (392
of 784 pixels scored). The network predicts the missing pixels.

![the task: 16 context images write into a fixed-size state, the query reads it, and only the greyed region is scored]({url_diagram})

The network is a 4-layer linear-attention recurrent network (a delta-rule /
KDA-style layer), d_model 256, 4 heads of dimension 64, about 4.03M parameters
total. Context tokens write into a matrix-valued state per layer, one 64x64
matrix per head; the query token never writes, and reads only the finished
state after all 16 context tokens have written to it:

```
S <- S * diag(alpha_t)              # per-channel forgetting
vhat = S k_t                        # what the state currently holds at this key
e = beta_t * (v_t - vhat)           # correction toward the true value
S <- S + e k_t^T                    # write
o_t = S q_t / sqrt(d_k)             # read
```

The state holds 4 x 64 x 64 = 16,384 numbers, regardless of how many images
were written into it — noticeably fewer than the 16 x 784 = 12,544 numbers of
raw context content, so the state is doing compression, not storing the
context verbatim. Pixel predictions come out of an MLP head reading the
state's output; nothing in this pipeline normalises a distribution over the 16
context images.

Error is masked MSE, normalised by the masked MSE of always predicting the
training-set average image:

```
score = mean((pred - truth)^2 over hidden pixels)
       / mean((train_mean_image - truth)^2 over hidden pixels)
```

Two things vary in what follows: whether the query's true image happens to be
one of the 16 context images (the answer is present — exact copying is
possible) or not (the answer is absent — it must be predicted from
similar-looking neighbours); and how the context is built, either 16 images
unrelated to the query or the query's 16 nearest neighbours by pixel distance
on the visible half.

The reference computation, used as both a baseline and a ruler, is:

```
d_i    = || visible(query) - visible(context_i) ||^2 / n_visible_pixels
w_i    = softmax(-d / tau)_i
output = sum_i  w_i * context_i
```

and the fitted temperature used to describe a network checkpoint is the tau
that makes this reference computation's output closest to that checkpoint's
own output on the same batch:

```
tau*  =  argmin_tau   masked_mse( reference_output(context, query, tau),
                                   network_output(context, query) )
```

## The trade-off exists in the reference computation alone, with no network involved

Sweep the reference computation's temperature with no network in the loop. On
nearest-neighbour contexts, moving its weight almost entirely onto the single
nearest neighbour (tau=0.03 down to tau=0.003) lowers copying error from 0.372
to 0.013 (a gain of 0.359) but raises prediction error from 0.553 to 0.672 (a
cost of 0.119). The two objectives have opposite optimal temperatures. This is
a property of the task and the data, not of anything a network learns.

![the temperature trade-off, no network involved]({url_tradeoff})

## Trained networks come to resemble a copy of one image, and held-out accuracy falls as they do

For each training checkpoint, the fitted temperature — which value of tau
makes the reference computation's output closest to the network's own output,
defined above — falls as training proceeds. On nearest-neighbour contexts it
starts at 0.03 early in training, where held-out prediction error is at its
best value measured for this network, 0.505, and falls to 0.00053 by the end
of training — meaning the network's own output has become nearly
indistinguishable from a copy of a single context image — where the same
error has risen to 0.666. On unrelated-image contexts the network's output
moves even closer to a copy of one image, fitted temperature 0.0017, and
prediction error rises further still, to 0.843.

![fitted temperature against held-out prediction error, trained and frozen checkpoints]({url_tau_d})

## Preventing that resemblance prevents the degradation

Training only the input embedding and output layers (0.60M of 4.03M
parameters) and leaving the four recurrent layers at random initialisation
means training can change how a token is embedded and how the state's output
is read out into pixels, but not how the state combines context tokens in the
first place. The fitted temperature describing this network stays at 0.03 for
the entire run — its output stays close to what the reference computation
would blend from several context images, never drifting toward a copy of one.
Held-out prediction error falls monotonically to 0.471 — the best number in
this comparison, ahead of the fully-trained network's best checkpoint (0.505),
the best the reference computation can do at any fixed temperature over the
same context (0.552), and a ridge regression baseline with no context at all
(0.631). A control that swaps in a different query's neighbours drops this
network's accuracy to 0.763, so the result is not simply memorising a prior
over digit shapes — the frozen network is reading the context it is given.

## A second way to make the network resemble a copy of one image, with no training at all

The same frozen network, given contexts built from progressively more distant
neighbours (the nearest neighbour, the 64th-nearest, the 512th-nearest, then
unrelated images), moves through exact-match accuracy 0.295, 0.527, 0.889,
0.988 — even though its fitted temperature never changes from 0.03. Held-out
prediction accuracy degrades over training more at each step, in the same
order: +0.002, +0.007, +0.094, +0.208. This matches how the reference
computation itself would behave if held at a fixed temperature while only the
distances changed: when one context item sits much closer than the rest, even
a fixed-temperature weighted average puts most of its weight on that one item.
Geometry alone reproduces the effect of a lower temperature, with nothing
inside the network having to change. Resemblance to a copy of one image has
two independent causes here — the training-driven one above, and this purely
geometric one — and either is enough to trade prediction for copying.

![exact-match accuracy and prediction degradation as distractor distance grows]({url_geometry})

## What this does not establish

The frozen network trains 0.60M parameters against the fully-trained
network's 4.03M, so this comparison confounds "kept from resembling a copy of
one image" with "fewer trainable parameters" — a frozen network matched in
trainable capacity has not been run. One configuration also does not fit the
account above: the frozen network on unrelated-image contexts never comes to
resemble a copy of one image (fitted temperature stays at 0.03) yet its
held-out accuracy still degrades over training, 0.570 to 0.778 — the largest
degradation of any frozen run measured. Coming to resemble a copy of one image
is sufficient to cause the degradation seen in the trained networks, but it is
evidently not the only mechanism, and the second one is not identified here.
All of this is one dataset (MNIST), one architecture, one masking pattern.

## Sources

`results.jsonl` rows `exp1`, `exp20`, `exp24`, `exp26`, `exp27`, `exp28`,
`baselines_M16_r14_knn_Q1`, `effective_tau2`, `ctx_ablation2`. Temperature
fitting: `projects/recall-gen/scripts/effective_tau.py`. Context construction:
`knn_offset` parameter in `projects/recall-gen/lib/train.py` and
`lib/evalsets.py`.
"""

    word_count = len(md.split())
    print("WORD COUNT (approx, includes headers/tables):", word_count)

    report_url = save_report(f"{PROJ}_report_09", md)
    print("REPORT:", report_url)

    REPORT_MD_PATH.write_text(md)
    print("LOCAL FILE:", REPORT_MD_PATH)


if __name__ == "__main__":
    main()
