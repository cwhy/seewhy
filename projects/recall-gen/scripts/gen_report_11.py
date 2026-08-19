"""Generates Report 11: recall and generalisation, the long careful version
(a self-contained reader-facing report, no project context assumed).

Numbers sourced from results.jsonl rows exp1, exp20, exp24, exp26, exp27, exp28,
baselines_M16_r14, baselines_M16_r14_knn_Q1, effective_tau2, ctx_ablation2,
metrics_beyond_mse. Most figures are REUSED BY URL from reports 8, 9 and 10
(already published, still live) rather than regenerated. Three figures are
new: the single worked episode, the 2x2 condition matrix shown as pictures
(setup only), and the 2x2 completions matrix (true/trained/frozen rows,
one block per condition).

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_11.py
"""
import json
import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent               # projects/recall-gen
sys.path.append(str(PROJECT_DIR.parents[1]))                       # repo root
sys.path.insert(0, str(PROJECT_DIR))                                # for `lib.*`

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.core import Cfg, row_mask, predict
from lib import evalsets
from lib.train import Run, build_pools, make_eval
from shared_lib.media import save_matplotlib_figure, save_media
from shared_lib.report import save_report

REPORT_MD_PATH = PROJECT_DIR / "reports" / "11-recall-and-generalisation.md"
RESULTS = PROJECT_DIR / "results.jsonl"
PROJ = "recall-gen"

rows = {}
for line in open(RESULTS):
    r = json.loads(line)
    rows[r["experiment"]] = r

CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20)

# ── figures reused by URL, already published and confirmed live ────────────
URL_TASK_DIAGRAM = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_task_diagram.svg"
URL_TRADEOFF      = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_tradeoff.svg"
URL_TAU_VS_ERROR  = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_tau_vs_error.svg"
URL_RECON_PRESENT = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_recon_present.png"
URL_RECON_ABSENT  = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_recon_absent.png"
URL_INVERSION     = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_inversion.svg"
URL_FRONTIER      = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_frontier.svg"
URL_NN_COMPLETIONS = "https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_completions_v2.png"
URL_DIAL          = "https://media.tanh.xyz/seewhy/26-08-18/recall-gen_dial.svg"


def _soft_lookup(ctx, qry, mask, tau):
    vis = 1.0 - mask
    d = (((qry[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1) / vis.sum()
    w = jax.nn.softmax(-d / tau, axis=-1)
    return jnp.einsum("eqm,emp->eqp", w, ctx)


def _load(name):
    with open(PROJECT_DIR / f"params_{name}.pkl", "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


# ── NEW Figure: one episode, shown completely ───────────────────────────────
def fig_worked_episode():
    """One held-out episode from the knn/absent condition (the query's true
    image is one of its 16 nearest neighbours' neighbourhoods but is not
    itself in the context — the harder, generalisation-relevant case used
    throughout the second half of this report), shown in full: all 16
    context images, the query with its bottom half hidden, and what five
    methods predict for the missing 392 pixels, each labelled with its own
    raw squared error on the hidden half. The episode shown is the
    median-difficulty (p50) example by the blend look-up's own per-sample
    error, a model-free ordering — not hand-picked, not the network's own
    best or worst case."""
    rn = Run(exp_name="", name="", M=16, Q=1, mask_rows=14, cfg=CFG)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    mask_j = jnp.array(mask)
    hid = mask > 0.5
    mean_img = pools["train"].mean(0)
    ev = evalsets.build(pools, mask, 16, 1, 512, mean_img, ctx_mode="knn", labels=labels)
    es = ev["D_novel_absent"]

    blend_pred_full = np.asarray(_soft_lookup(es.ctx, es.qry, mask_j, 0.03))
    per = (((blend_pred_full - np.asarray(es.qry)) ** 2) * mask)[:, 0, :].sum(-1) / mask.sum()
    order = np.argsort(per)
    i = int(order[int(0.5 * (len(order) - 1))])   # p50, model-free difficulty

    ctx_i = es.ctx[i]                 # (16, 784)
    qry_i = es.qry[i:i + 1]           # (1, 1, 784)
    ctx_i_b = es.ctx[i:i + 1]         # (1, 16, 784)
    truth = np.asarray(qry_i[0, 0])

    p_trained = _load("exp20")
    p_frozen = _load("exp24_best")
    eval_fn = make_eval(rn, mask_j)

    def net_pred(params):
        pred, _ = eval_fn(params, ctx_i_b, qry_i)
        return np.asarray(pred[0, 0])

    method_preds = [
        ("always predict\nthe mean image", np.asarray(mean_img)),
        ("near-copy look-up\n(tau=0.003)", np.asarray(_soft_lookup(ctx_i_b, qry_i, mask_j, 0.003))[0, 0]),
        ("blend look-up\n(tau=0.03)", np.asarray(_soft_lookup(ctx_i_b, qry_i, mask_j, 0.03))[0, 0]),
        ("fully-trained network\n(end of training)", net_pred(p_trained)),
        ("frozen-layer network\n(best checkpoint)", net_pred(p_frozen)),
    ]

    # Explicit inch-based layout: every axes rect is placed by a fixed
    # inches-from-top / inches-from-left budget, converted to figure
    # fraction only inside add_tile/add_text. One code path builds both
    # the top row (context + query) and the bottom row (predictions) —
    # this is deliberately NOT gridspec + tight_layout, which is what left
    # a large dead band between the two rows in an earlier version.
    FIG_W, FIG_H = 13.5, 4.65
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def add_text(x_in, y_top_in, s, ha="center", **kw):
        fig.text(x_in / FIG_W, 1.0 - y_top_in / FIG_H, s, ha=ha, va="top", **kw)

    def add_tile(x0_in, y_top_in, size_in, edgecolor, lw):
        rect = [x0_in / FIG_W, 1.0 - (y_top_in + size_in) / FIG_H,
                size_in / FIG_W, size_in / FIG_H]
        ax = fig.add_axes(rect)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(edgecolor); s.set_linewidth(lw)
        return ax

    add_text(FIG_W / 2, 0.05, "One held-out episode, shown completely: the true image "
             "is not among these 16 context images (the answer must be predicted)",
             fontsize=10.5)

    # ── top row: 16 context tiles + query tile ──────────────────────────────
    # A dedicated label band (LABEL_Y to TOP_Y, 0.32in) sits ABOVE the tile
    # row, not sharing a coordinate with it — the previous version nudged the
    # label 0.08in above the tile's top edge, which is less than one line of
    # text and let the tiles overstrike the label.
    LABEL_Y = 0.40
    TOP_Y, TILE_TOP = LABEL_Y + 0.32, 0.68
    add_text(0.15, LABEL_Y, "context — 16 nearest neighbours", ha="left",
              fontsize=8.5, color="#2a78d6")
    x = 0.30
    for k in range(16):
        ax = add_tile(x, TOP_Y, TILE_TOP, "#2a78d6", 1.1)
        ax.imshow(ctx_i[k].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        x += TILE_TOP + 0.06
    x += 0.30   # visual break before the query tile
    add_text(x, LABEL_Y, "query, bottom hidden", ha="left", fontsize=8.5,
              color="#eb6834")
    q_shown = truth * (1 - mask) + 0.5 * mask
    ax_q = add_tile(x, TOP_Y, TILE_TOP, "#eb6834", 1.6)
    ax_q.imshow(q_shown.reshape(28, 28), cmap="gray", vmin=0, vmax=1)

    # ── bottom row: true image + five method predictions ────────────────────
    TITLE_Y, TITLE_H = TOP_Y + TILE_TOP + 0.25, 0.50
    IMG_Y, TILE_BOT = TITLE_Y + TITLE_H, 1.80
    ERR_Y = IMG_Y + TILE_BOT + 0.05

    bot_specs = [("true image", None, "#c1121f")]
    for label, pred in method_preds:
        bot_specs.append((label, pred, "#3a3936"))
    n = len(bot_specs)
    total_w = n * TILE_BOT + (n - 1) * 0.35
    x = (FIG_W - total_w) / 2
    for label, pred, edgecolor in bot_specs:
        ax = add_tile(x, IMG_Y, TILE_BOT, edgecolor, 1.4)
        if pred is None:
            img, err_txt = truth, ""
        else:
            img = truth * (1 - mask) + pred * mask
            e = float((((pred - truth) ** 2) * mask).sum() / mask.sum())
            err_txt = f"squared error = {e:.3f}"
        ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        add_text(x + TILE_BOT / 2, TITLE_Y, label, fontsize=8.3)
        if err_txt:
            add_text(x + TILE_BOT / 2, ERR_Y, err_txt, fontsize=8.5)
        x += TILE_BOT + 0.35

    url = save_matplotlib_figure(f"{PROJ}_r11_worked_episode_v3", fig, format="png", dpi=150)
    plt.close(fig)
    return url


# ── NEW Figure: the four conditions, as pictures ────────────────────────────
def fig_conditions_matrix():
    """The 2x2 condition matrix, drawn from one real episode per cell rather
    than described in words. Columns: whether the query's true image is one
    of the 16 context images (present, left) or not (absent, right). Rows:
    whether the query image is drawn from the training pool the network's
    context-reading layers were exposed to during training (seen, top) or
    from MNIST's held-out test split (novel, bottom). Each cell shows four
    of that episode's 16 context images plus the query with its bottom half
    hidden; a green outline on a context tile marks it as the query's own
    image (present cells only)."""
    rn = Run(exp_name="", name="", M=16, Q=1, mask_rows=14, cfg=CFG)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    mean_img = pools["train"].mean(0)
    ev = evalsets.build(pools, mask, 16, 1, 512, mean_img, ctx_mode="knn", labels=labels)

    cells = [
        ("A_seen_present", "A — seen pool, target present\n(the training condition)"),
        ("B_novel_present", "B — novel pool, target present\n(does retrieval transfer to unseen images?)"),
        ("C_seen_absent", "C — seen pool, target absent\n(nothing to recall, familiar pool)"),
        ("D_novel_absent", "D — novel pool, target absent\n(the generalisation test)"),
    ]

    cells = [
        ("A_seen_present", "A — seen, present"),
        ("B_novel_present", "B — novel, present"),
        ("C_seen_absent", "C — seen, absent"),
        ("D_novel_absent", "D — novel, absent"),
    ]

    # Explicit inch-based layout, one code path for all four cells (the
    # earlier version positioned titles via a post-hoc get_position() call
    # that only took effect for the bottom row — the top row's title
    # overstruck the suptitle). Every text and every axes rect below is
    # placed from a fixed inches-from-top / inches-from-left budget with
    # reserved gaps, so a collision is structurally impossible.
    FIG_W, FIG_H = 11.5, 3.5
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def add_text(x_in, y_top_in, s, ha="center", **kw):
        fig.text(x_in / FIG_W, 1.0 - y_top_in / FIG_H, s, ha=ha, va="top", **kw)

    def add_tile(x0_in, y_top_in, size_in, edgecolor, lw):
        rect = [x0_in / FIG_W, 1.0 - (y_top_in + size_in) / FIG_H,
                size_in / FIG_W, size_in / FIG_H]
        ax = fig.add_axes(rect)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(edgecolor); s.set_linewidth(lw)
        return ax

    add_text(FIG_W / 2, 0.08, "The four conditions", fontsize=13)

    ROW_Y = [0.55, 2.00]      # title y (inches from top) for row 0 (A/B), row 1 (C/D)
    IMG_DY = 0.24             # gap between title baseline and image row
    TILE = 0.78
    COL_X0 = [0.35, 6.00]     # left edge (inches) for column 0 (A/C), column 1 (B/D)

    def place_cell(col_x0, title_y, cond, title):
        img_y = title_y + IMG_DY
        es = ev[cond]
        present = "present" in cond
        ctx_i = np.asarray(es.ctx[0])
        qry_i = np.asarray(es.qry[0, 0])
        tgt_idx = int(es.tgt_idx[0, 0]) if present else -1

        add_text(col_x0 + 2.5 * (TILE + 0.08), title_y, title, fontsize=10.5)

        x = col_x0
        for k in range(4):
            edge = "#2a9d3f" if k == tgt_idx else "#2a78d6"
            lw = 2.6 if k == tgt_idx else 1.0
            ax = add_tile(x, img_y, TILE, edge, lw)
            ax.imshow(ctx_i[k].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            x += TILE + 0.08
        x += 0.05
        add_text(x + TILE / 2, img_y + TILE / 2 - 0.15, "+12\nmore", fontsize=7.5,
                  color="#6b6a66")
        x += TILE + 0.20
        q_shown = qry_i * (1 - mask) + 0.5 * mask
        ax_q = add_tile(x, img_y, TILE, "#eb6834", 1.6)
        ax_q.imshow(q_shown.reshape(28, 28), cmap="gray", vmin=0, vmax=1)

    place_cell(COL_X0[0], ROW_Y[0], *cells[0])
    place_cell(COL_X0[1], ROW_Y[0], *cells[1])
    place_cell(COL_X0[0], ROW_Y[1], *cells[2])
    place_cell(COL_X0[1], ROW_Y[1], *cells[3])

    url = save_matplotlib_figure(f"{PROJ}_r11_conditions_matrix_v2", fig, format="png", dpi=150)
    plt.close(fig)
    return url


# ── NEW Figure: completions across all four conditions ──────────────────────
def fig_four_condition_completions():
    """The 2x2 condition matrix, this time showing what the two networks
    actually predict rather than just the raw episode setup. One block per
    condition (A seen/present top-left, B novel/present top-right,
    C seen/absent bottom-left, D novel/absent bottom-right). Within each
    block, 4 columns chosen at fixed percentiles (10th, 35th, 65th, 90th) of
    the blend look-up's (tau=0.03) own per-sample error within that
    condition — a model-free difficulty ranking computed separately per
    condition, never file order. 3 rows per block: the true image, the
    fully-trained network's prediction, and the frozen-layer network's
    prediction (a mean-image row was dropped to keep 4 blocks legible at
    once). Every predicted panel composites the true visible top half back
    in and is labelled with its own squared error on the hidden half only.
    Context throughout is the query's 16 nearest neighbours, matching the
    rest of this report's main results."""
    rn = Run(exp_name="", name="", M=16, Q=1, mask_rows=14, cfg=CFG)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    mask_j = jnp.array(mask)
    mean_img = pools["train"].mean(0)
    ev = evalsets.build(pools, mask, 16, 1, 512, mean_img, ctx_mode="knn", labels=labels)

    p_trained = _load("exp20")
    p_frozen = _load("exp24_best")
    eval_fn = make_eval(rn, mask_j)

    def net_pred(params, ctx_i_b, qry_i):
        pred, _ = eval_fn(params, ctx_i_b, qry_i)
        return np.asarray(pred[0, 0])

    PERCENTILES = [0.10, 0.35, 0.65, 0.90]
    conditions = [
        ("A_seen_present", "A — seen, present"),
        ("B_novel_present", "B — novel, present"),
        ("C_seen_absent", "C — seen, absent"),
        ("D_novel_absent", "D — novel, absent"),
    ]

    # Explicit inch-based layout, same discipline as the two figures above:
    # every tile and label placed from a fixed inches-from-top /
    # inches-from-left budget, so nothing can collide or overstrike.
    FIG_W, FIG_H = 11.3, 9.3
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def add_text(x_in, y_top_in, s, ha="center", **kw):
        fig.text(x_in / FIG_W, 1.0 - y_top_in / FIG_H, s, ha=ha, va="top", **kw)

    def add_tile(x0_in, y_top_in, size_in, edgecolor, lw):
        rect = [x0_in / FIG_W, 1.0 - (y_top_in + size_in) / FIG_H,
                size_in / FIG_W, size_in / FIG_H]
        ax = fig.add_axes(rect)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(edgecolor); s.set_linewidth(lw)
        return ax

    add_text(FIG_W / 2, 0.10,
              "Completions in all four conditions (columns: fixed difficulty percentiles within each condition)",
              fontsize=12)

    TILE = 0.92
    ROW_LABEL_W = 0.75
    COL_GAP = 0.10
    BLOCK_W = ROW_LABEL_W + 4 * TILE + 3 * COL_GAP
    BLOCK_H_TOP = 0.55       # condition title
    ROW_GAP = 0.06
    ERR_H = 0.20
    # 3 tiles, 2 inter-row gaps, and an error-label strip under every row
    # (rows 1 and 2 use it; row 0's is left blank rather than special-cased,
    # which keeps row spacing uniform and collision-proof).
    BLOCK_TILES_H = 3 * TILE + 2 * ROW_GAP + 3 * ERR_H
    BLOCK_H = BLOCK_H_TOP + BLOCK_TILES_H

    # The gap between the two blocks in a row must exceed ROW_LABEL_W, or
    # the right block's row label (right-aligned on its own x0) bleeds left
    # into the left block's tiles — this was 0.55in in an earlier version
    # and collided.
    BLOCK_GAP = 1.05
    BLOCK_X0 = [0.30, 0.30 + BLOCK_W + BLOCK_GAP]
    BLOCK_Y0 = [0.55, 0.55 + BLOCK_H + 0.30]

    row_specs = [("true image", None), ("fully-trained", p_trained), ("frozen-layer", p_frozen)]

    def place_block(x0, y0, cond, title):
        es = ev[cond]
        blend = np.asarray(_soft_lookup(es.ctx, es.qry, mask_j, 0.03))
        per = (((blend - np.asarray(es.qry)) ** 2) * mask)[:, 0, :].sum(-1) / mask.sum()
        order = np.argsort(per)
        cols = [int(order[int(p * (len(order) - 1))]) for p in PERCENTILES]

        add_text(x0 + BLOCK_W / 2, y0, title, fontsize=11.5, fontweight="bold")
        y_rows = y0 + BLOCK_H_TOP
        for r, (rlabel, params) in enumerate(row_specs):
            y_tile = y_rows + r * (TILE + ROW_GAP + ERR_H)
            fig.text(x0 / FIG_W, 1.0 - (y_tile + TILE / 2) / FIG_H, rlabel,
                      ha="right", va="center", fontsize=8.5)
            for c, ep_i in enumerate(cols):
                x_tile = x0 + ROW_LABEL_W + c * (TILE + COL_GAP)
                truth = np.asarray(es.qry[ep_i, 0])
                if params is None:
                    img, err_txt = truth, ""
                else:
                    ctx_b = es.ctx[ep_i:ep_i + 1]
                    qry_b = es.qry[ep_i:ep_i + 1]
                    pred = net_pred(params, ctx_b, qry_b)
                    img = truth * (1 - mask) + pred * mask
                    e = float((((pred - truth) ** 2) * mask).sum() / mask.sum())
                    err_txt = f"e={e:.3f}"
                edge = "#c1121f" if params is None else "#3a3936"
                ax = add_tile(x_tile, y_tile, TILE, edge, 1.0)
                ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                if err_txt:
                    add_text(x_tile + TILE / 2, y_tile + TILE + 0.02, err_txt, fontsize=7.2)

    place_block(BLOCK_X0[0], BLOCK_Y0[0], *conditions[0])
    place_block(BLOCK_X0[1], BLOCK_Y0[0], *conditions[1])
    place_block(BLOCK_X0[0], BLOCK_Y0[1], *conditions[2])
    place_block(BLOCK_X0[1], BLOCK_Y0[1], *conditions[3])

    url = save_matplotlib_figure(f"{PROJ}_r11_four_condition_completions_v2", fig, format="png", dpi=150)
    plt.close(fig)
    return url


def main():
    url_worked = fig_worked_episode()
    url_matrix = fig_conditions_matrix()
    url_four_cond = fig_four_condition_completions()
    print("fig_worked_episode:", url_worked)
    print("fig_conditions_matrix:", url_matrix)
    print("fig_four_condition_completions:", url_four_cond)

    md = f"""# Copying costs generalisation

This report answers one question. A network is trained to fill in a missing
part of an image by looking at sixteen example images. What does it actually
learn to do, and when does that learning help it handle an image it has
never seen before?

The training signal used here rewards only one thing: reproducing an answer
that is already sitting in the sixteen examples. A network that gets very
good at that reward becomes worse, not better, at the harder case. In that
harder case the answer is not sitting there at all, and has to be worked out
instead. A version of the same network is kept from specialising in that
reward, by leaving most of its parameters frozen at their random starting
values. That frozen version ends up doing the working-out part better than
any version of the network that was allowed to train freely.

Put in numbers: a fully-trained network's best measured performance on
held-out examples where the answer must be worked out is a normalised error
of **0.505**. (0.0 is a perfect prediction. 1.0 is no better than always
guessing the average training image. So 0.505 is roughly halfway between
useless and perfect.) Left training longer, the same network's error on
those same held-out examples rises to 0.666 — it gets worse with more
training, not better. A version of the network with its context-processing
layers frozen at random initialisation reaches **0.471**. That is better
than the fully-trained network manages at its best. It is also better than a
reference computation with no learned parameters at all, which simply
averages the sixteen example images and reaches 0.552 at its best setting.
And it is better than a ridge-regression baseline that is not shown the
sixteen examples at all, which reaches 0.631. The rest of this report builds
these numbers up from the task itself, so that by the end each one is a
plain measurement rather than a claim to be taken on trust.

## The task

Every example given to the network is a short sequence. First come sixteen
images of handwritten digits from the MNIST dataset, each a 28-by-28 grid of
pixels flattened into a list of 784 numbers. Then comes a seventeenth image
— the query — with its bottom fourteen rows (392 of its 784 pixels) replaced
by a neutral grey and hidden from the network. The network's job is to
output values for those 392 hidden pixels. Its answer is scored only on
that hidden region; the visible top half of the query is given for free and
is not part of the prediction task.

![the task: 16 context images write into a fixed-size state, the query reads it, and only the greyed region is scored]({URL_TASK_DIAGRAM})

The task is worth building this way because it can be set up in two
different ways that look identical from the network's point of view, but
call for opposite strategies. In one setup, the query's true, complete image
is literally one of the sixteen context images. The network has already
been shown the answer, and the correct move is to find and reproduce it. In
the other setup, the query's true image is not among the sixteen — nothing
in the context is a copy of the answer. The best a network can do is notice
which context images look similar to the visible top half of the query.
It can use them to make an informed guess about what the hidden bottom half
probably looks like. The first setup tests whether a network can look
something up. The second tests whether it can work something out from
related but non-identical examples. A single architecture, trained one way,
is asked to do both. The interesting question is whether getting good at the
first makes a network better or worse at the second.

## A worked example, from raw pixels to five guesses

Before any aggregate numbers, it helps to see one full episode end to end,
with nothing summarised away. The figure below shows a single held-out
example where the query's true image is not among its sixteen context
images. This is the harder of the two setups described above, and the one
this report cares about most. The example shown is not cherry-picked: it is
the one sitting at the midpoint (the 50th percentile) of a difficulty
ranking built without reference to any trained network. Specifically, the
ranking uses the per-example error of the blend reference computation
defined in the next section, so this example is neither an easy win nor a
worst case.

![One held-out episode, shown completely. Top: all 16 context images (the query's 16 nearest neighbours by pixel distance on its visible half; the true query image is not among them) and the query with its bottom half hidden. Bottom: the true image, then five methods' predictions for the missing 392 pixels, each with the true visible half composited back in and its own raw squared error on the hidden half printed underneath. The episode is the median-difficulty example (p50) by a model-free difficulty ranking — the blend look-up's own per-sample error — not hand-picked and not chosen by any trained network's error.]({url_worked})

Reading left to right along the bottom row: always predicting the average
training-set image ignores the sixteen context images entirely. It produces
a uniform grey smear, with the highest error of the five methods on this
example. The near-copy look-up puts almost all its weight on whichever
single context image looks most similar to the query's visible top half.
It produces a sharp, digit-like completion, because it commits fully to one
neighbour's answer. The blend look-up spreads its weight across several
similar-looking neighbours instead, and produces a softer, more averaged
completion. The fully-trained network and the frozen-layer network each
produce their own completions. Their errors on this one example should be
read as one data point, not as the pattern. The aggregate comparisons later
in this report, over 512 such episodes, are what the rest of the argument
rests on. What this figure is for is showing, concretely, what "guessing the
hidden half of a digit from sixteen similar-looking digits" actually looks
like as pixels. That is before any of it gets reduced to a single number.

## How the network works

The network is a four-layer recurrent architecture built around
linear-attention layers of a delta-rule / KDA type. It keeps a
matrix-valued internal state that context images write into one at a time,
and that the query then reads from once, after all sixteen context images
have been processed. It has about 4.03 million parameters in total, arranged
as d_model 256 with 4 attention heads of dimension 64 each.

Each layer's state update, for every context token in turn, is:

```
S <- S * diag(alpha_t)              # per-channel forgetting
vhat = S k_t                        # what the state currently holds at this key
e = beta_t * (v_t - vhat)           # correction toward the true value
S <- S + e k_t^T                    # write
o_t = S q_t / sqrt(d_k)             # read
```

`S` is the state, one 64-by-64 matrix per head, per layer. Each context
token computes a key `k_t` and value `v_t`. It checks what the state
currently predicts for that key (`vhat`), and writes a correction toward the
token's true value, scaled by a learned gate `beta_t`. The query token never
writes — it only performs the final read, `o_t = S q_t`, once the state has
finished absorbing all sixteen context images.

The state's total size is fixed regardless of how many images were written
into it: 4 heads times 64 times 64 numbers per layer, which is 16,384
numbers per layer. That is fewer than the 16 images times 784 pixels —
12,544 numbers — that make up the raw context. So the state cannot simply be
a lossless copy of the sixteen images. Whatever it captures about them has
to be a compressed summary, not a verbatim record.

Pixel predictions are produced by a small output network reading off the
state after the query's read. It is worth being explicit about one thing
that is absent from this architecture: there is no softmax anywhere in it.
Nothing in the pipeline computes an explicit weighted average over the
sixteen context images, the way a conventional attention mechanism would.
Any tendency for the network's output to resemble one particular context
image, or to resemble a blend of several, has to be an emergent effect of
training. It is something the delta-rule state update above produces on its
own, not a mechanism built in the way it is in the reference computation
described next. This distinction matters later. Everywhere below that a
"temperature" is used to describe the network, it is a description fitted
after the fact, not something the network has any explicit control over.

## The four conditions

Every evaluation below sits in one of four conditions. These are formed by
crossing two independent choices. The first is whether the query's true
image is present in the sixteen-image context or absent from it. The second is whether the query image itself comes from the pool of images
the network's context-processing layers were exposed to during training
(seen). The alternative is that it comes from MNIST's held-out test split,
never seen during training in any role (novel). The figure below shows one
real episode from each of the four cells.

![The four conditions, drawn from one real held-out episode per cell rather than described in words. Columns: whether the query's true image is one of the 16 context images (present, left column) or not (absent, right column). Rows: whether the query is from the pool the network's context-reading layers were trained on (seen, top row) or from MNIST's held-out test split (novel, bottom row). Each cell shows 4 of that episode's 16 context images (a green border marks the one that is the query's own image, present cells only) and the query with its bottom half hidden.]({url_matrix})

Condition A, seen pool with the target present, is the condition the
training loss is actually computed on. The network is directly optimised to
do well here. Condition B, novel pool with the target present, asks a
different question. Does whatever lets the network solve A also work on an
image it never trained on, when the answer is still sitting in the context?
Condition C, seen pool with the target absent, asks what the network does
when the context carries no usable answer but the query itself is a
familiar kind of image. Condition D, novel pool with the target absent, is
the hardest and most important cell in this matrix. The query is
unfamiliar, and the context contains no copy of the answer, so a good score
here can only come from generalising — using the context's
similar-but-not-identical images to make a sound guess. Every headline
number in this report, including 0.505, 0.666 and 0.471, is a Condition D
number.

The figure below adds the piece those four cells describe in words but do
not yet show: what the network actually predicts, in each of the four
conditions, next to the true image.

![Completions across all four conditions, one block per cell (A top-left, B top-right, C bottom-left, D bottom-right), columns chosen at fixed percentiles (10th, 35th, 65th, 90th) of the blend look-up's own per-sample error within that condition — a model-free ranking, not file order. Rows within each block: the true image; the fully-trained network's prediction; the frozen-layer network's prediction; each predicted panel composites the true visible top half back in and is labelled with its own squared error on the hidden half.]({url_four_cond})

The present conditions (A and B, left column of blocks) are near-exact for
the fully-trained network across every column shown: the answer is sitting
in the context, and the network reproduces it. The absent conditions (C and
D, right column of blocks) are visibly not exact for either network, and the
frozen network's completions there look softer and less committed than the
trained network's. Seen versus novel makes far less visible difference than
present versus absent does. Blocks A and B look similar to each other, and
so do blocks C and D, while A and C (or B and D) look very different. That
ordering — present/absent dominating, seen/novel barely mattering — is the
same ordering the aggregate numbers in the rest of this report describe.

## Three reference computations

Before judging any trained network, it helps to have computations that are
not trained at all. Then a network's score can be read against something
concrete, rather than in isolation. Three are used throughout.

The first is simply predicting the training-set average image for every
query, ignoring the sixteen context images entirely. This is the definition
of "no better than chance" used everywhere in this report: every error
number below is normalised so that this computation scores exactly 1.0. A
network beating 1.0 is doing something with the context that plain averaging
does not.

The second is a ridge regression that maps the query's visible top half to a
prediction of its hidden bottom half. It is fitted on the training pool but,
like the mean-image computation, never shown the sixteen context images at
evaluation time. It scores 0.631 on Condition D. Because it ignores the
context, this baseline draws a line. It shows how much of a network's
ability to predict a hidden region comes purely from knowing what digits
generally look like, independent of the sixteen context images. A network
that cannot beat 0.631 is not making effective use of its context at all.

The third is the reference computation used most throughout this report: a
plain weighted average of the sixteen context images, with no learned
parameters. The weight given to each context image depends on how close it
is, pixel by pixel, to the visible top half of the query:

```
d_i    = || visible(query) - visible(context_i) ||^2 / n_visible_pixels
w_i    = softmax(-d / tau)_i
output = sum_i  w_i * context_i
```

The single parameter `tau` (temperature) controls how concentrated this
weighting is. At a low temperature such as 0.003, almost all the weight
lands on whichever single context image is closest to the query, so the
output is nearly an exact copy of that one image. At a higher temperature
such as 0.03, the weight spreads out over several near neighbours, and the
output becomes their blend. This reference computation matters for two
separate reasons. It is itself a baseline a trained network can be compared
against. It is also the only computation here that literally contains a
temperature, so it doubles as a ruler. Given a trained network's actual
output on a batch of episodes, it is possible to ask which value of `tau`
would have made this reference computation's output closest to the
network's own output. That fitted value is what "the network's temperature"
means every time it is used below. The network itself has no such parameter
and no explicit averaging step.

## What the training objective actually rewards

Before looking at what trained networks do, it is worth asking what the
objective they are trained on would reward if it could be optimised
perfectly, with no network in the way at all. This can be measured directly
by sweeping the reference computation's temperature. It is scored on both
Condition A (target present, the actual training condition) and Condition D
(target absent, the generalisation test), with no trained model involved.

![the temperature trade-off, no network involved]({URL_TRADEOFF})

Lowering `tau` from 0.03 to 0.003 moves the reference computation's weight
almost entirely onto the single nearest context image. That lowers its
error on the present condition from 0.372 to 0.013, a large gain. The gain
makes sense: a target that is literally present is best served by copying
it exactly. But the same change raises its error on the absent condition from
0.553 to 0.672, a real cost. A specific, confident, copied answer is
frequently the wrong digit when the true answer was never in the context to
copy. These two curves have opposite optimal temperatures. This is a
mathematical property of squared-error scoring applied to this data, not
something any trained network invents. Squared error is minimised, in
expectation, by the average of every plausible answer. So on the absent
condition it structurally prefers a blurred blend over a sharp, specific,
possibly-wrong guess, even though a plausible digit is a perfectly
reasonable thing to output. A confident, correctly-shaped digit that happens
to be the wrong digit is punished harder under squared error than a blur
that is wrong about everything equally. The training objective used
throughout this project is exactly this squared-error loss, computed only on
Condition A episodes. So the objective a network is trained on has an
unambiguous preference for sharp copying. It has no built-in reason to
prefer blending, because Condition D episodes never appear in the loss it
is optimising.

## What trained networks actually do

The reference computation above shows what an idealised, tau-controlled
averaging process would trade off. The next question is whether a trained
network — which has no explicit temperature and no explicit averaging
mechanism at all — moves along the same trade-off as it trains.

To answer that, each training checkpoint's output is fitted to the closest
matching temperature of the reference computation, using the same procedure
described above. Take the checkpoint's actual output on a batch of Condition
B (novel pool, target present) episodes, and find the `tau` whose reference
output is closest to it in squared error. This fitting is done on Condition
B rather than Condition D for a specific reason: the reference computation's
behaviour is only well defined when a target is actually present to copy or
not copy. Fitting a temperature to Condition D output does not give a stable
estimate the same way, because on Condition D the reference computation is
already just doing prediction, not exhibiting a clean copying/blending
trade-off.

![fitted temperature against held-out prediction error, trained and frozen checkpoints]({URL_TAU_VS_ERROR})

Early in training, the fully-trained network's fitted temperature is 0.03,
comparable to the blend end of the reference sweep. Its Condition D error is
at 0.505, the best value measured for this network at any point in
training. As training continues, the fitted temperature keeps falling: to
0.00053 by the end of training. That means the network's own output has
become nearly indistinguishable from copying a single context image, even
on episodes where doing so is the wrong move. Over the same stretch of
training, Condition D error rises from 0.505 to 0.666. On a version of the
task where the context is sixteen images unrelated to the query, rather than
nearest neighbours, the same drift goes further still: fitted temperature
0.0017, Condition D error 0.843. The direction is consistent across every
trained checkpoint measured. As training proceeds, the network's output
comes to resemble a copy of a single context image more and more closely.
Its ability to handle episodes where the answer is genuinely absent gets
worse, not better.

The frozen-layer network gives the same measurement a very different
answer. Training only the input embedding and the output layers — 0.60
million of the network's 4.03 million parameters — leaves the four
recurrent, context-processing layers at their random initialisation.
Training can still change how a pixel is turned into a token, and how the
state's final read is turned back into pixels. But it cannot change how the
state combines information from the sixteen context tokens in the first
place. Across every checkpoint of this frozen-layer run, the fitted
temperature stays at 0.03. It never drifts toward copying a single image,
because the part of the network that would have to drift is not being
trained. Condition D error under this run falls monotonically over training,
reaching 0.471 at its best checkpoint. That is below the fully-trained
network's best value (0.505). It is also below the best the reference
computation can reach at any fixed temperature over the same context
(0.552), and below the ridge baseline that ignores context entirely (0.631).

A frozen network that never drifts toward copying could, in principle, be
doing something much simpler than actually reading its context. For
instance, it could have learned a generic prior over what digits look like,
and be ignoring the sixteen context images altogether — that too would
produce a temperature that never moves. To rule this out, a control swaps in
a different, unrelated query's sixteen nearest neighbours as the context,
while keeping the original query fixed. If the frozen network were ignoring
its context, this swap should not matter. In fact it drops the frozen
network's Condition D accuracy from 0.471 to 0.763, close to the level of
having no useful context at all. The frozen network is reading the context
it is actually given, not falling back on a generic prior independent of it.

Two things have now been established. The training objective itself rewards
sharpening toward a copy of one context image, and punishes that same
sharpening on the harder, target-absent case. A network free to train all
its layers follows exactly that gradient. A network prevented from making
the same drift never suffers the same degradation.

## What the completions actually look like

The temperature and error numbers above describe an aggregate tendency
across hundreds of held-out episodes. This section shows what that tendency
looks like in individual predictions, first on the easier setup where the
answer is present, then on the harder one where it is absent. Both figures
below use the query's sixteen nearest neighbours as context, and the same
six queries appear in both. The queries are chosen at fixed percentiles
(5th, 23rd, 41st, 59th, 77th, 95th) of a model-free difficulty ranking, not
by hand and not by any trained network's own error. Every predicted panel
composites the network's prediction for the hidden 392 pixels back onto the
query's true visible top half. This is needed because the network also
emits values for the visible pixels, which are never scored and would be
misleading to show unmodified.

![Six queries whose true image is one of the 16 unrelated context images. Rows, top to bottom: the true image; what the network is given (bottom half hidden); always predicting the mean training image; a model-free nearest-match look-up at temperature 0.03; the fully-trained network; the frozen-layer network. Each panel shows the true visible half composited with that method's predicted hidden half, labelled with its own raw squared error on the hidden half only.]({URL_RECON_PRESENT})

When the answer is present in the context, the fully-trained network does
not merely approximate it. Across all six example columns its squared error
on the hidden half is 0.00. This is exactly the copying behaviour the
training objective rewards, executed essentially perfectly.

![The same six queries and row order as above, but the true image is now not in the context — the answer must be predicted rather than copied.]({URL_RECON_ABSENT})

When the answer is absent, the two networks visibly diverge in style. The
fully-trained network does not blur its guess toward an average digit
shape. It commits to a specific, confident, and often wrong completion.
Examples seen: a plausible-looking 9 where the true digit was something
else, a curled tail that turns a 7 into something 9-like, and a doubled
stroke that changes what a 2 looks like. The frozen-layer network's
completions are visibly softer and
less committed, and score a lower squared error at four of the six example
columns. At the single hardest column of the six, the ordering reverses and
the fully-trained network scores marginally lower error there. The look-up
row on this figure is also informative in its own right: at the two hardest
columns it scores worse than simply predicting the average image. That means
the sixteen nearest-neighbour context images, for those two particular
queries, happened to carry essentially no usable information about the true
hidden region.

## Which metric decides the winner

Everything up to this point has scored predictions with squared error. It is
worth asking whether a different, reasonable way of judging a completion's
quality would agree with that verdict. Squared error's structural preference
for blur is established above, and it means a low squared-error score is not
automatically the same thing as a completion that looks like a real,
correctly-identified digit.

Two further metrics are introduced for this purpose, and each needs its own
definition before its numbers mean anything. **Realism** measures, for a
predicted hidden half, its per-pixel distance to the single closest real
hidden half found anywhere in the training pool. It asks only "does this
look like some real digit's bottom half", never asking whether it is the
right digit. Lower is more realistic, and there is no upper bound other
than whatever the least realistic completion in a given batch happens to
score. **Digit identity** takes that same nearest real training match and
checks whether its label agrees with the query's own true label. It asks the
question realism cannot, namely whether the completion resembles the correct
digit rather than merely some digit. Its natural ceiling is set by the true
image itself: even the true, unaltered hidden half is not always closest to
a training example sharing its own label. Its closest match sometimes
belongs to a different digit that happens to look similar, so the true image
scores 0.869 rather than 1.000. That 0.869 is the honest ceiling this metric
can reach, not a network failing to hit 1.0.

Squared error is minimised by an average over plausible answers, established
earlier in this report. Realism and digit identity are not. A metric based
on nearest-neighbour distance to real images has no preference for
blending. A metric based on matching a discrete label has no preference for
hedging between several possible digits. Running the two look-up
temperatures used throughout this report through all three metrics shows the
predicted disagreement directly:

| temperature | squared error (lower better) | realism (lower better) | digit identity (higher better, ceiling 0.869) |
|---|---|---|---|
| tau=0.003 (near-copy) | 0.672 | 0.0154 | 0.805 |
| tau=0.03 (blend) | 0.553 | 0.0180 | 0.756 |

![the inversion: the two look-up temperatures on all three metrics]({URL_INVERSION})

The blend wins on squared error, 0.553 against 0.672. The near-copy wins on
both of the other two metrics: realism 0.0154 against 0.0180, and digit
identity 0.805 against 0.756. This is the same underlying computation, with
one parameter changed. Its ranking against the alternative fully inverts,
depending only on which metric is used to read the result.

The same three metrics, computed on the trained and frozen networks over 512
held-out Condition D episodes, extend the table:

| | squared error | realism | digit identity (ceiling 0.869) |
|---|---|---|---|
| true image (ceiling) | 0.000 | 0.0161 | 0.869 |
| mean image (blur floor) | 1.000 | 0.0360 | 0.113 |
| look-up, tau=0.003 (near-copy) | 0.672 | 0.0154 | 0.805 |
| look-up, tau=0.03 (blend) | 0.553 | 0.0180 | 0.756 |
| trained network, best checkpoint | 0.505 | 0.0169 | 0.729 |
| trained network, end of training | 0.666 | 0.0172 | 0.758 |
| frozen-layer network, best checkpoint | 0.471 | 0.0166 | 0.744 |
| frozen-layer network, end of training | 0.474 | 0.0165 | 0.738 |

![realism against squared error, every row in the table]({URL_FRONTIER})

The finding that matters most in this table is this: no trained network, in
any row, beats the model-free near-copy look-up on either of the two
metrics that do not structurally reward blur. That look-up's own scores are
realism 0.0154 and digit identity 0.805.
This holds for the best and the final checkpoints of both the
fully-trained and the frozen-layer networks. That is four network rows, all
four worse on realism, all four worse on digit identity, than simply copying
the single most similar context image. On squared error alone, the frozen
network's best checkpoint (0.471) is the best result in the whole table,
including the near-copy look-up (0.672) and the blend look-up (0.553). On
realism and digit identity, a computation with no learned parameters at all
still wins.

One entry in this table looks backwards at first glance and is worth
addressing directly: the near-copy look-up scores a better realism (0.0154)
than the true image itself (0.0161). This is not a measurement error. A
copied training image is, by construction, a training image, so its
distance to the nearest training image is at or near zero. A held-out
query's own true hidden half is a real digit too, but it is drawn from
MNIST's test split, not the training pool. Its single nearest match among
training images sits, on average, slightly further away than an exact copy
would. The same effect is what keeps digit identity below a perfect 1.0
even for the true image, as already noted above.

Finally, six completions of the same query set are shown below in pixels,
this time with both squared error and realism printed under each panel. That
way the disagreement between the two metrics is visible example by example,
rather than only in the aggregate table.

![six completions of the same six queries, chosen at fixed percentiles of a model-free difficulty measure, true visible half composited back in, each labelled with its squared error (green, e) and realism (cyan, r)]({URL_NN_COMPLETIONS})

No individual panel is barred from scoring well on both measures at once —
several do. But the aggregate table above is what decides which *strategy*
wins. Averaged over the full 512-episode set, the blend look-up wins on
squared error, and the near-copy look-up wins on realism and digit
identity. That is a statement about the two strategies as a whole, not a
claim about
any single panel shown here.

## A geometric route to the same failure

Training is not the only thing that can push a computation toward
resembling a copy of one context image. The frozen-layer network's context
is normally built from the query's genuinely nearest neighbours. A separate
set of runs instead builds the context from progressively more distant
neighbours: the nearest, the 64th-nearest, the 512th-nearest, and finally
sixteen entirely unrelated images. The network's weights and its fitted
temperature (0.03 throughout, since these are all frozen-layer runs) never
change at all across this sweep.

![The dial: all four frozen runs. Left panel shows exact-match accuracy on the context item rising as the true image is deliberately placed further from its distractors; right panel shows the resulting degradation in held-out prediction accuracy over training, largest for the most separated context.]({URL_DIAL})

As the distractors are moved further away, exact-match identification
accuracy on which context item is the target rises steadily. It goes 0.295
at the nearest-neighbour setting, 0.527 at the 64th-nearest, 0.889 at the
512th-nearest, and 0.988 with unrelated images. This matches how the reference
computation itself would behave at a fixed temperature if only the distances
changed. When one context item sits much closer than the rest, even a
fixed-temperature weighted average ends up putting most of its weight on
that one item. That happens simply because the distances driving the
softmax have become more separated, not because the temperature has moved.
In the same order, held-out prediction accuracy degrades progressively more
over training: +0.002, +0.007, +0.094, +0.208 (each figure is the rise in
Condition D error from the run's best checkpoint to its final one). Moving
toward resembling a copy of one image, and the generalisation cost that
follows it, can therefore be produced two different ways in this project. It
can come from training, as shown earlier, or from context geometry alone,
with no weights changing at all.

## What this does not establish

**A capacity confound.** The frozen-layer network trains 0.60 million
parameters; the fully-trained network trains all 4.03 million. Every
comparison in this report between "frozen" and "fully-trained" is therefore
also a comparison between fewer and more trainable parameters. This report
cannot separate "kept from resembling a copy of one image" from "simply has
less capacity to overfit" as the explanation for the frozen network's
advantage. Settling this would need a frozen-layer network whose trainable
parameter count is matched to the fully-trained one some other way. One
option is freezing a different subset of layers of the same total size —
which has not been run.

**An unexplained residual.** One configuration does not fit the account
built up across this report: the frozen-layer network run on entirely
unrelated-image context never develops a fitted temperature away from 0.03.
By the account above, it should therefore not degrade — yet its held-out
error still rises from 0.570 to 0.778 over training, the largest degradation
of any frozen run measured. Coming to resemble a copy of one image is
evidently sufficient to cause degradation, since preventing it prevents the
degradation seen in the fully-trained network. But this one result shows it
is not the only mechanism that can cause it, and no alternative mechanism is
identified here. Settling this would need instrumenting what else changes
over training in this particular frozen run, which has not been done.

**Scope.** All of this is one dataset (MNIST), one architecture (a
four-layer delta-rule linear-attention network), and one masking pattern
(the bottom half of a 28-by-28 image). Nothing here has been checked against
a different dataset, a different recurrent architecture, or a different
missing-region shape. None of the numbers in this report should be assumed
to transfer to those settings without rerunning it.

## Sources

`results.jsonl` rows `exp1`, `exp20`, `exp24`, `exp26`, `exp27`, `exp28`,
`baselines_M16_r14`, `baselines_M16_r14_knn_Q1`, `effective_tau2`,
`ctx_ablation2`, `metrics_beyond_mse`. Temperature fitting:
`projects/recall-gen/scripts/effective_tau.py`. Metric computation:
`projects/recall-gen/scripts/metrics_beyond_mse.py`. Context construction:
`knn_offset` parameter in `projects/recall-gen/lib/train.py` and
`lib/evalsets.py`.
"""

    word_count = len(md.split())
    print("WORD COUNT (approx, includes headers/tables):", word_count)

    report_url = save_report(f"{PROJ}_report_11", md)
    print("REPORT:", report_url)

    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text(md)
    print("LOCAL FILE:", REPORT_MD_PATH)


if __name__ == "__main__":
    main()
