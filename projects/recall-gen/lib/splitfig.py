"""The report-12 figure, made domain-parametric.

Report 12 drew one picture and one bar chart: six blocks of completions —
three levels of novelty down, answer present/absent across — and the same six
blocks as numbers with the references that bound them. That figure is the
argument, and it says the same thing about garments and chess positions as it
does about digits, so it is written once here and called by each report's
generator rather than copied into it.

Nothing about a dataset appears below. The caller passes the domain, the three
novelty bands, and the networks to draw; `lib/domains.py` supplies the tiles.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT.parents[1]))

from .core import Cfg, masked_mse, identification
from . import domains, evalsets
from .train import Run, build_pools, build_mask, build_visible, make_eval
from shared_lib.media import save_matplotlib_figure

SPLIT = {
    "A_seen_present":  ("train",     True),
    "B_novel_present": ("held",      True),
    "C_seen_absent":   ("train",     False),
    "D_novel_absent":  ("held",      False),
    "E_same_present":  ("held_same", True),
    "F_same_absent":   ("held_same", False),
}

# (band label, present condition, absent condition), increasing novelty downward
BANDS = [
    ("A", "A_seen_present", "C_seen_absent"),
    ("B", "E_same_present", "F_same_absent"),
    ("C", "B_novel_present", "D_novel_absent"),
]

BLUE, ORANGE, GREEN = "#2f6fbf", "#e07a3c", "#3f9a6e"


def load_params(exp: str):
    with open(PROJECT / f"params_{exp}.pkl", "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def _soft_lookup(ctx, qry, mask_j, tau, vis=None):
    """Similarity-weighted average of the context items' hidden halves.

    Used here only to RANK episodes by difficulty. It involves no trained
    network, so the columns a figure shows are chosen without reference to any
    result the figure is about.
    """
    vis = (1.0 - mask_j) if vis is None else vis
    d = (((qry[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1) / vis.sum()
    return jnp.einsum("eqm,emp->eqp", jax.nn.softmax(-d / tau, -1), ctx)


def build(domain, mask_rows, train_cls, held_cls, cfg, ctx_mode="iid", Q=4,
          M=16, n_eval=512, seed=20260825):
    rn = Run(exp_name="", name="", domain=domain, M=M, Q=Q, mask_rows=mask_rows,
             cfg=cfg, train_digits=train_cls, held_digits=held_cls, conditions=SPLIT)
    pools, labels = build_pools(rn)
    mask = build_mask(rn)
    ev = evalsets.build(pools, mask, M, Q, n_eval, pools["train"].mean(0),
                        conditions=SPLIT, labels=labels, ctx_mode=ctx_mode, seed=seed,
                        vis=build_visible(rn))
    return rn, mask, ev


def oracle_id_acc(ev, mask, cond) -> float:
    """The identification accuracy a PERFECT prediction would score.

    Identification asks which of the M context items the output most resembles,
    on hidden coordinates only, and calls it a hit when that is the target. The
    metric silently assumes the M hidden halves are distinguishable. On sparse
    chess endgames they often are not — several context positions can have an
    identical, nearly empty queenside — so a network that reconstructs the
    target exactly still loses the tie-break and scores below 1.0.

    Feeding the TRUE target in as the prediction measures that ceiling. It is
    the reference point every id_acc in these reports is read against; chance,
    1/M, is the other end.
    """
    es = ev[cond]
    assert es.present, "identification is only defined where the target is present"
    acc, _ = identification(es.qry, es.ctx, es.tgt_idx, jnp.array(mask))
    return float(acc)


def score_all(params, domain, mask_rows, train_cls, held_cls, cfg,
              ctx_mode="iid", Q=4, M=16, n_eval=512, seed=20260825):
    """Every condition's nmse and id_acc for one checkpoint on one context type.

    `grid` draws these numbers but does not hand them back, and a report that
    quotes a 2x2 of train-context against test-context needs them as values, not
    as pixels. Same eval sets `grid` builds, so a quoted number and the red
    number in the matching figure are the same measurement.
    """
    rn, mask, ev = build(domain, mask_rows, train_cls, held_cls, cfg,
                         ctx_mode=ctx_mode, Q=Q, M=M, n_eval=n_eval, seed=seed)
    mask_j = jnp.array(mask)
    eval_fn = make_eval(rn, mask_j)
    out = {}
    for cond, es in ev.items():
        E = es.ctx.shape[0]
        se = hits = 0.0
        for i in range(0, E, 128):
            c, q = es.ctx[i:i + 128], es.qry[i:i + 128]
            pred, argmin = eval_fn(params, c, q)
            w = c.shape[0] / E
            se += w * float(masked_mse(pred, q, mask_j))
            if es.present:
                hits += w * float((argmin == es.tgt_idx[i:i + 128]).mean())
        out[cond] = {"nmse": se / es.mse_mean,
                     "id_acc": hits if es.present else float("nan")}
    return out


def nearest_distractor(ev, mask, cond):
    """Per episode: distance from the target to the closest OTHER context item.

    Identification picks the context item closest to the network's output on
    hidden coordinates. It therefore depends on two separate things — how well
    the network reconstructs, and how far the target sits from its nearest
    rival. This measures the second, with no network involved.

    Returned in units of the pool's own mean-image error, the same units every
    nmse in this project is quoted in, so a margin and a model error can be put
    on one axis.

    An earlier version of this measurement swept isotropic Gaussian noise on a
    perfect reconstruction instead. That does not work: independent noise over
    392 pixels adds almost the same offset to every candidate's distance, so the
    ranking barely moves and every pool stayed at 1.000 out to sigma 1.1. The
    quantity identification actually depends on is this one.
    """
    es = ev[cond]
    assert es.present, "identification is only defined where the target is present"
    mask_j = jnp.array(mask)
    ctx, qry = es.ctx, es.qry[:, :1]
    d = (((qry[:, :, None, :] - ctx[:, None, :, :]) ** 2) * mask_j).sum(-1) / mask_j.sum()
    d = np.array(np.asarray(d)[:, 0])                       # (E, M)
    E = d.shape[0]
    d[np.arange(E), np.asarray(es.tgt_idx)[:, 0]] = np.inf  # drop the target itself
    return d.min(-1) / es.mse_mean


def margin_cdf(name, series, marks=(), title=""):
    """Cumulative distribution of the nearest-distractor distance, one line per
    pool. `marks` is [(label, colour, x)] — where a network's own error sat.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for lab, colour, vals in series:
        v = np.sort(np.asarray(vals))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=1.9, color=colour,
                label=f"{lab}  (median {np.median(v):.2f})")
    # Staggered: two networks with similar error put their vertical rules a few
    # hundredths apart, and rotated labels at a common height overlap.
    for k, (lab, colour, x) in enumerate(marks):
        ax.axvline(x, color=colour, ls=":", lw=1.6)
        ax.text(x, 0.985 - 0.30 * k, lab, rotation=90, fontsize=7.2,
                color=colour, ha="right", va="top")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("distance from the target to its nearest rival in the context\n"
                  "(hidden pixels, in units of that pool's mean-image error)",
                  fontsize=9.5)
    ax.set_ylabel("fraction of episodes at or below", fontsize=9.5)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.4, loc="lower right")
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url


def grid(name, domain, mask_rows, train_cls, held_cls, cfg, nets, band_labels,
         headline, sub, legend, ctx_mode="iid", Q=4, pct=(0.10, 0.30, 0.50, 0.70, 0.90),
         tile=0.80, lab_w=1.30, n_eval=512):
    """Six blocks: three novelty bands down, answer present/absent across.

    Panel numbers are NORMALISED error — squared error over the hidden
    coordinates divided by the error of drawing the average training item for
    that same pool. Raw error would not be comparable down the figure, since
    each pool has its own normaliser. 1.00 means the panel is no better than
    the average item.

    Columns inside a block are episodes at fixed percentiles of the blend
    look-up's own per-episode error on the ABSENT condition of that band. That
    ranking uses no trained network, and because a band's present and absent
    conditions share their queries, a column is the same query in both blocks
    of its row. Predicted panels composite the true visible half back over the
    predicted hidden half.

    `nets` is a list of (row label, params). `band_labels` replaces BANDS' three
    placeholder labels with the domain's own wording.
    """
    rn, mask, ev = build(domain, mask_rows, train_cls, held_cls, cfg, ctx_mode, Q,
                         n_eval=n_eval)
    dom = domains.get(domain)
    board = dom.kind == "board"
    mask_j = jnp.array(mask)
    eval_fn = make_eval(rn, mask_j)

    def aggregate(params, es):
        """Score over every episode of the block, not just the ones drawn."""
        se, acc = 0.0, 0.0
        E = es.ctx.shape[0]
        for i in range(0, E, 128):
            c, q = es.ctx[i:i + 128], es.qry[i:i + 128]
            pred, _ = eval_fn(params, c, q)
            w = c.shape[0] / E
            se += w * float(masked_mse(pred, q, mask_j))
            if board:
                acc += w * domains.piece_accuracy(pred, q, mask, domain)
        return se / es.mse_mean, acc

    # LAB_W has to hold the longest per-row label at its own font size. It is a
    # parameter because the chess label carries a second number and overflowed
    # left out of its own block and across the tiles of the block beside it.
    LAB_W, COL_GAP, ROW_GAP, ERR_H = lab_w, 0.08, 0.05, 0.17
    TILE = tile
    NCOL = len(pct)
    BLOCK_W = LAB_W + NCOL * TILE + (NCOL - 1) * COL_GAP
    BLOCK_TITLE = 0.30
    NROW = 1 + len(nets)
    BLOCK_H = BLOCK_TITLE + NROW * TILE + (NROW - 1) * ROW_GAP + NROW * ERR_H
    BAND_LAB = 1.75
    GAP_X, GAP_Y = 0.55, 0.34
    # The header is as tall as its content, not a constant. A fixed 1.28 was
    # tuned for two legend lines; a third line pushed the caption straight into
    # the "answer IS in the context" block titles, which is invisible in the
    # file and obvious in the picture.
    Y_LEG, LEG_H, SUB_H = 0.44, 0.18, 0.16
    y_sub = Y_LEG + LEG_H * len(legend) + 0.06
    TOP = y_sub + SUB_H * (sub.count("\n") + 1) + 0.42
    FIG_W = BAND_LAB + 2 * BLOCK_W + GAP_X + 0.25
    FIG_H = TOP + 3 * BLOCK_H + 2 * GAP_Y + 0.15

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def txt(x, y, s, ha="center", **kw):
        fig.text(x / FIG_W, 1.0 - y / FIG_H, s, ha=ha, va="top", **kw)

    def tile_ax(x, y, edge):
        ax = fig.add_axes([x / FIG_W, 1.0 - (y + TILE) / FIG_H,
                           TILE / FIG_W, TILE / FIG_H])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(edge); sp.set_linewidth(1.0)
        return ax

    txt(FIG_W / 2, 0.16, headline, fontsize=12.5)
    for k, (line, colour) in enumerate(legend):
        txt(FIG_W / 2, Y_LEG + LEG_H * k, line, fontsize=8.8, color=colour)
    txt(FIG_W / 2, y_sub, sub, fontsize=8.8, color="#555", linespacing=1.5)

    for b, ((_, cond_p, cond_a), band_label) in enumerate(zip(BANDS, band_labels)):
        y0 = TOP + b * (BLOCK_H + GAP_Y)
        txt(BAND_LAB - 0.18, y0 + BLOCK_H / 2 - 0.30, band_label, ha="right",
            fontsize=10.5, fontweight="bold")

        es_a = ev[cond_a]
        blend = np.asarray(_soft_lookup(es_a.ctx, es_a.qry, mask_j, 0.03))
        per = (((blend - np.asarray(es_a.qry)) ** 2) * mask)[:, 0, :].sum(-1) / mask.sum()
        order = np.argsort(per)
        cols = [int(order[int(p * (len(order) - 1))]) for p in pct]

        for k, (cond, head) in enumerate([(cond_p, "answer IS in the context"),
                                          (cond_a, "answer is NOT in the context")]):
            x0 = BAND_LAB + k * (BLOCK_W + GAP_X)
            es = ev[cond]
            if b == 0:
                txt(x0 + LAB_W + (NCOL * TILE + (NCOL - 1) * COL_GAP) / 2, y0 - 0.30,
                    head, fontsize=10.5, fontweight="bold")
            for r, (rlabel, params) in enumerate([("true item", None)] + nets):
                y = y0 + BLOCK_TITLE + r * (TILE + ROW_GAP + ERR_H)
                lx = (x0 + LAB_W - 0.10) / FIG_W
                if params is None:
                    fig.text(lx, 1.0 - (y + TILE / 2) / FIG_H, rlabel,
                             ha="right", va="center", fontsize=8.6)
                else:
                    nm, ac = aggregate(params, es)
                    line = f"all {n_eval} episodes: {nm:.3f}"
                    if board:
                        line += f"   pieces {ac * 100:.0f}%"
                    fig.text(lx, 1.0 - (y + TILE / 2 - 0.09) / FIG_H, rlabel,
                             ha="right", va="center", fontsize=8.6)
                    fig.text(lx, 1.0 - (y + TILE / 2 + 0.10) / FIG_H, line,
                             ha="right", va="center", fontsize=7.6, color="#a03030")
                for c, i in enumerate(cols):
                    x = x0 + LAB_W + c * (TILE + COL_GAP)
                    truth = np.asarray(es.qry[i, 0])
                    if params is None:
                        img, lab, edge, hide = truth, "", "#c1121f", False
                    else:
                        pred, _ = eval_fn(params, es.ctx[i:i + 1], es.qry[i:i + 1])
                        pred = np.asarray(pred[0, 0])
                        img = domains.composite(truth, pred, mask)
                        raw = float((((pred - truth) ** 2) * mask).sum() / mask.sum())
                        lab = f"{raw / es.mse_mean:.2f}"
                        edge, hide = "#3a3936", False
                    ax = tile_ax(x, y, edge)
                    domains.draw(ax, img, domain, mask=mask, hidden_shading=hide)
                    if lab:
                        txt(x + TILE / 2, y + TILE + 0.015, lab, fontsize=7.4)

    url = save_matplotlib_figure(name, fig, format="png", dpi=150)
    plt.close(fig)
    return url


def grid_spec(domain, mask_rows, train_cls, held_cls, cfg, nets, band_labels,
              legend, ctx_mode="iid", Q=4, pct=(0.10, 0.30, 0.50, 0.70, 0.90),
              n_eval=512, M=16):
    """Everything the merged grid draws, as data — no drawing.

    Split out so the same figure can be rendered by matplotlib (`grid_merged`)
    or typeset by Typst (`lib/typstfig.tile_grid`) without the two computing
    different things and drifting apart.

    A pool's present and absent conditions share their queries by construction
    (see `lib/evalsets`), which is what lets one truth row serve both; the
    assertion below is load-bearing rather than defensive.
    """
    rn, mask, ev = build(domain, mask_rows, train_cls, held_cls, cfg, ctx_mode, Q,
                         M=M, n_eval=n_eval)
    dom = domains.get(domain)
    board = dom.kind == "board"
    mask_j = jnp.array(mask)
    eval_fn = make_eval(rn, mask_j)

    def aggregate(params, es):
        se, acc = 0.0, 0.0
        E = es.ctx.shape[0]
        for i in range(0, E, 128):
            c, q = es.ctx[i:i + 128], es.qry[i:i + 128]
            pred, _ = eval_fn(params, c, q)
            w = c.shape[0] / E
            se += w * float(masked_mse(pred, q, mask_j))
            if board:
                acc += w * domains.piece_accuracy(pred, q, mask, domain)
        return se / es.mse_mean, acc

    palette = [c for _, c in legend] + ["#333333"] * len(nets)
    bands = []
    for (_, cond_p, cond_a), band_label in zip(BANDS, band_labels):
        es_p, es_a = ev[cond_p], ev[cond_a]
        assert np.allclose(np.asarray(es_p.qry), np.asarray(es_a.qry)), (
            f"{cond_p} and {cond_a} no longer share queries; one truth row "
            "cannot stand for two different images")
        blend = np.asarray(_soft_lookup(es_a.ctx, es_a.qry, mask_j, 0.03))
        per = (((blend - np.asarray(es_a.qry)) ** 2) * mask)[:, 0, :].sum(-1) / mask.sum()
        order = np.argsort(per)
        cols = [int(order[int(p_ * (len(order) - 1))]) for p_ in pct]

        rows = [{"label": "true item", "colour": "#c1121f", "agg": None,
                 "tiles": [{"img": np.asarray(es_p.qry[i, 0]), "num": None}
                           for i in cols]}]
        for (rlabel, params), colour in zip(nets, palette):
            for tag, es in ((("answer IS in the context"), es_p),
                            (("answer is NOT there"), es_a)):
                nm, ac = aggregate(params, es)
                agg = f"all {n_eval} episodes: {nm:.3f}"
                if board:
                    agg += f"   pieces {ac * 100:.0f}%"
                tiles = []
                for i in cols:
                    truth = np.asarray(es.qry[i, 0])
                    pred, _ = eval_fn(params, es.ctx[i:i + 1], es.qry[i:i + 1])
                    pred = np.asarray(pred[0, 0])
                    raw = float((((pred - truth) ** 2) * mask).sum() / mask.sum())
                    tiles.append({"img": domains.composite(truth, pred, mask),
                                  "num": f"{raw / es.mse_mean:.2f}"})
                rows.append({"label": f"{rlabel}\n{tag}", "colour": colour,
                             "agg": agg, "tiles": tiles})
        bands.append({"label": band_label, "rows": rows})
    return {"domain": domain, "mask": mask, "bands": bands, "n_eval": n_eval}


def grid_merged(name, domain, mask_rows, train_cls, held_cls, cfg, nets,
                band_labels, headline, sub, legend, ctx_mode="iid", Q=4,
                pct=(0.10, 0.30, 0.50, 0.70, 0.90), tile=0.80, lab_w=1.30,
                n_eval=512, M=16):
    """`grid`, with the two blocks merged into one.

    `grid` draws a band as two side-by-side blocks — answer present, answer
    absent — each with its own "true item" row. But a pool's present and absent
    conditions SHARE their queries by construction (see `lib/evalsets`), so the
    truth row is drawn twice and the figure is twice as wide as it needs to be.

    Here each band is one block: a single truth row, then two rows per network,
    present above absent. The same query therefore appears once at the top of a
    column and the reader compares straight down it — which is the comparison
    the figure exists to make, and which the split layout put a block-width
    apart.

    The sharing is asserted rather than assumed: if the two conditions ever stop
    drawing the same queries, this collapses two different images into one and
    the figure would silently lie.
    """
    rn, mask, ev = build(domain, mask_rows, train_cls, held_cls, cfg, ctx_mode, Q,
                         M=M, n_eval=n_eval)
    dom = domains.get(domain)
    board = dom.kind == "board"
    mask_j = jnp.array(mask)
    eval_fn = make_eval(rn, mask_j)

    def aggregate(params, es):
        se, acc = 0.0, 0.0
        E = es.ctx.shape[0]
        for i in range(0, E, 128):
            c, q = es.ctx[i:i + 128], es.qry[i:i + 128]
            pred, _ = eval_fn(params, c, q)
            w = c.shape[0] / E
            se += w * float(masked_mse(pred, q, mask_j))
            if board:
                acc += w * domains.piece_accuracy(pred, q, mask, domain)
        return se / es.mse_mean, acc

    NCOL = len(pct)
    TILE, COL_GAP, ROW_GAP, ERR_H = tile, 0.08, 0.05, 0.17
    LAB_W = lab_w
    NROW = 1 + 2 * len(nets)                      # truth, then present/absent per net
    BLOCK_TITLE = 0.30
    BLOCK_H = BLOCK_TITLE + NROW * TILE + (NROW - 1) * ROW_GAP + NROW * ERR_H
    BLOCK_W = LAB_W + NCOL * TILE + (NCOL - 1) * COL_GAP
    BAND_LAB = 1.75
    GAP_Y = 0.40
    Y_LEG, LEG_H, SUB_H = 0.44, 0.18, 0.16
    y_sub = Y_LEG + LEG_H * len(legend) + 0.06
    TOP = y_sub + SUB_H * (sub.count("\n") + 1) + 0.42
    FIG_W = BAND_LAB + BLOCK_W + 0.25
    FIG_H = TOP + 3 * BLOCK_H + 2 * GAP_Y + 0.15

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def txt(x, y, s_, ha="center", **kw):
        fig.text(x / FIG_W, 1.0 - y / FIG_H, s_, ha=ha, va="top", **kw)

    txt(FIG_W / 2, 0.16, headline, fontsize=12.5)
    for k, (line, colour) in enumerate(legend):
        txt(FIG_W / 2, Y_LEG + LEG_H * k, line, fontsize=8.8, color=colour)
    txt(FIG_W / 2, y_sub, sub, fontsize=8.8, color="#555", linespacing=1.5)

    for b, ((_, cond_p, cond_a), band_label) in enumerate(zip(BANDS, band_labels)):
        y0 = TOP + b * (BLOCK_H + GAP_Y)
        es_p, es_a = ev[cond_p], ev[cond_a]
        assert np.allclose(np.asarray(es_p.qry), np.asarray(es_a.qry)), (
            f"{cond_p} and {cond_a} no longer share queries; the merged layout "
            "would draw one truth row for two different images")
        txt(BAND_LAB - 0.18, y0 + BLOCK_H / 2 - 0.30, band_label, ha="right",
            fontsize=10.5, fontweight="bold")

        blend = np.asarray(_soft_lookup(es_a.ctx, es_a.qry, mask_j, 0.03))
        per = (((blend - np.asarray(es_a.qry)) ** 2) * mask)[:, 0, :].sum(-1) / mask.sum()
        order = np.argsort(per)
        cols = [int(order[int(p_ * (len(order) - 1))]) for p_ in pct]

        x0 = BAND_LAB
        rows_spec = [("true item", None, es_p, "#c1121f")]
        # Each network's rows take the colour of its own legend line, so a row
        # label and the sentence explaining it match. `legend` entries are
        # (text, colour) pairs; the padding covers a caller passing fewer legend
        # lines than networks.
        palette = [c for _, c in legend] + ["#333333"] * len(nets)
        for (rlabel, params), colour in zip(nets, palette):
            rows_spec.append((f"{rlabel}\nanswer IS in the context", params, es_p, colour))
            rows_spec.append((f"{rlabel}\nanswer is NOT there", params, es_a, colour))

        for r, (rlabel, params, es, colour) in enumerate(rows_spec):
            y = y0 + BLOCK_TITLE + r * (TILE + ROW_GAP + ERR_H)
            lx = (x0 + LAB_W - 0.10) / FIG_W
            if params is None:
                fig.text(lx, 1.0 - (y + TILE / 2) / FIG_H, rlabel, ha="right",
                         va="center", fontsize=8.6)
            else:
                nm, ac = aggregate(params, es)
                line = f"all {n_eval} episodes: {nm:.3f}"
                if board:
                    line += f"   pieces {ac * 100:.0f}%"
                fig.text(lx, 1.0 - (y + TILE / 2 - 0.11) / FIG_H, rlabel,
                         ha="right", va="center", fontsize=8.4, color=colour,
                         linespacing=1.3)
                fig.text(lx, 1.0 - (y + TILE / 2 + 0.16) / FIG_H, line,
                         ha="right", va="center", fontsize=7.6, color="#a03030")
            for c, i in enumerate(cols):
                x = x0 + LAB_W + c * (TILE + COL_GAP)
                truth = np.asarray(es.qry[i, 0])
                if params is None:
                    img, lab, edge = truth, "", "#c1121f"
                else:
                    pred, _ = eval_fn(params, es.ctx[i:i + 1], es.qry[i:i + 1])
                    pred = np.asarray(pred[0, 0])
                    img = domains.composite(truth, pred, mask)
                    raw = float((((pred - truth) ** 2) * mask).sum() / mask.sum())
                    lab = f"{raw / es.mse_mean:.2f}"
                    edge = "#3a3936"
                ax = fig.add_axes([x / FIG_W, 1.0 - (y + TILE) / FIG_H,
                                   TILE / FIG_W, TILE / FIG_H])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(edge); sp.set_linewidth(1.0)
                domains.draw(ax, img, domain, mask=mask, hidden_shading=False)
                if lab:
                    txt(x + TILE / 2, y + TILE + 0.015, lab, fontsize=7.4)

    url = save_matplotlib_figure(name, fig, format="png", dpi=150)
    plt.close(fig)
    return url


def bars(name, rows, exps, baseline_row, labels, colours=(BLUE, ORANGE, GREEN),
         ylabel="normalised error (lower is better)", ylim=1.35,
         nothing_label="no better than\nthe average item"):
    """The same six blocks as numbers, with the reference that bounds them.

    `exps` is a list of (legend label, results.jsonl experiment key).
    """
    order_p = ["A_seen_present", "E_same_present", "B_novel_present"]
    order_a = ["C_seen_absent", "F_same_absent", "D_novel_absent"]
    finals = [rows[e]["final"] for _, e in exps]
    bl = rows[baseline_row]["baselines"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    w = 0.78 / len(exps)
    for ax, order, title in [(axes[0], order_p, "answer IS in the context"),
                             (axes[1], order_a, "answer is NOT in the context")]:
        x = np.arange(3)
        for j, ((lab, _), f) in enumerate(zip(exps, finals)):
            off = (j - (len(exps) - 1) / 2) * w
            ax.bar(x + off, [f[c]["nmse"] for c in order], w * 0.92,
                   label=lab, color=colours[j % len(colours)])
        ax.plot(x, [bl[c]["n_ridge"] for c in order], "k^", ms=7,
                label="ridge (ignores the context)")
        ax.axhline(1.0, color="#888", ls="--", lw=1.2)
        # In the margin past the last group, not over the plot. Report 12 put this
        # at x=2.42 inside the axes, which was clear only because its bars all sat
        # below 1.0; a taller ylim drops it straight onto a bar.
        ax.set_xlim(-0.62, 3.05)
        ax.text(2.52, 1.02, nothing_label, fontsize=7.5, color="#666",
                ha="left", va="bottom")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, ylim)
    axes[0].set_ylabel(ylabel, fontsize=9.5)
    axes[0].legend(fontsize=8.4, loc="upper left")
    fig.tight_layout()
    url = save_matplotlib_figure(name, fig, format="svg")
    plt.close(fig)
    return url
