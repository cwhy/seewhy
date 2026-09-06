"""What actually comes back, drawn — not which index it was scored as.

The project's synthetic domain is registered with `kind="board"`, so
`domains.draw` renders its items as chess positions with piece glyphs. That is
right for chess and wrong here: a synthetic item is continuous in [0,1] except in
the 40% of worlds drawn in simplex mode, and glyphs turn a smooth vector into a
confident-looking arrangement of rooks. These panels draw the raw vector instead.

An item is 8 x 8 x 13 — rank, file, channel — and the mask hides files a to d
with all their channels, so reshaping to 8 rows x 104 columns puts the hidden
half in one contiguous block on the left. Grey is the hole.

Two panels, both showing only episodes where identification says FAILURE:

    seen worlds   the returned item is a near-duplicate of the answer. The rows
                  look alike; the metric scored them 0.
    novel worlds  the output matches nothing in the context. This is the real
                  failure, and it is on the transfer axis, not the capacity one.

Episodes are chosen at fixed percentiles of the cost of the pick, so the columns
are a spread of the failures rather than a selection of them.

    .venv/bin/python projects/recall-gen/scripts/recall_images.py exp49
"""
import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))
sys.path.insert(0, str(PROJECT))

from lib import splitfig, domains
from lib.core import predict
from rescore import rows as read_rows, run_from_row
from diag_identification import episode_geometry, quartiles
from shared_lib.media import save_matplotlib_figure

PROJ = "recall-gen"
PCT = (0.10, 0.30, 0.50, 0.70, 0.90)


def as_panel(vec, domain="synth"):
    """One item as a 2-D array to look at.

    Images keep their own shape. A synthetic item is 8 x 8 x 13 — rank, file,
    channel — and is laid out as 8 rows by (file, channel), which puts the
    masked files in one contiguous block. `domains.draw` is not used for the
    synthetic domain: it is registered `kind="board"` and would render a
    continuous vector as chess glyphs.
    """
    d = domains.get(domain)
    w = domains.content_width(domain)
    v = np.asarray(vec).reshape(-1)[:w]
    return v.reshape(d.shape[0], -1) if d.kind == "image" else v.reshape(8, 8 * 13)


def draw(ax, vec, hole=None, domain="synth"):
    d = domains.get(domain)
    v = as_panel(vec, domain)
    if hole is not None:
        v = np.where(as_panel(hole, domain) > 0.5, np.nan, v)
    cm = plt.get_cmap("gray" if d.kind == "image" else "magma").copy()
    cm.set_bad("#9aa0a6")
    ax.imshow(v, cmap=cm, vmin=0.0, vmax=1.0,
              aspect="equal" if d.kind == "image" else "auto",
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#888")


def collect(exp, row, cond, n_eval=512, as_domain=None, M=None):
    rn = run_from_row(row)
    cfg = rn.cfg
    domain = as_domain or rn.domain
    dom = domains.get(domain)
    mask_rows = dom.n_mask_default if as_domain else rn.mask_rows
    if as_domain:
        tr, hd = dom.split
    else:
        tr = tuple(rn.train_digits) if rn.train_digits else None
        hd = tuple(rn.held_digits) if rn.held_digits else None
    M = rn.M if M is None else M
    _, mask, ev = splitfig.build(domain, mask_rows, tr, hd, cfg,
                                 ctx_mode=row.get("ctx_mode", "iid"),
                                 Q=rn.Q, M=M, n_eval=n_eval)
    params = splitfig.load_params(exp)
    es = ev[cond]
    mask_j = jnp.array(mask)
    preds = []
    for i in range(0, es.ctx.shape[0], 128):
        preds.append(np.asarray(predict(params, es.ctx[i:i + 128], es.qry[i:i + 128],
                                        mask_j, cfg)[:, 0, :]))
    pred = np.concatenate(preds)
    ctx, qry = np.asarray(es.ctx), np.asarray(es.qry)[:, 0]
    tgt = np.asarray(es.tgt_idx)[:, 0]
    g = episode_geometry(pred, ctx, qry, tgt, mask)
    m = mask > 0.5
    E = ctx.shape[0]
    ar = np.arange(E)
    cost = ((ctx[ar, g["pick"]][:, m] - qry[:, m]) ** 2).mean(-1) / es.mse_mean
    d_ctx = (((pred[:, None, m] - ctx[:, :, m]) ** 2).mean(-1)).min(-1) / es.mse_mean
    # How far apart the items in THIS episode are, so "is the output sitting on
    # an item" can be asked at the scale of the episode rather than in absolute
    # units that grow with the world's spread.
    spread = (((ctx[:, :, m] - qry[:, None, m]) ** 2).mean(-1)).mean(-1) / es.mse_mean
    # the prediction shown is the true visible half with the predicted hole in it
    shown = domains.composite(qry, pred, mask)
    return dict(mask=mask, ctx=ctx, qry=qry, tgt=tgt, pred=shown, spread=spread,
                domain=domain,
                hit=g["hit"], pick=g["pick"], margin=g["margin"] / es.mse_mean,
                cost=cost, d_ctx=d_ctx, qi=quartiles(g["margin"]))


def panel(fig, gs, D, sel, title, note, annot):
    rows = ["query as given", "true answer", "what the network returned",
            "nearest stored item"]
    n = len(sel)
    wrapped = textwrap.fill(note, 132)
    nlines = wrapped.count("\n") + 1
    inner = gs.subgridspec(len(rows) + 1, n + 1,
                           height_ratios=[0.62 + 0.20 * nlines] + [1] * len(rows),
                           width_ratios=[1.05] + [1] * n, hspace=0.22, wspace=0.06)
    ax = fig.add_subplot(inner[0, :]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.00, title, fontsize=10.5, fontweight="bold", va="top")
    ax.text(0, 0.62, wrapped, fontsize=7.8, color="#444", va="top", linespacing=1.5)
    for r, lab in enumerate(rows):
        a = fig.add_subplot(inner[r + 1, 0]); a.axis("off")
        a.text(1.0, 0.5, lab, fontsize=7.6, ha="right", va="center")
    for c, e in enumerate(sel):
        for r in range(len(rows)):
            a = fig.add_subplot(inner[r + 1, c + 1])
            dm = D["domain"]
            if r == 0:
                draw(a, D["qry"][e], hole=D["mask"], domain=dm)
            elif r == 1:
                draw(a, D["qry"][e], domain=dm)
            elif r == 2:
                draw(a, D["pred"][e], domain=dm)
            else:
                draw(a, D["ctx"][e, D["pick"][e]], domain=dm)
            if r == 0:
                a.set_title(annot(D, e), fontsize=6.8, pad=3.0, linespacing=1.35)


def figure(exp, name=None, verbose=False):
    """Build and upload the two-panel figure. Returns (url, numbers it quotes)."""
    row = {r["experiment"]: r for r in read_rows()}[exp]
    a = argparse.Namespace(exp=exp)

    A = collect(exp, row, "A_seen_present")
    B = collect(exp, row, "B_novel_present")

    def pick_cols(D, fail, q, key):
        s = (D["qi"] == q) & ((~D["hit"]) if fail else D["hit"])
        idx = np.flatnonzero(s)
        v = D["cost"] if key == "cost" else D["d_ctx"] / D["spread"]
        order = idx[np.argsort(v[idx])]
        return [order[min(int(p * len(order)), len(order) - 1)] for p in PCT]

    # Panel A: the metric says FAIL. Closest-rival quartile of seen worlds,
    # only the episodes scored 0, spread over what the mistake costs.
    selA = pick_cols(A, fail=True, q=0, key="cost")
    # Panel B: the metric says PASS. The novel-world quartile where
    # identification is high and the output is furthest from any stored item,
    # spread over that distance measured at the episode's own scale.
    selB = pick_cols(B, fail=False, q=2, key="ratio")
    for nm, D in ((("seen", A), ("novel", B)) if verbose else ()):
        print(f"-- {nm}")
        for q in range(4):
            s0 = D["qi"] == q
            print(f"   q{q}  n={s0.sum():3d}  id={D['hit'][s0].mean():.3f}  "
                  f"margin={D['margin'][s0].mean():.3f}  spread={D['spread'][s0].mean():.3f}  "
                  f"d_ctx={D['d_ctx'][s0].mean():.4f}  "
                  f"d_ctx/spread={(D['d_ctx'][s0]/D['spread'][s0]).mean():.3f}  "
                  f"cost={D['cost'][s0].mean():.4f}")

    fig = plt.figure(figsize=(13.4, 9.6), dpi=200)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, hspace=0.26, top=0.885, bottom=0.04,
                          left=0.055, right=0.985)
    fig.suptitle("Identification is wrong in both directions",
                 fontsize=13.8, fontweight="bold", x=0.055, ha="left", y=0.972)
    fig.text(0.055, 0.936,
             f"{a.exp}. Grey is the hidden half the network had to produce; every "
             "other panel is a whole item. Columns are the 10th to 90th percentile "
             "of the quantity each row is about, so they are a spread of the cases "
             "rather than a selection.",
             fontsize=8.2, color="#444", ha="left")
    ratA = (A["d_ctx"][selA] / A["spread"][selA]).mean()
    ratB = (B["d_ctx"][selB] / B["spread"][selB]).mean()
    panel(fig, gs[0], A, selA,
          "The metric says FAIL. The content is right.",
          f"Worlds seen in training, closest-rival quartile, only episodes scored 0. "
          f"The item returned sits {A['cost'][selA].mean():.3f} from the true answer "
          f"while the rivals are {A['margin'][selA].mean():.3f} apart — a near-duplicate "
          f"came back, and identification scored it zero.",
          lambda D, e: (f"rivals {D['margin'][e]:.3f} apart\n"
                        f"returned item is {D['cost'][e]:.3f} off"))
    panel(fig, gs[1], B, selB,
          "The metric says PASS. The content is not in the context.",
          f"Worlds never seen, only episodes identification got RIGHT "
          f"({B['hit'][B['qi'] == 2].mean():.3f} of that quartile). The output sits "
          f"{ratB:.2f} of the episode's own item spacing from the nearest stored item, "
          f"against {ratA:.2f} above: it is near the right neighbourhood without being "
          f"any item that is actually in memory.",
          lambda D, e: (f"output {D['d_ctx'][e] / D['spread'][e]:.2f} of the\n"
                        f"item spacing from any item"))
    url = save_matplotlib_figure(name or f"{PROJ}_r22_recalled_content_v1", fig)
    plt.close(fig)
    stats = {"ratio_seen": float(ratA), "ratio_novel": float(ratB),
             "cost_seen": float(A["cost"][selA].mean()),
             "margin_seen": float(A["margin"][selA].mean()),
             "id_novel_q2": float(B["hit"][B["qi"] == 2].mean()),
             "by_q": {nm: [dict(q=q, id=float(D["hit"][D["qi"] == q].mean()),
                                margin=float(D["margin"][D["qi"] == q].mean()),
                                spread=float(D["spread"][D["qi"] == q].mean()),
                                ratio=float((D["d_ctx"][D["qi"] == q]
                                             / D["spread"][D["qi"] == q]).mean()),
                                cost=float(D["cost"][D["qi"] == q].mean()))
                           for q in range(4)]
                      for nm, D in (("seen", A), ("novel", B))}}
    return url, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    a = ap.parse_args()
    url, _ = figure(a.exp, verbose=True)
    print("fig:", url)


if __name__ == "__main__":
    main()


def compare_figure(exps, as_domain, name, cond="B_novel_present", n_eval=512):
    """Two architectures answering the same episodes, drawn side by side.

    `exps` is [(row label, experiment)]. Columns are episodes at fixed
    percentiles of how far the FIRST listed model's output sits from the nearest
    stored item, measured at the episode's own scale — so the columns are chosen
    by the baseline's difficulty and not by whether the comparison flatters
    anyone.
    """
    by_exp = {r["experiment"]: r for r in read_rows()}
    D = {e: collect(e, by_exp[e], cond, n_eval=n_eval, as_domain=as_domain)
         for _, e in exps}
    first = D[exps[0][1]]
    ratio = first["d_ctx"] / first["spread"]
    order = np.argsort(ratio)
    sel = [order[min(int(p * len(order)), len(order) - 1)] for p in PCT]

    rows = ["query as given", "true answer"] + \
           [f"returned by {lab}" for lab, _ in exps] + ["nearest stored item"]
    n = len(sel)
    fig = plt.figure(figsize=(12.6, 1.35 * len(rows) + 2.0), dpi=200)
    fig.patch.set_facecolor("white")
    dom_name = domains.get(as_domain).name.replace("synth_to_", "")
    gs = fig.add_gridspec(len(rows), n + 1, left=0.16, right=0.985,
                          top=0.845, bottom=0.03,
                          width_ratios=[0.02] + [1] * n, hspace=0.10, wspace=0.06)
    fig.suptitle(f"What comes back on {dom_name}, a dataset neither network trained on",
                 fontsize=13.2, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.905,
             "Both networks saw only the synthetic prior. Columns are the 10th to "
             f"90th percentile of how far {exps[0][0]}'s output sits from the nearest "
             "stored item, so the episodes are chosen by the baseline. Grey is the "
             "hidden half each network had to produce.",
             fontsize=8.4, color="#444", ha="left", va="top")
    for r, lab in enumerate(rows):
        a = fig.add_subplot(gs[r, 0]); a.axis("off")
        a.text(-0.6, 0.5, lab, fontsize=8.2, ha="right", va="center",
               transform=a.transAxes)
    for c, e in enumerate(sel):
        for r in range(len(rows)):
            a = fig.add_subplot(gs[r, c + 1])
            if r == 0:
                draw(a, first["qry"][e], hole=first["mask"], domain=as_domain)
                a.set_title(f"{ratio[e]:.2f} of the\nitem spacing", fontsize=6.8, pad=3)
            elif r == 1:
                draw(a, first["qry"][e], domain=as_domain)
            elif r <= 1 + len(exps):
                draw(a, D[exps[r - 2][1]]["pred"][e], domain=as_domain)
            else:
                draw(a, first["ctx"][e, first["pick"][e]], domain=as_domain)
    url = save_matplotlib_figure(name, fig)
    plt.close(fig)
    stats = {lab: {"committed": float((D[e]["d_ctx"] / D[e]["spread"]).mean()),
                   "committed_sel": float((D[e]["d_ctx"][sel] / D[e]["spread"][sel]).mean())}
             for lab, e in exps}
    return url, stats


def compare_grid(exps, pools, name, n_col=5, n_eval=512):
    """Every pool and every architecture in one figure: what actually came back.

    `exps` is [(row label, experiment)], `pools` is [(block label, as_domain or
    None, condition)]. Within a block the rows are the query, the true answer,
    one row per architecture, and the item the first architecture's output landed
    nearest — so a row that matches the answer is a recall that worked, whatever
    index the metric assigned it.

    Columns are episodes at fixed percentiles of how far the FIRST listed
    architecture's output sits from the nearest stored item, at the episode's own
    scale. The baseline chooses the episodes; the comparison does not.

    Blocks are sized by the aspect of the items they draw, so a 28 x 28 image and
    a synthetic item laid out 8 x 104 can share one figure without either being
    squashed.
    """
    by_exp = {r["experiment"]: r for r in read_rows()}
    blocks = []
    for entry in pools:
        blab, dom, cond = entry[:3]
        # Two checkpoints trained at different context lengths must be READ at
        # the same length, or "nearest stored item" means something different in
        # each row and the comparison is not one.
        M = entry[3] if len(entry) > 3 else None
        D = {e: collect(e, by_exp[e], cond, n_eval=n_eval, as_domain=dom, M=M)
             for _, e in exps}
        first = D[exps[0][1]]
        ratio = first["d_ctx"] / first["spread"]
        order = np.argsort(ratio)
        sel = [order[min(int(p * len(order)), len(order) - 1)]
               for p in np.linspace(0.10, 0.90, n_col)]
        shp = as_panel(first["qry"][0], first["domain"]).shape
        blocks.append(dict(lab=blab, D=D, first=first, ratio=ratio, sel=sel,
                           domain=first["domain"], aspect=shp[0] / shp[1]))

    rows = ["query as given", "true answer"] + \
           [f"returned by {lab}" for lab, _ in exps] + ["nearest stored item"]
    nrow = len(rows)
    COL_W, LAB_W, TITLE_H = 1.30, 2.05, 0.46
    heights = [TITLE_H + nrow * max(COL_W * b["aspect"], 0.30) for b in blocks]
    fig_w = LAB_W + n_col * COL_W + 0.35
    fig_h = sum(heights) + 1.45
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(len(blocks), 1, height_ratios=heights,
                             top=1 - 1.30 / fig_h, bottom=0.012,
                             left=0.012, right=0.995, hspace=0.30)
    fig.suptitle("What comes back, every pool and both architectures",
                 fontsize=14.5, fontweight="bold", x=0.012, ha="left",
                 y=1 - 0.30 / fig_h)
    fig.text(0.012, 1 - 0.66 / fig_h,
             "Both networks trained on the synthetic prior alone. Columns are the "
             f"10th to 90th percentile of how far {exps[0][0]}'s output sits from the "
             "nearest stored item, so the baseline picks the episodes. Grey is the "
             "hidden half each network had to produce.",
             fontsize=8.6, color="#444", ha="left", va="top")

    for bi, b in enumerate(blocks):
        inner = outer[bi].subgridspec(
            nrow + 1, n_col + 1, height_ratios=[0.42] + [1] * nrow,
            width_ratios=[LAB_W / COL_W] + [1] * n_col, hspace=0.10, wspace=0.05)
        t = fig.add_subplot(inner[0, :]); t.axis("off")
        t.set_xlim(0, 1); t.set_ylim(0, 1)
        cm = np.mean([b["ratio"][b["sel"]]])
        t.text(0, 0.72, b["lab"], fontsize=10.6, fontweight="bold", va="center")
        t.text(0.30, 0.72, f"— {exps[0][0]} sits {cm:.2f} of the item spacing "
                           f"from anything stored, over these columns",
               fontsize=7.8, color="#555", va="center")
        for r, lab in enumerate(rows):
            a = fig.add_subplot(inner[r + 1, 0]); a.axis("off")
            a.text(0.985, 0.5, lab, fontsize=8.0, ha="right", va="center",
                   transform=a.transAxes)
        for c, e in enumerate(b["sel"]):
            for r in range(nrow):
                a = fig.add_subplot(inner[r + 1, c + 1])
                if r == 0:
                    draw(a, b["first"]["qry"][e], hole=b["first"]["mask"],
                         domain=b["domain"])
                elif r == 1:
                    draw(a, b["first"]["qry"][e], domain=b["domain"])
                elif r <= 1 + len(exps):
                    draw(a, b["D"][exps[r - 2][1]]["pred"][e], domain=b["domain"])
                else:
                    draw(a, b["first"]["ctx"][e, b["first"]["pick"][e]],
                         domain=b["domain"])
    url = save_matplotlib_figure(name, fig)
    plt.close(fig)
    return url, {b["lab"]: {lab: float((b["D"][e]["d_ctx"] / b["D"][e]["spread"]).mean())
                            for lab, e in exps} for b in blocks}
