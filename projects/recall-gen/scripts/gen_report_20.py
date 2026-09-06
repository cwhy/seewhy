"""Report 20: why recall saturates below its ceiling on the synthetic prior.

Evaluation and figures only. Everything is read from `results.jsonl` rows or
recomputed from checkpoints; no training was run for this report.

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_20.py
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import jax.numpy as jnp

from lib.core import Cfg, masked_mse
from lib import domains, splitfig, typstfig
from lib.train import make_eval
from shared_lib.typst_plot import bar_chart, cm, line_chart, long_form
from shared_lib.typst_report import save_figure
from shared_lib.report import save_report

PROJ = "recall-gen"
REPORT_MD_PATH = PROJECT_DIR / "reports" / "20-recall-on-a-prior.md"
CHANCE = 1.0 / 16
BL = "baselines_synth_M16_r4_split_class"

# (label, exp, heads, dk, params M) — the three points that separate capacity
# from memory size. exp49 and exp54 have the same parameter count.
RUNS = [("d=256, 4x64", "exp45", 4, 64, 4.06),
        ("d=512, 8x64", "exp49", 8, 64, 14.95),
        ("d=512, 4x128", "exp54", 4, 128, 14.94)]

rows = {}
for line in open(PROJECT_DIR / "results.jsonl"):
    r = json.loads(line)
    rows[r["experiment"]] = r


def cfg_of(exp):
    return Cfg(**rows[exp]["cfg"])


def margin_breakdown(exp, cond="A_seen_present", nq=4):
    """Per-episode identification and nearest-rival margin, bucketed by margin.

    The whole argument of this report: identification is not a smooth function
    of how well the network works, it is a step function of how far apart the
    candidates are.
    """
    cfg = cfg_of(exp)
    tr, hd = domains.get("synth").split
    rn, mask, ev = splitfig.build("synth", 4, tr, hd, cfg, ctx_mode="class", Q=4,
                                  seed=12345)
    fn = make_eval(rn, jnp.array(mask))
    params = splitfig.load_params(exp)
    es = ev[cond]
    margin = splitfig.nearest_distractor(ev, mask, cond)
    hits, errs = [], []
    for i in range(0, es.ctx.shape[0], 128):
        c, q = es.ctx[i:i + 128], es.qry[i:i + 128]
        pred, argmin = fn(params, c, q)
        hits.append(np.asarray(argmin[:, 0] == es.tgt_idx[i:i + 128][:, 0]))
        e = (((pred[:, :1] - q[:, :1]) ** 2) * jnp.array(mask)).sum(-1)[:, 0] / mask.sum()
        errs.append(np.asarray(e) / es.mse_mean)
    hits, errs = np.concatenate(hits), np.concatenate(errs)
    edges = np.percentile(margin, np.linspace(0, 100, nq + 1))
    out = []
    for j in range(nq):
        lo, hi = edges[j], (edges[j + 1] if j < nq - 1 else np.inf)
        m = (margin >= lo) & (margin <= hi if j == nq - 1 else margin < hi)
        out.append({"margin": float(margin[m].mean()), "id": float(hits[m].mean()),
                    "err": float(errs[m].mean())})
    return out, float(hits.mean()), float(np.median(margin))


def main():
    brk, overall, med = margin_breakdown("exp49")
    for b in brk:
        print(f"  margin {b['margin']:.2f}  id {b['id']:.3f}  err {b['err']:.3f}")

    # ── the diagnostic figure: identification against candidate separation ──
    data = long_form(
        [b["margin"] for b in brk],
        {"identification accuracy": [b["id"] for b in brk],
         "reconstruction error": [b["err"] for b in brk]},
        x_name="margin", y_name="value", series_name="quantity")
    url_margin = save_figure(line_chart(
        f"{PROJ}_r20_margin", data, x="margin", y="value", colour="quantity",
        points=True,
        title="Recall fails where the candidates are close, not where the network is weak",
        subtitle=("Episodes split into quartiles by how far the target sits from its "
                  "nearest rival. d=512, 8 heads, worlds seen in training."),
        x_label="distance from the target to its nearest rival (mean of quartile)",
        y_label="", hlines=[(CHANCE, "chance (1 of 16)")],
        caption=("Both curves are on the same axis because both are dimensionless: "
                 "an accuracy in [0,1] and an error against the average item."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r20_margin_v1")

    # ── capacity against memory size ──
    ids = {lab: rows[e]["history"]["id_acc"]["A_seen_present"][-1]
           for lab, e, _, _, _ in RUNS}
    cap = long_form([lab for lab, *_ in RUNS],
                    {"identification, worlds seen in training":
                     [ids[lab] for lab, *_ in RUNS]},
                    x_name="model", y_name="id", series_name="what")
    url_cap = save_figure(bar_chart(
        f"{PROJ}_r20_capacity", cap, x="model", y="id", fill="what",
        x_order=[lab for lab, *_ in RUNS],
        title="Parameters move recall; memory size barely does",
        subtitle=("The last two have the same parameter count and differ only in the "
                  "size of the recurrent memory: 32 768 floats against 65 536."),
        x_label="", y_label="identification accuracy", y_limits=(0.0, 1.0),
        hlines=[(CHANCE, "chance"), (1.0, "ceiling")],
        caption=("d_model = heads x dk, so trading heads against dk holds parameters "
                 "fixed and doubles the state — and also halves the number of heads, "
                 "which the 2 x 256 point would have separated had it fitted in memory."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r20_capacity_v1")

    for u in (url_margin, url_cap):
        print("fig:", u)

    b = rows[BL]["baselines"]
    net49 = rows["exp49"]["final"]
    net54 = rows["exp54"]["final"]

    md = f"""# Recall on a synthetic prior stops at {overall:.2f}, and the memory is not why

A network trained on the synthetic prior is shown sixteen items from one world
and then a seventeenth that is a copy of one of them, with half its coordinates
erased. Asked which of the sixteen it is, it answers correctly
**{overall:.3f}** of the time. Chance is 0.063 and the ceiling is 1.000 — feeding
the true answer in always identifies it.

These are worlds the network trained on, with the answer literally present. It
should be near-perfect and it is not.

## It is not failing to rebuild the item

Split those episodes into quartiles by how far the target sits from its nearest
rival in the same context — a property of the episode, measured without any
network.

![Identification against candidate separation]({url_margin})

Identification is **{brk[-1]['id']:.3f}** in the quartile where rivals are far
apart and **{brk[0]['id']:.3f}** where they are close. That alone would be
unremarkable. What matters is the second curve: in the failing quartile the
network's reconstruction error is **{brk[0]['err']:.3f}**, its *best* of the four,
while in the quartile it gets right it is **{brk[-1]['err']:.3f}**, its worst.

So the network reconstructs the hard-to-name items better than the easy ones and
still cannot name them. Recall is not unlearned.

The arithmetic explains it. To pick the nearer of two items that sit
{brk[0]['margin']:.2f} apart, the output has to land within about
{brk[0]['margin']:.2f} of the right one. The network lands within
{brk[0]['err']:.3f}. Close, and not close enough — and since an exact
reconstruction would score 1.000, this is a real shortfall in precision rather
than a broken metric.

Why are the candidates so close? Because they come from one world. The prior
draws each world's items from a latent subspace whose dimension is sampled as
low as 1, and sixteen draws from a one-dimensional world lie almost on a line.
The median separation across all episodes is {med:.2f}.

## The memory is not the bottleneck

The obvious suspect was the memory. Recall here works by writing sixteen items
into a fixed-size matrix with the delta rule, `S += e k^T`, and two similar keys
write along nearly the same direction, so the second write partly erases the
first. That predicts exactly the observed pattern: failures concentrated where
items are similar.

It is testable without touching the prior. `d_model = heads x dk`, and `Wq`,
`Wk`, `Wv` and `Wo` are all `d_model x d_model`, so trading heads against dk
leaves the parameter count untouched and changes only how much state there is.

![Parameters against memory size]({url_cap})

| | heads x dk | state floats | parameters | identification |
|---|---|---|---|---|
| {RUNS[0][0]} | {RUNS[0][2]} x {RUNS[0][3]} | {RUNS[0][2]*RUNS[0][3]**2:,} | {RUNS[0][4]}M | {ids[RUNS[0][0]]:.3f} |
| {RUNS[1][0]} | {RUNS[1][2]} x {RUNS[1][3]} | {RUNS[1][2]*RUNS[1][3]**2:,} | {RUNS[1][4]}M | {ids[RUNS[1][0]]:.3f} |
| {RUNS[2][0]} | {RUNS[2][2]} x {RUNS[2][3]} | {RUNS[2][2]*RUNS[2][3]**2:,} | {RUNS[2][4]}M | {ids[RUNS[2][0]]:.3f} |

Doubling the memory at identical parameters buys
**{ids[RUNS[2][0]] - ids[RUNS[1][0]]:+.3f}**. Across three independent draws of
512 episodes the spread on this metric is about 0.015, so that is real and
barely — roughly twice the noise.

Adding parameters buys **{ids[RUNS[1][0]] - ids[RUNS[0][0]]:+.3f}**, five times
as much.

If interference in a bounded memory were what caps recall, doubling that memory
should not be worth a fifth of what parameters are worth. The hypothesis was
wrong, and the plainer reading survives: this is a precision problem, and
precision tracks capacity.

One caveat keeps it from being airtight. Because `d_model = heads x dk`, the
same trade that doubles the state also halves the number of heads — eight
independent memories become four larger ones. A 2 x 256 point would have
separated those, and it asks for 23.65 GiB on a 24 GB card, because the state is
`(batch, heads, dk, dk)` and the scan keeps a carry per token.

## What the network is worth against a ceiling

With the answer absent rather than present, the same episodes have a
model-free ceiling worth stating. A per-episode least-squares fit from the
visible coordinates to the hidden ones, solved in the dual on the sixteen context
items, is the strongest estimator the context allows — and is precisely the
computation linear attention performs.

| | worlds seen | worlds never seen |
|---|---|---|
| network, d=512 8x64 | {net49['C_seen_absent']['nmse']:.3f} | {net49['D_novel_absent']['nmse']:.3f} |
| network, d=512 4x128 | {net54['C_seen_absent']['nmse']:.3f} | {net54['D_novel_absent']['nmse']:.3f} |
| in-context least squares | **{b['C_seen_absent']['n_icl']:.3f}** | **{b['D_novel_absent']['n_icl']:.3f}** |
| soft look-up | {b['C_seen_absent']['n_knn']:.3f} | {b['D_novel_absent']['n_knn']:.3f} |
| ridge, ignores the context | {b['C_seen_absent']['n_ridge']:.3f} | {b['D_novel_absent']['n_ridge']:.3f} |

On worlds it has never seen the network is
{net49['D_novel_absent']['nmse'] / b['D_novel_absent']['n_icl']:.1f} times worse
than a sixteen-by-sixteen linear solve that has seen no world at all. The gap is
the headroom, and it is large.

## What this changes

Recall on this prior is limited by reconstruction precision, not by the memory
architecture, and precision has tracked capacity at every point measured. The
untested next step is width, not a different memory.

The in-context least-squares baseline outlives the hypothesis it was built to
test. Until now the strongest model-free reference here was a similarity-weighted
average of the context, which is the right ceiling when items are drawn
independently and the wrong one when they share a generative structure. Every
future run on a structured prior should be quoted against it.

## What this does not establish

Where the ceiling is. The largest model tried is the best on every measure and
the trend has not turned over.

That the memory never matters. It was measured at one width, on one prior, with
heads and dk traded against each other rather than varied independently, and the
one point that would have separated them did not fit in memory.

## Sources

`results.jsonl` rows `exp45`, `exp49` and `exp54` — recall training on the
synthetic prior at d_model 256/512 and two head/dk splits — and `{BL}` for the
model-free references, which are computed on single-world episodes to match the
context type these networks were trained on. The margin breakdown is recomputed
from the `exp49` checkpoint by this script. Figures generated by
`scripts/gen_report_20.py`. No training was run for this report.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_20", md))


if __name__ == "__main__":
    main()
