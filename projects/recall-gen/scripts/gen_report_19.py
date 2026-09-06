"""Report 19: the recall-trained network alone, on MNIST, across capacity.

Reports 16 to 18 compare three training objectives. Two of them sit at chance on
every real dataset, which leaves little to say about them and crowds the one arm
that does something. This report drops them and follows the recall arm across the
four sizes it has been trained at, on MNIST, under both kinds of context.

Evaluation and figures only; every number comes from a `_stdeval_` row written by
`scripts/standard_eval.py`.

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_19.py
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

import jax.numpy as jnp

from lib.core import Cfg, masked_mse
from lib import domains, splitfig, synthfig, typstfig
from shared_lib.report import save_report

PROJ = "recall-gen"
TARGET = "mnist"
DOM = "synth_to_mnist"
REPORT_MD_PATH = PROJECT_DIR / "reports" / "19-recall-only-mnist.md"

# Every recall-trained run on this prior, in increasing order of compute.
# `params` is what makes compute a usable shared axis: 192k steps at 4.06M and
# 48k at 14.95M land within 8% of each other, so the last two points are a
# matched-compute comparison of budget against width rather than two rungs of
# one ladder.
RUNS = [("12k steps, d=256", "exp41", "#c2d4ec", 12000, 4.06e6),
        ("48k steps, d=256", "exp45", "#6f9bd1", 48000, 4.06e6),
        ("192k steps, d=256", "exp48", "#1b4a86", 192000, 4.06e6),
        ("48k steps, d=512", "exp49", splitfig.GREEN, 48000, 14.95e6)]
BIG = "exp49"
BANDS = {"A": "A_seen_present", "E": "E_same_present", "B": "B_novel_present"}

rows = {}
for line in open(PROJECT_DIR / "results.jsonl"):
    r = json.loads(line)
    rows[r["experiment"]] = r


def sc(exp, cond, key, ctx=None):
    """Score for one condition, under the context type that condition belongs in.

    The synthetic bands (A/C, E/F) are read under the context type the network
    was TRAINED on. These networks train on single-world episodes, and scoring
    them on iid mixtures of sixteen unrelated worlds asks a question they were
    never trained for — the same checkpoint reads 1.35 there and 0.41 on its own
    task. The real-dataset band (B/D) stays iid, because a real dataset genuinely
    has no world structure to respect.
    """
    row = rows[f"{exp}_stdeval_{DOM}"]
    if ctx is None:
        ctx = row.get("train_ctx_scored", "iid") if cond[0] in "ACEF" else "iid"
    return row["scores"][ctx][cond][key]


def pool(exp, cond, key, ctx=None):
    row = rows[f"{exp}_stdeval_{DOM}"]
    if ctx is None:
        ctx = row.get("train_ctx_scored", "iid") if cond[0] in "ACEF" else "iid"
    return row["pool"][ctx][cond][key]


def bl(key, ctx="iid"):
    name = (f"baselines_{DOM}_M16_r14_split"
            + ("" if ctx == "iid" else "_knn"))
    return rows[name]["baselines"]["D_novel_absent"][key]


def real_denominator():
    """Error of drawing the average real MNIST item — independent of the prior."""
    tr, hd = domains.get(DOM).split
    cfg = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20,
              d_in=domains.PAD_W)
    _, mask, ev = splitfig.build(DOM, 14, tr, hd, cfg, ctx_mode="iid", Q=4,
                                 seed=20260825)
    q = ev["D_novel_absent"].qry
    return float(masked_mse(jnp.broadcast_to(q.reshape(-1, q.shape[-1]).mean(0),
                                             q.shape), q, jnp.array(mask)))


def main():
    den = real_denominator()
    scale = lambda v, ctx="iid": v * bl("mse_mean", ctx) / den

    sweep = [(lab, col,
              {k: sc(e, c, "id_acc") for k, c in BANDS.items()},
              {k: sc(e, c, "nmse") for k, c in BANDS.items()},
              scale(sc(e, "D_novel_absent", "nmse")))
             for lab, e, col, _, _ in RUNS]
    refs = {"ridge": scale(bl("n_ridge")), "soft": scale(bl("n_knn"))}

    per_ctx = {c: (sc(BIG, "B_novel_present", "id_acc", c),
                   scale(sc(BIG, "D_novel_absent", "nmse", c), c))
               for c in ("iid", "knn")}
    refs_ctx = {c: scale(bl("n_knn", c), c) for c in ("iid", "knn")}
    margins = {c: pool(BIG, "B_novel_present", "margin_med", c)
               for c in ("iid", "knn")}

    url_merged = typstfig.capacity_two_panel(
        f"{PROJ}_r19_capacity_v7",
        [(lab, st, pr, r[2], r[4]) for (lab, _, _, st, pr), r in zip(RUNS, sweep)],
        refs)
    url_ctx = typstfig.context_chart(f"{PROJ}_r19_context_v3", per_ctx, margins)
    nets = [(lab, splitfig.load_params(e)) for lab, e, _, _, _ in
            (RUNS[1], RUNS[3])]
    tr, hd = domains.get(DOM).split
    GRID_LEGEND = [("recall-trained, d_model=512: during training its answer was "
                    "ALWAYS one of the sixteen context items.", splitfig.BLUE)]
    spec = splitfig.grid_spec(
        DOM, 14, tr, hd,
        Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=domains.PAD_W),
        [nets[1]],
        ["the prior\nworlds seen in training",
         "the prior\nworlds never seen",
         "real MNIST\nnever seen at all"],
        GRID_LEGEND, ctx_mode="iid", Q=4)
    url_grid = typstfig.tile_grid(
        f"{PROJ}_r19_grid_v5", spec,
        headline="The largest recall network. Trained only on the prior, never shown MNIST.",
        sub=("The prior's own items are shown as 28x28 because that is the shape these 832 coordinates are read in.\n"
             "Red is the score over all 512 episodes, against the average training item; numbers under tiles are single episodes."),
        legend=GRID_LEGEND)
    for u in (url_merged, url_ctx, url_grid):
        print("fig:", u)

    small, big = sweep[0], sweep[-1]

    md = f"""# On MNIST, capacity buys retrieval and not completion

A network was trained on a synthetic prior — random low-dimensional worlds, a
fresh one every episode — and never shown an MNIST digit. It was then asked to
do two things with real MNIST: find a digit sitting in its context, and predict
one that is not there.

Trained at four sizes, those two abilities go in opposite directions.

| | finding a digit | predicting a digit |
|---|---|---|
| 12k steps, d=256 | {small[2]['B']:.3f} | {small[4]:.3f} |
| 48k steps, d=256 | {sweep[1][2]['B']:.3f} | {sweep[1][4]:.3f} |
| 192k steps, d=256 | {sweep[2][2]['B']:.3f} | {sweep[2][4]:.3f} |
| 48k steps, d=512 | **{big[2]['B']:.3f}** | **{big[4]:.3f}** |
| chance / do-nothing | 0.063 | 1.000 |

Finding nearly triples. Predicting gets slightly worse, and ends above the score
for ignoring the input entirely.

{synthfig.terms(TARGET, den)}

Only the recall-trained network appears below. The completion-trained and
frozen networks are covered in report 17; both sit at chance on MNIST at every
size tried, which is a short story and not this one.

## What scales

![Capacity against the two abilities]({url_merged})

The left panel splits finding by novelty band. All three climb together and none
has flattened.

On worlds the network trained on, identification goes from
{small[2]['A']:.3f} to {big[2]['A']:.3f}. On worlds from the same prior that it
has never seen, {small[2]['E']:.3f} to {big[2]['E']:.3f}. On real MNIST,
{small[2]['B']:.3f} to {big[2]['B']:.3f}.

Two things are worth reading off that. The first is that the middle band is the
one that says whether the prior works at all, and at the smallest size it does
not — {small[2]['E']:.3f} against chance of 0.063 is barely a signal. The
capability appears with capacity, and an earlier version of this work concluded
the prior had failed on the strength of the small model alone.

The second is the widening gap between the first two bands. At the smallest size
the network is {small[2]['A']-small[2]['E']:.3f} better on worlds it trained on
than on fresh ones; at the largest, {big[2]['A']-big[2]['E']:.3f}. Bigger models
memorise more of the training worlds — and still generalise better in absolute
terms.

## What does not

The right panel is prediction: {small[4]:.3f}, {sweep[1][4]:.3f},
{sweep[2][4]:.3f}, {big[4]:.3f}, where 1.0 is the error of drawing the average
real digit. Higher is worse, so it drifts the wrong way. Sixteen times the
compute, spent two different ways, does not move it.

The x axis is compute — steps times parameters — which makes the last two points
a controlled comparison rather than two rungs of one ladder. 192 000 steps at
4.06M parameters and 48 000 at 14.95M land within 8% of the same compute, and the
wider model is higher on every finding series.

Both references sit where they sat. Ridge, a linear map fitted on the prior that
never looks at the context, scores {refs['ridge']:.3f}. The soft look-up, which
does nothing but average the sixteen context images by similarity and involves no
training at all, scores {refs['soft']:.3f} — better than every network on this
list.

So the context contains something usable for prediction, and more capacity does
not make the network any better at using it. Whatever retrieval is learning from
scale, prediction is not learning it too.

![What the largest network produces]({url_grid})

## The context matters more than the network

Everything above uses sixteen unrelated images as context. Replace them with the
query's own sixteen nearest neighbours and both numbers move — in opposite
directions again.

![The same network, two kinds of context]({url_ctx})

Identification falls from {per_ctx['iid'][0]:.3f} to {per_ctx['knn'][0]:.3f}.
Prediction improves from {per_ctx['iid'][1]:.3f} to {per_ctx['knn'][1]:.3f}.

This is not a paradox, and the pool diagnostics say why. A neighbour context is
assembled *from* the query's closest matches, so it is by construction a
low-margin context: the median distance from a target to its nearest rival falls
from {margins['iid']:.2f} to {margins['knn']:.2f}. The answer becomes easier to
reconstruct and harder to name, because the sixteen candidates are now nearly the
same picture.

That is the same mechanism report 15 measured on real training data, reproduced
here on a network that has only ever seen synthetic worlds.

## What this establishes

That retrieval from a synthetic prior transfers to MNIST, partially, and scales
with capacity across every size tried — {small[2]['B']:.3f} to
{big[2]['B']:.3f}, with no sign of a ceiling.

That prediction does not transfer, does not scale, and is beaten by a
similarity-weighted average of the context that requires no training whatsoever.

And that the two abilities respond to capacity and to context in opposite
directions, which is the same split this project has been measuring since report
1, arriving here through a knob that is not the training objective.

## What this does not establish

Where the retrieval ceiling is. The largest model tried is the best on every
band, and the step sweep was still climbing when it stopped; d=1024 is the
obvious next point and has not been run.

Why prediction is immovable. It survived four times the steps and four times the
parameters unchanged, which is evidence against a pure compute explanation, but
sixteen examples of an 800-dimensional world is very little to infer a world
from, and that is a property of the task design rather than of the network.

## Sources

`results.jsonl` rows `exp41`, `exp45`, `exp48` and `exp49` — the recall-trained
runs on the synthetic prior at 12k/48k/192k steps and d_model 256/512 — with
their `_stdeval_{DOM}` rows from `scripts/standard_eval.py`, which carry both
context types, the identification ceilings and the nearest-rival margins.
`baselines_{DOM}_M16_r14_split` and its `_knn` counterpart hold the ridge, soft
look-up and average-item references. Completion is divided by the error of the
average real MNIST item ({den:.4f}) rather than the average training item,
because the training pool here is synthetic and would not be a meaningful
reference for a real digit. Figures generated by `scripts/gen_report_19.py`. No
training was run for this report.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_19", md))


if __name__ == "__main__":
    main()
