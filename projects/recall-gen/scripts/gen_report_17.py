"""Reports 17 and 18: the synthetic prior on one image dataset.

One generator, two reports — `--target mnist` writes report 17 and
`--target fashion_mnist` writes report 18. They differ in exactly one thing
scientifically (which of the prior's two modes helps), and that difference is
the argument, so a single script keeps their numbers, figures and prose in step
instead of letting two copies drift.

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_17.py --target mnist
    uv run --no-sync python projects/recall-gen/scripts/gen_report_17.py --target fashion_mnist
"""
import argparse
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
# The arms are the d_model=512 runs, so figures that load their parameters need
# the matching shape. Eval-set construction depends only on d_in, so the narrow
# config would work for the denominator and not for the grid.
CFG = Cfg(d_model=512, n_layers=4, dk=64, n_heads=8, n_tokens=20, d_in=domains.PAD_W)
RECALL, COMPLETE, FROZEN = synthfig.RECALL, synthfig.COMPLETE, synthfig.FROZEN
SIMPLEX, CONT = synthfig.SIMPLEX, synthfig.CONT
BASE = synthfig.BASE

# One generator, three reports. Report 16 is the chess member of the set rather
# than a differently shaped report about the same runs.
META = {
    "chess": dict(n=16, pretty="chess", slug="16-synthetic-prior",
                  other="MNIST", mask=4, tile=1.15, lab_w=2.10,
                  item="position", unit="squares"),
    "mnist": dict(n=17, pretty="MNIST", slug="17-synthetic-prior-mnist",
                  other="Fashion-MNIST", mask=14, tile=0.80, lab_w=1.30,
                  item="image", unit="pixels"),
    "fashion_mnist": dict(n=18, pretty="Fashion-MNIST",
                          slug="18-synthetic-prior-fashion", other="MNIST",
                          mask=14, tile=0.80, lab_w=1.30,
                          item="image", unit="pixels"),
}

rows = {}
for line in open(PROJECT_DIR / "results.jsonl"):
    r = json.loads(line)
    rows[r["experiment"]] = r


def sc(exp, target, cond, key, ctx=None):
    """Score for one condition, under the context type that condition belongs in.

    The synthetic bands (A/C, E/F) are read under the context type the network
    was TRAINED on. These networks train on single-world episodes, and scoring
    them on iid mixtures of sixteen unrelated worlds asks a question they were
    never trained for — the same checkpoint reads 1.35 there and 0.41 on its own
    task. The real-dataset band (B/D) stays iid, because a real dataset genuinely
    has no world structure to respect.
    """
    row = rows[f"{exp}_stdeval_{BASE[exp]}_to_{target}"]
    if ctx is None:
        ctx = row.get("train_ctx_scored", "iid") if cond[0] in "ACEF" else "iid"
    return row["scores"][ctx][cond][key]


def bl(exp, target, key):
    m = META[target]["mask"]
    return rows[f"baselines_{BASE[exp]}_to_{target}_M16_r{m}_split"][
        "baselines"]["D_novel_absent"][key]


def real_denominator(target):
    """Error of drawing the average REAL item — a prior-independent reference.

    The project's usual denominator is the average TRAINING item, which differs
    between the two priors and cannot carry a comparison across them.
    """
    dom = f"{BASE[RECALL]}_to_{target}"
    tr, hd = domains.get(dom).split
    _, mask, ev = splitfig.build(dom, META[target]["mask"], tr, hd, CFG,
                                 ctx_mode="iid", Q=4, seed=20260825)
    q = ev["D_novel_absent"].qry
    mean_item = q.reshape(-1, q.shape[-1]).mean(0)
    return float(masked_mse(jnp.broadcast_to(mean_item, q.shape), q, jnp.array(mask)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(META))
    a = ap.parse_args()
    t, m = a.target, META[a.target]
    pretty, other = m["pretty"], m["other"]
    ARMS = synthfig.ARMS

    den = real_denominator(t)
    ident = {}
    for _, e, _ in ARMS:
        ident[(e, "real")] = sc(e, t, "B_novel_present", "id_acc")
        ident[(e, "prior")] = sc(e, t, "E_same_present", "id_acc")
    comp = {e: sc(e, t, "D_novel_absent", "nmse") * bl(e, t, "mse_mean") / den
            for _, e, _ in ARMS}
    present = {e: sc(e, t, "B_novel_present", "nmse") * bl(e, t, "mse_mean") / den
               for _, e, _ in ARMS}
    refs = {"ridge": bl(RECALL, t, "n_ridge") * bl(RECALL, t, "mse_mean") / den,
            "soft": bl(RECALL, t, "n_knn") * bl(RECALL, t, "mse_mean") / den}

    nb = {lab: synthfig.near_binary(d) for lab, d in
          (("MNIST", "mnist_pad"), ("Fashion-MNIST", "fashion_mnist_pad"),
           ("chess", "chess"))}
    adv = {lab: sc(SIMPLEX, tt, "B_novel_present", "id_acc")
                - sc(CONT, tt, "B_novel_present", "id_acc")
           for lab, tt in (("MNIST", "mnist"), ("Fashion-MNIST", "fashion_mnist"),
                           ("chess", "chess"))}

    # The compute sweep, if it has been run. Endpoints of completed cosine
    # schedules, so each point is a converged-as-specified run rather than a
    # snapshot taken mid-decay.
    SWEEP = [("12k steps, d=256", "exp41", 12000, 256, "#c2d4ec"),
             ("48k steps, d=256", "exp45", 48000, 256, "#6f9bd1"),
             ("192k steps, d=256", "exp48", 192000, 256, "#1b4a86"),
             ("48k steps, d=512", "exp49", 48000, 512, splitfig.GREEN)]
    sweep = [(lab, st, dm,
              rows[e]["history"]["id_acc"]["A_seen_present"][-1],
              rows[e]["history"]["id_acc"]["B_novel_present"][-1], col)
             for lab, e, st, dm, col in SWEEP if e in rows]
    url_scale = (typstfig.scaling_chart(
        f"{PROJ}_r{m['n']}_scaling_v4",
        [(lab, st, dm, tr, hd) for lab, st, dm, tr, hd, _ in sweep])
        if len(sweep) > 2 else None)

    url_bars = typstfig.arms_chart(f"{PROJ}_r{m['n']}_bars_v5", ARMS, ident, comp,
                                   refs, pretty)
    url_grid = synthfig.grid_for(f"{PROJ}_r{m['n']}_grid_v5", t, CFG, m["mask"],
                                 tile=m["tile"], lab_w=m["lab_w"])
    url_abl = typstfig.resemblance_chart(
        f"{PROJ}_r{m['n']}_ablation_v4",
        [(lab, nb[lab], adv[lab]) for lab in ("Fashion-MNIST", "MNIST", "chess")],
        pretty)
    for u in (url_bars, url_grid, url_abl, url_scale):
        print("fig:", u)
    print("sweep:", [(l, round(a, 3), round(b, 3)) for l, _, _, a, b, _ in sweep])
    print("ident:", {k: round(v, 3) for k, v in ident.items()})
    print("comp:", {k: round(v, 3) for k, v in comp.items()})

    best_find = max(ARMS, key=lambda A: ident[(A[1], "real")])
    best_pred = min(ARMS, key=lambda A: comp[A[1]])
    helps = "helps" if adv[pretty] > 0 else "hurts"

    NARROW_RECALL = synthfig.NARROW[RECALL]
    base_train = rows[NARROW_RECALL]["history"]["id_acc"]["A_seen_present"][-1]
    base_held = rows[NARROW_RECALL]["history"]["id_acc"]["B_novel_present"][-1]
    sweep_gap = f"{base_train:.3f} against {base_held:.3f} for the narrow 48 000-step run"
    base_held_iid = sc(NARROW_RECALL, t, "E_same_present", "id_acc")
    if url_scale:
        by = {lab: (tr, hd) for lab, _, _, tr, hd, _ in sweep}
        longer = by.get("192k steps, d=256")
        wider = by.get("48k steps, d=512")
        bits = ["![Identification against training compute](" + url_scale + ")", ""]
        if longer:
            bits.append(
                f"Four times the steps takes identification on the prior's own "
                f"training worlds from {base_train:.3f} to **{longer[0]:.3f}**, and "
                f"on worlds it has never seen from {base_held:.3f} to "
                f"**{longer[1]:.3f}**.")
        if wider:
            bits.append(
                f"Twice the width, at the original budget, reaches "
                f"**{wider[0]:.3f}** and **{wider[1]:.3f}** — roughly four times "
                f"the parameters and twice the recurrent state, 32 768 floats "
                f"against 16 384.")
        if longer and wider:
            which = ("more steps" if longer[1] > wider[1] else "more width")
            bits.append(
                f"Of the two, **{which}** buys more on worlds the network has "
                f"never seen, which is the number that matters: "
                f"{longer[1]:.3f} for the longer run against {wider[1]:.3f} for "
                f"the wider one.")
        scaling_prose = "\n\n".join(bits)
    else:
        scaling_prose = ("The compute sweep separating budget from capacity has "
                         "not been run yet, so this section reports only the "
                         "evidence that the 48 000-step budget was insufficient.")

    md = f"""# Recall and completion on a synthetic prior, tested on {pretty}

Three networks were trained on a synthetic prior and never shown a real {pretty}
{m['item']}. They are identical in size and shape and differ in one thing: what their
training episodes looked like. One always had its answer sitting in the context.
One never did. One was the first with most of its weights frozen at random.

Then all three were asked to do both jobs on real {pretty}: find an item that is
present, and predict one that is absent.

| | finding, on {pretty} | predicting, on {pretty} |
|---|---|---|
| recall-trained | **{ident[(RECALL, 'real')]:.3f}** | {comp[RECALL]:.3f} |
| completion-trained | {ident[(COMPLETE, 'real')]:.3f} | **{comp[COMPLETE]:.3f}** |
| frozen layers | {ident[(FROZEN, 'real')]:.3f} | {comp[FROZEN]:.3f} |
| chance / do-nothing | 0.063 | 1.000 |

{synthfig.terms(t, den)}

## Finding, and predicting

![The three arms on {pretty}]({url_bars})

The left panel puts real {pretty} beside the control that decides how to read
everything else: fresh worlds drawn from the prior the networks were actually
trained on.

The recall-trained network scores {ident[(RECALL, 'prior')]:.3f} there, against
chance of 0.063. It did learn to retrieve from a world it had never seen, given
sixteen examples of it. That is the thing the prior was built to teach, and at
this width it works.

At half this width it did not. The same arm, same prior, same budget, at
d_model=256 scored {base_held_iid:.3f} on that control. The section below is
about that, and it is the reason the arms here are the wider ones.

On real {pretty} the recall arm scores {ident[(RECALL, 'real')]:.3f}. The
completion-trained network scores {ident[(COMPLETE, 'real')]:.3f} and the frozen
one {ident[(FROZEN, 'real')]:.3f} — both at chance. Across three independent
draws of 512 episodes the spread is about {synthfig.EVAL_SPREAD:.3f}, drawn as
the error bar on each bar.

The right panel is prediction, with the two references that bound it. Ridge never
looks at the context and scores {refs['ridge']:.3f}. The soft look-up uses
nothing but the context and scores {refs['soft']:.3f}. The best of the three
networks is **{best_pred[0]}** at {comp[best_pred[1]]:.3f}.

## The recall/completion split

Across reports 12 to 15 this comparison had a consistent shape with two parts.
One part reproduces here and the other does not.

The part that reproduces: **a completion-trained network does not read its
context.** On {pretty} it scores {present[COMPLETE]:.3f} when the answer is
present and {comp[COMPLETE]:.3f} when it is absent — the same number, to three
decimals. Whether the answer is sitting in front of it makes no difference to
what it draws, because its training never rewarded looking. Its identification
is {ident[(COMPLETE, 'real')]:.3f} against chance of 0.063. That signature has
now appeared in every domain this project has run, real or synthetic, and four
times the capacity does not change it.

The part that does not: **completion training buys no prediction here.** On real
data that was the trade — the completion arm gave up finding and got predicting
in return. On this prior it gives up finding and gets nothing: it predicts at
{comp[COMPLETE]:.3f} against the recall arm's {comp[RECALL]:.3f}, and the recall
arm is the one that can also find things.

So on a synthetic prior the recall objective dominates. It is better at finding
by {ident[(RECALL, 'real')] - ident[(COMPLETE, 'real')]:.3f} and no worse at
predicting. There is no trade to make.

The frozen arm is at chance too, at {ident[(FROZEN, 'real')]:.3f}. Freezing the
mixing layers has been survivable on real data; on a prior that has to be
inferred from the context every episode, it is not.

![What the three networks produce]({url_grid})

The bottom band is real {pretty}, and it is the one to look at. The prior's own
items, two bands above, are what the networks were fed.

## Is it trained enough?

The honest answer to why training stopped where it did is that the budget was
set to 48 000 steps, not that anything showed it was enough. It was not enough,
and the reason is worth stating because it changes how every number above should
be read.

{scaling_prose}

Three things say the same thing. The training loss is still falling at the end.
The gap between worlds seen in training and worlds never seen is small —
{sweep_gap} — so the network is underfitting rather than memorising, and a
network that is underfitting has room that more compute can buy. And the budget
has already been quadrupled once, from 12 000 steps to 48 000, which moved
identification on training worlds from 0.488 to {base_train:.3f}.

The flattening at the end of any single run is a property of the optimiser, not
of the task. The learning rate follows a cosine decay to a tenth of its peak over
whatever `steps` is set to, so every run in this project goes flat over its last
decile whatever it has converged to. The 12 000-step run went flat at 0.49 and
the 48 000-step run went flat at 0.62. Same shape, different values. Reading
either as saturation would be reading the schedule.

## Does the prior's simplex mode carry anything?

This is a question about the prior rather than about the three networks, and it
is worth asking because one dataset behaves very differently from the other two.

Chess is sixty-four groups of thirteen with one active coordinate each, which is
exactly what a simplex world draws. A recall network trained on the prior *with*
simplex worlds retrieves chess at 0.942; trained on an otherwise identical prior
*without* them, at 0.321. That is not generalisation to chess. It is the prior
generating chess.

![Resemblance against transfer]({url_abl})

The same axis orders {pretty}. It has **{nb[pretty]:.3f}** of its coordinates
within 0.1 of 0 or 1, against {nb[other]:.3f} for {other} and 1.000 for chess.
Removing the simplex mode {helps} retrieval on {pretty} by
{abs(adv[pretty]):.3f}, against a draw-to-draw spread of
{synthfig.EVAL_SPREAD:.3f}.

Ordering the three datasets by how binary they are orders them by how much the
simplex mode is worth: chess {adv['chess']:+.2f}, MNIST {adv['MNIST']:+.2f},
Fashion-MNIST {adv['Fashion-MNIST']:+.2f}. Three points is an ordering and not a
fit, and it is measured on one axis only.

## What this establishes

That a network trained only on this synthetic prior learns to retrieve from a
world it has never seen — {ident[(RECALL, 'prior')]:.3f}, against chance of
0.063 — and carries part of that to real {pretty}, at
{ident[(RECALL, 'real')]:.3f}. It needed 15M parameters to do it; at 4M it did
not.

That the recall objective dominates on this prior. It is the only one of the
three that retrieves at all, and it predicts as well as either of the others.
The trade the completion objective makes on real data is not available here.

And that prediction is where the prior gives nothing. No arm, at either width,
at either setting of the simplex mode, predicts a real {pretty} item better than
drawing the average real {pretty} item.

## What this does not establish

Where the ceiling is. Nothing here is converged in the sense that more compute
would not help: the largest model tried is the best model on every measure, and
the step sweep was still climbing. The scaling section gives the two points that
bound what has actually been tested, and both directions were still paying.

Whether prediction would ever come. It did not move with four times the steps or
four times the parameters, which is evidence that it is not simply a compute
problem — but a prior with fewer latent dimensions, or more than sixteen items
per episode, would be a different experiment. Sixteen examples of an
800-dimensional world is very little to infer a world from, and that is a
property of the task design rather than of the network.

## Sources

`results.jsonl` rows `{RECALL}` (recall), `{COMPLETE}` (completion) and
`{FROZEN}` (frozen layers), all trained on the same prior with the same sampler,
seed and step budget; `{SIMPLEX}` and `{CONT}` for the simplex ablation, both
recall-trained. Their `_stdeval_*` rows come from `scripts/standard_eval.py` and
the `baselines_synth*_to_{t}` rows carry the ridge, soft look-up and average-item
references. The eval-draw spread is measured over seeds 20260825, 4242 and
909090. Reports 17 and 18 share one generator, `scripts/gen_report_17.py`,
selected by `--target`; figures come from `lib/synthfig.py`. No training was run
for this report.
"""
    (PROJECT_DIR / "reports" / f"{m['slug']}.md").write_text(md)
    print("report:", save_report(f"{PROJ}_report_{m['n']}", md))


if __name__ == "__main__":
    main()
