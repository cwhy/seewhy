"""Report 21: identification on the synthetic prior fails by direction, not by size.

Report 20 attributed the shortfall to precision — the reconstruction not landing
close enough to the target — and concluded that precision tracks capacity. This
report measures the quantity identification actually depends on and finds the
explanation does not hold: an error of the same SIZE pointed at random identifies
almost perfectly, so size was never the constraint.

Evaluation and figures only. Every number is recomputed from the exp45, exp49 and
exp54 checkpoints by `scripts/diag_identification.py`; no training was run.

Run on the GPU box:
    .venv/bin/python projects/recall-gen/scripts/gen_report_21.py
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

from shared_lib.typst_plot import bar_chart, cm, line_chart, long_form
from shared_lib.typst_report import save_figure
from shared_lib.report import save_report

from diag_identification import analyse
from rescore import rows as read_rows

PROJ = "recall-gen"
REPORT_MD_PATH = PROJECT_DIR / "reports" / "21-identification-is-directional.md"
CHANCE = 1.0 / 16
V = "v1"

# (label, exp, params M) — the three checkpoints report 20 built its capacity
# argument from, reused here so the two reports are about the same networks.
RUNS = [("d256 4x64", "exp45"),
        ("d512 8x64", "exp49"),
        ("d512 4x128", "exp54")]
PARAMS = {"exp45": "4.06M", "exp49": "14.95M", "exp54": "14.94M"}
MAIN = "exp49"          # the network report 20 drew its margin figure from


def main():
    by_exp = {r["experiment"]: r for r in read_rows()}
    R = {e: analyse(e, by_exp[e]) for _, e in RUNS}
    m = R[MAIN]
    qs = m["quartiles"]
    c = m["controls"]
    mlab = [f"{q['margin']:.2f}" for q in qs]

    # ── 1. the control that decides it ───────────────────────────────────────
    d1 = long_form(
        mlab,
        {"the network's own output": [q["id_acc"] for q in qs],
         "exact answer + random error of the SAME size": [q["iso"] for q in qs],
         "exact answer": [q["oracle"] for q in qs]},
        x_name="margin", y_name="id", series_name="prediction")
    u1 = save_figure(bar_chart(
        f"{PROJ}_r21_isotropic", d1, x="margin", y="id", fill="prediction",
        x_order=mlab, position="dodge",
        title="An error of the same size, pointed at random, identifies almost perfectly",
        subtitle=("Episodes in quartiles by nearest-rival distance, left group = "
                  "closest rivals. Within a group: exact answer, then that answer "
                  "carrying the network's own error magnitude in a random "
                  "direction, then the network."),
        x_label="distance from the target to its nearest rival",
        y_label="identification accuracy", y_limits=(0.0, 1.05),
        hlines=[(CHANCE, "chance (1 of 16)")],
        caption=("If the shortfall were a matter of not landing close enough, the "
                 "middle bar would fall with the right one. It stays at the "
                 "ceiling, so the network's error is not merely large — it is "
                 "aimed."),
        width=cm(17), height=cm(9)), name=f"{PROJ}_r21_isotropic_{V}")

    # ── 2. where the error points ────────────────────────────────────────────
    d2 = long_form(
        [i + 1 for i in range(len(qs))],
        {"fraction of the way to the context centroid": [q["shrink"] for q in qs],
         "alignment with the centroid direction": [q["cos_cent"] for q in qs],
         "alignment with the nearest-rival direction": [q["cos_rival"] for q in qs]},
        x_name="margin", y_name="value", series_name="quantity")
    u2 = save_figure(line_chart(
        f"{PROJ}_r21_direction", d2, x="margin", y="value", colour="quantity",
        points=True,
        title="The pull toward the context average vanishes where identification succeeds",
        subtitle=(f"d=512, 8 heads, worlds seen in training, answer present. "
                  f"Quartile 1 has the closest rivals ({mlab[0]}), quartile 4 the "
                  f"furthest ({mlab[3]}). Alignment is a cosine: 0 unrelated, "
                  f"1 exactly along."),
        x_label="margin quartile", y_label="", y_limits=(0.0, 1.0),
        caption=("The blue curve is the finding: the output sits nearly half way "
                 "to the context average in the quartiles that fail, and barely "
                 "moves off the target in the ones that succeed."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r21_direction_{V}")

    # ── 3. undoing the shrinkage, with no weight touched ─────────────────────
    gains = sorted(float(k) for k in c["deshrink_q0"])
    d3 = long_form(
        [f"{gg:.2f}" for gg in gains],
        {lab: [R[e]["controls"]["deshrink_q0"][f"{gg:.2f}"] for gg in gains]
         for lab, e in RUNS},
        x_name="gain", y_name="id", series_name="network")
    u3 = save_figure(line_chart(
        f"{PROJ}_r21_deshrink", d3, x="gain", y="id", colour="network", points=True,
        title="Pushing the output away from the centroid recovers recall, free",
        subtitle=("Hardest quartile only. The weights are frozen; the only change "
                  "is rescaling the finished output about the context mean."),
        x_label="gain applied to (output - context centroid)",
        y_label="identification accuracy",
        hlines=[(CHANCE, "chance")],
        caption=("Gain 1.00 is the network as trained. The peak of each curve sits "
                 "where the measured shrinkage says it should."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r21_deshrink_{V}")

    # ── 4. addressing against naming ─────────────────────────────────────────
    best_layer = {e: max(R[e]["addressing"], key=lambda a: a["any_head_hit_q0"])
                  for _, e in RUNS}
    d4 = long_form(
        [lab for lab, _ in RUNS],
        {"the memory points at the target (best layer, any head)":
            [best_layer[e]["any_head_hit_q0"] for _, e in RUNS],
         "the output names the target":
            [R[e]["quartiles"][0]["id_acc"] for _, e in RUNS]},
        x_name="network", y_name="value", series_name="stage")
    u4 = save_figure(bar_chart(
        f"{PROJ}_r21_addressing", d4, x="network", y="value", fill="stage",
        x_order=[lab for lab, _ in RUNS], position="dodge",
        title="The memory finds the right item more often than the output says so",
        subtitle=(f"Hardest quartile, worlds seen in training, answer present. "
                  f"{RUNS[0][0]} is {PARAMS['exp45']} parameters; the other two are "
                  f"{PARAMS['exp49']} and {PARAMS['exp54']}."),
        x_label="", y_label="fraction of episodes", y_limits=(0.0, 1.0),
        hlines=[(CHANCE, "chance")],
        caption=("The retrieval weights are read off the state itself, so this is "
                 "what the memory did, not an inference about what it could do."),
        width=cm(17), height=cm(9)), name=f"{PROJ}_r21_addressing_{V}")

    for u in (u1, u2, u3, u4):
        print("fig:", u)

    # ── numbers the prose quotes ─────────────────────────────────────────────
    q0 = qs[0]
    bl = best_layer[MAIN]
    peak = lambda e, key: max(R[e]["controls"][key].items(), key=lambda kv: kv[1])
    pk49, pk49v = peak(MAIN, "deshrink_q0")
    pk49a, pk49av = peak(MAIN, "deshrink")
    pred_gain = lambda e: 1.0 / (1.0 - R[e]["quartiles"][0]["shrink"])
    r45, r54 = R["exp45"], R["exp54"]
    p45, p45v = peak("exp45", "deshrink_q0")
    p54, p54v = peak("exp54", "deshrink_q0")
    nhid = int(np.round(1.0 / (q0["cos_rival"] ** 2))) if q0["cos_rival"] else 0

    md = f"""# Recall on the synthetic prior fails by direction, not by size

> **Superseded in part by report 22.** The mechanics below hold: identification
> depends on the direction of the error rather than its size, the error leans
> toward the average of the context, and the memory addresses the right slot more
> often than the output names it. The framing does not. This report treats
> recovering identification as the goal, and measured as content rather than as an
> index the same outputs already score 0.987 — there was no shortfall to recover.
> Read the derivation here as a reason to stop using argmin over near-duplicates,
> not as a repair to the network.

Report 20 measured a network that reconstructs an item well and still cannot say
which item it is. It explained that as precision: the reconstruction lands
{qs[0]['err']:.3f} from the target while the nearest rival sits
{qs[0]['margin']:.3f} away, so the output is not close enough to win. From there
it concluded that precision tracks capacity, and that scale is the lever.

The explanation does not survive its own control. Take the exact answer, corrupt
it with an error of **the same size the network makes**, point that error in a
random direction, and ask the same question. It identifies the right item
**{c['isotropic_same_norm']:.3f}** of the time overall, and
**{qs[0]['iso']:.3f}** of the time in the quartile where the network scores
{qs[0]['id_acc']:.3f}.

Size was never the constraint. The network's error is not too big. It is aimed.

This report measures where it is aimed, and finds that a correction with no
trained parameters in it recovers a large part of the shortfall.

## The measurements, defined

Every term this report uses, in the order it needs them.

**Item, world, episode.** One *item* is a vector of {R[MAIN]['cfg']['d_in']}
numbers. A *world* is a generative process that produces items — a random linear
map from a latent code whose dimension is drawn as low as 1, so items from one
world can lie almost on a line. An *episode* is sixteen items from a single
world, followed by a seventeenth.

**Context, query, mask.** The first sixteen items are the *context*; the network
sees them whole. The seventeenth is the *query*, and roughly half its
coordinates are erased by a *mask*. Erased coordinates are *hidden*; the rest are
*visible*. The network outputs the hidden coordinates.

**Recall.** In these runs the query is always an exact copy of one of the
sixteen context items. The answer is present, and the task is to find it.

**Identification.** The metric this report is about. Take the network's output,
measure its squared distance to each of the sixteen context items on hidden
coordinates only, and see whether the closest one is the item the query was
copied from. *Chance* is 1 in 16, or {CHANCE:.3f}. The *ceiling* — feeding the
true answer in and asking the same question — is {c['oracle_id']:.3f} here.
Hidden coordinates only, so a network that merely copies the visible half of the
query cannot score.

**Nearest rival, and the margin.** For a given episode, the *nearest rival* is
whichever of the other fifteen context items sits closest to the true target.
The *margin* is that distance. It is a property of the episode, computable with
no network involved. Episodes are split into four *quartiles* by margin, so the
left-hand quartile is the quarter of episodes whose candidates are hardest to
tell apart.

**Reconstruction error.** Mean squared error between the output and the true
target, over hidden coordinates, divided by the error of ignoring the input and
drawing the average training item. So 1.0 is the do-nothing score, and the
margin is quoted in the same units.

**The error vector.** Write `e` for `output - target` and `Delta` for
`nearest rival - target`, both restricted to hidden coordinates. These are the
only two quantities identification depends on, which the next section derives.

**rho.** The component of `e` along `Delta`, divided by the length of `Delta`.
The single number that decides whether the nearest rival beats the target.

**Alignment.** A cosine between `e` and some direction: 0 means unrelated, 1
means exactly along it. Reported against `Delta` and against the direction from
the target to the *context centroid*, which is the plain average of the sixteen
context items.

**Shrinkage.** How far `e` travels from the target toward that centroid, as a
fraction of the whole distance. 0 is no pull; 1 lands on the average of the
context.

**Retrieval weights.** Defined in the addressing section below, where they are
needed.

## Why only one direction matters

Identification compares two squared distances. Expanding the second around the
first, with `e` and `Delta` as above:

    ||output - rival||^2 - ||output - target||^2  =  ||Delta||^2 - 2 e.Delta

The rival wins exactly when that difference is negative, which rearranges to a
condition on one number:

    rho  =  e.Delta / ||Delta||^2  >  1/2

Read the identity rather than the algebra. Every part of the error that points
across the target-to-rival line cancels out of both distances. Only the part
that points *along* it survives. The size of the error appears nowhere.

This is why an error of the network's own magnitude, pointed at random, is
harmless. The hidden coordinates number a few hundred, and a random direction
puts about one over the square root of that on any particular axis. The project
had already recorded this without drawing the conclusion: `nearest_distractor`
in `lib/splitfig.py` carries a note that an earlier isotropic-noise sweep left
identification at 1.000 out to sigma 1.1, and abandoned the measurement as
uninformative. It was not uninformative. It was the answer.

![Identification under three predictions of the same episodes]({u1})

The middle bar is the control. It has the network's error magnitude, quartile by
quartile, and none of the network's error direction.

## Where the error actually points

If not at random, then where. Two candidate directions are worth measuring: the
nearest rival, which is what an interference story predicts, and the centroid of
the context, which is what a hedging story predicts.

![Alignment of the error with two directions]({u2})

| margin quartile | {mlab[0]} | {mlab[1]} | {mlab[2]} | {mlab[3]} |
|---|---|---|---|---|
| identification | {qs[0]['id_acc']:.3f} | {qs[1]['id_acc']:.3f} | {qs[2]['id_acc']:.3f} | {qs[3]['id_acc']:.3f} |
| reconstruction error | {qs[0]['err']:.3f} | {qs[1]['err']:.3f} | {qs[2]['err']:.3f} | {qs[3]['err']:.3f} |
| rho | {qs[0]['rho']:+.3f} | {qs[1]['rho']:+.3f} | {qs[2]['rho']:+.3f} | {qs[3]['rho']:+.3f} |
| shrinkage toward centroid | {qs[0]['shrink']:.3f} | {qs[1]['shrink']:.3f} | {qs[2]['shrink']:.3f} | {qs[3]['shrink']:.3f} |
| alignment with centroid | {qs[0]['cos_cent']:+.3f} | {qs[1]['cos_cent']:+.3f} | {qs[2]['cos_cent']:+.3f} | {qs[3]['cos_cent']:+.3f} |
| alignment with nearest rival | {qs[0]['cos_rival']:+.3f} | {qs[1]['cos_rival']:+.3f} | {qs[2]['cos_rival']:+.3f} | {qs[3]['cos_rival']:+.3f} |

Read the shrinkage row first. It falls from {qs[0]['shrink']:.3f} to
{qs[3]['shrink']:.3f} across the quartiles, tracking identification almost
exactly: where the output sits nearly half way to the average of its sixteen
candidates, it cannot name any of them; where it stays on the target, it can.

The two alignments are closer together, and the centroid direction wins by less
than the shrinkage row suggests. It leads in the two quartiles that fail
({qs[0]['cos_cent']:.2f} against {qs[0]['cos_rival']:.2f}, and
{qs[1]['cos_cent']:.2f} against {qs[1]['cos_rival']:.2f}) and the two are level
in the third ({qs[2]['cos_cent']:.2f} against {qs[2]['cos_rival']:.2f}), where
identification is already {qs[2]['id_acc']:.3f} and the question is moot. So the
centroid is the better description of the error where the error matters, and the
two directions are not cleanly separable elsewhere.

The shrinkage row is not what confusion between two similar items looks like.
Pairwise confusion would leave the output on the line between two candidates
without pulling it toward the average of all sixteen, and would not switch off
as the margin grows. This is regression to the mean of the context — and it is
what squared error asks for. A network uncertain about
which of sixteen items it is looking at minimises expected squared error by
outputting their average, weighted by how likely each is. Identification then
takes a hard nearest-neighbour decision over that average and is punished for
exactly the hedge the loss paid for.

Note the row that report 20 rested on. `rho` in the hardest quartile is
{qs[0]['rho']:+.3f}, and the threshold is 0.500. The network sits on the
decision boundary. Identification there is close to a coin toss, which makes it
a poor dependent variable for comparing architectures — small changes in the
network move it by large, noisy amounts, or not at all.

## Undoing the shrinkage without touching a weight

Shrinkage is a one-parameter defect, so it has a one-parameter correction. Take
the finished output, and push it away from the context centroid:

    corrected  =  centroid + gain * (output - centroid)

No retraining. No gradient. The weights are frozen and the network is not
consulted a second time.

![Identification against de-shrink gain, hardest quartile]({u3})

| network | as trained | best gain | at that gain | gain predicted by the shrinkage |
|---|---|---|---|---|
| {RUNS[0][0]}, {PARAMS['exp45']} | {r45['quartiles'][0]['id_acc']:.3f} | x{p45} | {p45v:.3f} | x{pred_gain('exp45'):.2f} |
| {RUNS[1][0]}, {PARAMS['exp49']} | {qs[0]['id_acc']:.3f} | x{pk49} | {pk49v:.3f} | x{pred_gain('exp49'):.2f} |
| {RUNS[2][0]}, {PARAMS['exp54']} | {r54['quartiles'][0]['id_acc']:.3f} | x{p54} | {p54v:.3f} | x{pred_gain('exp54'):.2f} |

The last column is the check that makes this an explanation rather than a fitted
curve. If the error is a pull of size `s` toward the centroid, the correction
that undoes it is a gain of `1/(1-s)`, computed from the shrinkage column of the
previous table and never from identification. The predicted and the measured
optimum agree on all three networks.

Over all episodes rather than the hardest quartile, {MAIN} goes from
{c['model_id']:.3f} to {pk49av:.3f} at gain x{pk49a}.

Set that beside what it is competing with. The parameter increase report 20
built its conclusion on — 4.06M to 14.95M, roughly four times the parameters and
twice the training cost — bought +0.145 on this metric. A single scalar applied
after the fact buys {pk49av - c['model_id']:+.3f} overall and
{pk49v - qs[0]['id_acc']:+.3f} on the quartile the whole diagnosis rested on.

It is a partial correction, not a fix: {pk49v:.3f} against a ceiling of
{c['oracle_id']:.3f}. Some of the deficit is real error. Roughly a third of it
was never a deficit at all.

## The memory is not where it is lost

Report 20's other claim was that the memory is not the bottleneck, argued from a
capacity sweep. The claim is right and the argument was indirect. The memory can
be read.

Recall here runs on a matrix-valued state written by the delta rule. Each context
token writes `S <- S + e_i k_i^T`, where `k_i` is that item's *key* and `e_i` is
the correction the write applies, and between writes the state decays by a
learned per-channel factor. After all sixteen writes the state is a sum of those
outer products, so the query's read is

    output  =  sum_i  e_i * w_i,      w_i = (k_i * A_i) . q / sqrt(dk)

with `A_i` the decay that survives from write `i` to the end and `q` the query's
own key. Those `w_i` are the *retrieval weights*: this architecture's version of
an attention distribution over the sixteen items, recovered from the state rather
than inferred. Whether `w_i` is largest at the target separates *addressing* the
right slot from *reconstructing* what is in it.

![Addressing against naming, hardest quartile]({u4})

On the hardest quartile, in {MAIN}'s best layer (layer {bl['layer']}), at least
one head puts its largest retrieval weight on the correct item in
**{bl['any_head_hit_q0']:.3f}** of episodes. The finished output names that item
in **{qs[0]['id_acc']:.3f}**.

Addressing runs ahead of naming, on every network tested. Whatever is lost is
lost after the state has already found the item. That is a direct measurement of
the thing the capacity sweep could only probe, and it settles it in the same
direction.

One further observation the scalar metric hides. In {RUNS[2][0]} the addressing
concentrates into a single sharp layer — layer
{max(r54['addressing'], key=lambda a: a['z_gap'])['layer']} reaches a
separation of {max(a['z_gap'] for a in r54['addressing']):+.2f} standard
deviations between the target's weight and the rival's — while another layer
goes blind. In {RUNS[1][0]} the same work is spread evenly across all four
layers. Same parameter count, same final score, different machine. Doubling the
head dimension reorganised the computation rather than adding to it.

## What this changes

Report 20's headline claim about the memory stands. Its explanation of the
shortfall does not, and the recommendation that followed from it should be
withdrawn.

The sentence "this is a precision problem, and precision tracks capacity" is
wrong on the first clause, which makes the second irrelevant. Identification does
not depend on how close the output lands. It depends on which way the output
leans, and it leans toward the average of the context because squared error pays
for leaning that way.

Three things follow.

Identification should be quoted with `rho` beside it, or with the isotropic
control beside it. A number that can be moved by {pk49av - c['model_id']:+.3f}
with a scalar is not measuring what it appears to measure.

Low-margin episodes are a step function evaluated at its own threshold. Sweeping
architectures against identification on the hardest quartile measures noise
around a decision boundary. Either report the margin-resolved table, or use a
metric that degrades smoothly.

The next experiment on this prior is an objective, not a size. Squared error over
a set of candidates selects for the hedge this report measures. A discrimination
term, or a whitened output space, addresses the cause; more parameters buy a
smaller hedge only incidentally.

## What this does not establish

That capacity is irrelevant. The correction is partial, the largest network is
still the best at every gain, and {RUNS[0][0]} responds to it far less
({p45v:.3f} at its own optimum) because more of its error is genuinely
misdirected rather than merely shrunk.

That the fix transfers. The gain is one number chosen on the same 512 episodes
it is scored on. The peak is broad and it is predicted independently by the
shrinkage, so it is not a fitted artefact, but it has not been validated on a
held-out draw.

That this is the whole error. Alignment with the centroid is
{qs[0]['cos_cent']:.2f}, not 1.00. A substantial part of the error points
somewhere neither direction tested here describes.

Anything about the absent-answer task. Every number here is from
`A_seen_present`: worlds seen in training, answer present in the context. This
report is about recall, not about completion.

## Sources

`results.jsonl` rows `exp45`, `exp49` and `exp54` — recall training on the
synthetic prior at d_model 256/512 and two head/dk splits, the same three
checkpoints report 20 used. Every number is recomputed from those checkpoints by
`scripts/diag_identification.py`, which also carries the derivation above in its
module docstring. Episodes are the project's standard eval draw at M=16,
condition `A_seen_present`, context type `class` — single-world episodes, the
type these networks were trained on. Identification is scored on the first query
of each episode, matching `lib/splitfig.nearest_distractor`; published numbers
average four queries, which is why {MAIN} reads {c['model_id']:.3f} here against
0.759 there. Figures generated by `scripts/gen_report_21.py`. No training was run
for this report.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_21", md))


if __name__ == "__main__":
    main()
