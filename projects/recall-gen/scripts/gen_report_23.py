"""Report 23: swapping the mixer roughly doubles recall on real data.

Report 22 established the metric — whether the content that comes back is the
content that was stored — and with it a baseline: the delta-rule network recalls
almost perfectly inside its training worlds and collapses on real datasets. This
report runs the control that follows, and that this project had never run.

Evaluation and figures only, over exp49 and exp56. No training was run here;
exp56 itself is `experiments56.py`.

Run on the GPU box:
    .venv/bin/python projects/recall-gen/scripts/gen_report_23.py
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT_DIR))

from shared_lib.typst_plot import bar_chart, cm, long_form
from shared_lib.typst_report import save_figure
from shared_lib.report import save_report

from recall_quality import analyse as quality
from recall_images import compare_grid
from rescore import rows as read_rows

PROJ = "recall-gen"
REPORT_MD_PATH = PROJECT_DIR / "reports" / "23-attention-transfers.md"
V = "v1"
KDA, ATT = "exp49", "exp56"
LAB = {KDA: "delta rule (exp49)", ATT: "attention (exp56)"}
# (label in figures, domain to score under, None = the domain it trained on)
POOLS = [("synthetic, novel worlds", None),
         ("MNIST", "synth_to_mnist"),
         ("Fashion-MNIST", "synth_to_fashion_mnist"),
         ("chess", "synth_to_chess")]


def main():
    by_exp = {r["experiment"]: r for r in read_rows()}
    S = {}
    for e in (KDA, ATT):
        S[e] = {"self": quality(e, by_exp[e])}
        for _, d in POOLS:
            if d:
                S[e][d] = quality(e, by_exp[e], as_domain=d)

    def novel(e, d):
        return S[e]["self" if d is None else d]["B_novel_present"]

    def seen(e):
        return S[e]["self"]["A_seen_present"]

    u_img, img_stats = compare_grid(
        [(LAB[KDA], KDA), (LAB[ATT], ATT)],
        [("Synthetic — worlds seen in training", None, "A_seen_present"),
         ("Synthetic — worlds never seen", None, "B_novel_present"),
         ("MNIST", "synth_to_mnist", "B_novel_present"),
         ("Fashion-MNIST", "synth_to_fashion_mnist", "B_novel_present"),
         ("Chess", "synth_to_chess", "B_novel_present")],
        f"{PROJ}_r23_allpools_{V}")
    print("fig:", u_img)
    print(json.dumps(img_stats, indent=2))

    labs = [lab for lab, _ in POOLS]
    d1 = long_form(labs,
                   {LAB[KDA]: [novel(KDA, d)["R"] for _, d in POOLS],
                    LAB[ATT]: [novel(ATT, d)["R"] for _, d in POOLS]},
                   x_name="pool", y_name="R", series_name="mixer")
    u1 = save_figure(bar_chart(
        f"{PROJ}_r23_quality", d1, x="pool", y="R", fill="mixer", x_order=labs,
        position="dodge",
        title="Recall quality on data neither network trained on",
        subtitle=("Both trained only on the synthetic prior. 1.0 means the item that "
                  "came back is as good as the right one; 0.0 means no better than "
                  "grabbing a context item at random."),
        x_label="", y_label="recall quality", y_limits=(0.0, 1.05),
        caption=("Attention carries 13.88M parameters against the delta rule's "
                 "14.95M, so the comparison runs against it."),
        width=cm(17), height=cm(9)), name=f"{PROJ}_r23_quality_{V}")

    d2 = long_form(["synthetic, worlds seen"] + labs,
                   {LAB[KDA]: [seen(KDA)["committed"]]
                              + [novel(KDA, d)["committed"] for _, d in POOLS],
                    LAB[ATT]: [seen(ATT)["committed"]]
                              + [novel(ATT, d)["committed"] for _, d in POOLS]},
                   x_name="pool", y_name="committed", series_name="mixer")
    u2 = save_figure(bar_chart(
        f"{PROJ}_r23_committed", d2, x="pool", y="committed", fill="mixer",
        x_order=["synthetic, worlds seen"] + labs, position="dodge",
        title="What is still broken: the output does not land on a stored item",
        subtitle=("Distance from the output to the nearest context item, divided by "
                  "how far apart that episode's items are. Lower is better."),
        x_label="", y_label="distance to nearest stored item, at episode scale",
        caption=("Attention improves every pool and closes none of them. Against "
                 f"{seen(ATT)['committed']:.3f} on the worlds it trained on, real data "
                 "still reads five to ten times higher."),
        width=cm(17), height=cm(9)), name=f"{PROJ}_r23_committed_{V}")

    pk, pa = S[KDA]["self"]["present_absent"], S[ATT]["self"]["present_absent"]
    d3 = long_form([LAB[KDA], LAB[ATT]],
                   {"answer present in the context": [pk["A_present"], pa["A_present"]],
                    "answer removed": [pk["C_absent"], pa["C_absent"]]},
                   x_name="mixer", y_name="nmse", series_name="condition")
    u3 = save_figure(bar_chart(
        f"{PROJ}_r23_present_absent", d3, x="mixer", y="nmse", fill="condition",
        x_order=[LAB[KDA], LAB[ATT]], position="dodge",
        title="How much the context is worth, on the worlds each network trained on",
        subtitle="Same queries, same episodes; only whether the answer is among the sixteen.",
        x_label="", y_label="normalised error", hlines=[(1.0, "ignore the input")],
        caption="The gap is recall. Attention widens it from 5.7x to 18.4x.",
        width=cm(16), height=cm(9)), name=f"{PROJ}_r23_present_absent_{V}")

    for u in (u1, u2, u3):
        print("fig:", u)

    rk, ra = by_exp[KDA], by_exp[ATT]
    gap_k = pk["C_absent"] / pk["A_present"]
    gap_a = pa["C_absent"] / pa["A_present"]
    mn_k, mn_a = novel(KDA, "synth_to_mnist"), novel(ATT, "synth_to_mnist")
    fa_k, fa_a = novel(KDA, "synth_to_fashion_mnist"), novel(ATT, "synth_to_fashion_mnist")
    ch_k, ch_a = novel(KDA, "synth_to_chess"), novel(ATT, "synth_to_chess")
    sy_k, sy_a = novel(KDA, None), novel(ATT, None)

    md = f"""# Attention roughly doubles recall on real data, with fewer parameters

Two networks, identical in width, depth, heads, steps, learning rate, seed and
training episodes. They differ in one thing: how the sixteen context items reach
the query. One compresses them into a fixed matrix with the delta rule. The other
attends to them.

Trained on the synthetic prior alone and then asked to recall from datasets
neither has seen:

| recalling from | delta rule | attention | |
|---|---|---|---|
| synthetic worlds it never saw | {sy_k['R']:.3f} | **{sy_a['R']:.3f}** | |
| MNIST | {mn_k['R']:.3f} | **{mn_a['R']:.3f}** | +{100 * (mn_a['R'] / mn_k['R'] - 1):.0f}% |
| Fashion-MNIST | {fa_k['R']:.3f} | **{fa_a['R']:.3f}** | +{100 * (fa_a['R'] / fa_k['R'] - 1):.0f}% |
| chess | {ch_k['R']:.3f} | **{ch_a['R']:.3f}** | |

Attention did this with **{ra['n_params'] / 1e6:.2f}M parameters against
{rk['n_params'] / 1e6:.2f}M** — {100 * (1 - ra['n_params'] / rk['n_params']):.0f}%
fewer — and trained in **{ra['time_s']:.0f} seconds against {rk['time_s']:.0f}**.

For scale: the capacity increase report 20 built its argument on took the same
architecture from 4.06M to 14.95M parameters, four times the training cost, and
did essentially nothing for transfer. Changing the mixer roughly doubles it.

## What is being measured

**The task.** Sixteen items from one world form the *context*; a seventeenth, the
*query*, is an exact copy of one of them with about half its coordinates erased.
The network produces the erased half. The answer is in memory, and the job is to
get it back.

**Recall quality (R).** Take the item the network's output most resembles. Measure
how far that item is from the true answer — the *cost* of the mistake — and
compare it to what a uniformly random context item would cost:

    R  =  1  -  cost / cost of a random pick

R = 1 means the returned item is as good as the right one. R = 0 means no better
than guessing. It does not care which index came back, which is the point: when a
world's items are near-duplicates, returning the duplicate is correct behaviour
for a fuzzy memory. Report 22 covers why the older `identification` metric —
argmin over the sixteen, scoring 0 for a near-duplicate — is the wrong question
here, and is wrong in both directions.

**Committed.** How far the output sits from the *nearest* context item, divided by
how far apart that episode's items are. Low means the output is sitting on
something that is actually in memory rather than floating between items. The
division matters: distances grow with how spread out a world is, so the raw
number is not comparable across pools.

**Present against absent.** The same queries scored once with the answer among the
sixteen and once with it removed. The gap is the whole of recall and involves no
index.

Errors are squared error over hidden coordinates, divided by the error of
ignoring the input and drawing the average item, so 1.0 is the do-nothing score.

**The two mixers.** The delta rule writes each context item into one dk x dk state
matrix, `S <- S + e k^T`, and the query reads `S q`. That read is a linear
combination of everything written, so two similar keys blur into each other, and
sixteen items of {rk['cfg']['d_in']} numbers are compressed into
{rk['state_floats']:,} floats. Attention keeps every item and picks among them
with a softmax, which can be arbitrarily sharp. Its cost is quadratic in sequence
length, which is irrelevant at seventeen tokens.

The two are held to the same information channel. Under the delta rule query
tokens never write, so nothing can read them; under attention they are masked out
as keys. Causal masking is kept because a scan cannot see the future.

## What comes back

![Every pool, both architectures]({u_img})

Five pools, five episodes each, both networks answering the same episodes. The
columns are the 10th to 90th percentile of how far the **delta rule's** output
sits from the nearest stored item, so the episodes are chosen by the baseline
rather than by the comparison.

Read the top block against the bottom three. On worlds it trained on, both rows
of output are the stored item. On MNIST and Fashion-MNIST neither is, and
attention's row is the closer of the two without being the answer.

## Recall quality across four pools

![Recall quality]({u1})

The synthetic column is worth reading first: {sy_k['R']:.3f} to {sy_a['R']:.3f} on
worlds drawn from the same prior the network trained on but never seen. That gap
is small because the delta rule was already close to solving it. The real-data
columns are where the mixers separate.

Chess moves least in relative terms, from {ch_k['R']:.3f} to {ch_a['R']:.3f}, and
the reason is in the prior: 40% of the synthetic worlds are drawn in simplex mode,
one active coordinate per group of thirteen, which is exactly a chess piece plane.
Chess sits nearly inside the prior's support. MNIST and Fashion-MNIST do not, and
they are where the change is worth {100 * (mn_a['R'] / mn_k['R'] - 1):.0f}% and
{100 * (fa_a['R'] / fa_k['R'] - 1):.0f}%.

## The context is worth more to attention

![Present against absent]({u3})

| | answer present | answer removed | ratio |
|---|---|---|---|
| delta rule | {pk['A_present']:.4f} | {pk['C_absent']:.4f} | {gap_k:.1f}x |
| attention | {pa['A_present']:.4f} | {pa['C_absent']:.4f} | **{gap_a:.1f}x** |

Both numbers move in the right direction: attention is better with the answer
present and *worse* with it absent. That is the signature of a network leaning
harder on retrieval and less on a learned prior, which is what a memory system
should do.

## What attention does not fix

![Distance to the nearest stored item]({u2})

| distance to nearest stored item, at episode scale | delta rule | attention |
|---|---|---|
| synthetic, worlds seen in training | {seen(KDA)['committed']:.3f} | **{seen(ATT)['committed']:.3f}** |
| synthetic, worlds never seen | {sy_k['committed']:.3f} | {sy_a['committed']:.3f} |
| MNIST | {mn_k['committed']:.3f} | {mn_a['committed']:.3f} |
| Fashion-MNIST | {fa_k['committed']:.3f} | {fa_a['committed']:.3f} |
| chess | {ch_k['committed']:.3f} | {ch_a['committed']:.3f} |

Every pool improves and none of them closes. Against
{seen(ATT)['committed']:.3f} on the worlds it trained on, attention still reads
{mn_a['committed']:.3f} on MNIST and {fa_a['committed']:.3f} on Fashion-MNIST —
five to ten times further from anything actually in memory.

So the compression was a real limiter and it was not the only one. On a genuinely
new distribution both networks produce something in the right neighbourhood
rather than the stored item itself. Sharper addressing raised the quality of the
neighbourhood; it did not make the network commit.

Two suspects remain, and they are the ones this experiment was designed to
isolate down to. A single `W_pix` maps {rk['cfg']['d_in']} raw numbers to the
embedding before any mixing, so items differing in ways the synthetic prior never
varied can be collapsed before a mixer ever sees them — which would look exactly
like this. And squared error still pays for hedging between candidates, which is
what `committed` measures; attention made that hedge cheaper to avoid without
making it unrewarded.

## What this changes

The architecture question that reports 19 and 20 answered with capacity has a
much larger answer in the mixer. Nothing in the capacity sweep approached
{100 * (mn_a['R'] / mn_k['R'] - 1):.0f}% on transfer, and this run cost
{ra['time_s'] / 60:.0f} minutes.

Any further work on this prior should carry an attention arm. The delta rule is
the interesting object — a bounded memory is the point of the architecture — but
it now has a reference above it, and the gap between them is the thing to close.

`n_heads x dk == d_model` is still asserted in `init_params`. It is what
confounded report 20's capacity arms, and it should be removed before any further
capacity comparison, on either mixer.

## What this does not establish

That attention is the right answer. Its state grows with the number of items; the
delta rule's does not. For a memory that must hold more than sixteen things the
comparison has not been run, and the interesting question — sharp addressing at
bounded state — is untouched by this result.

That the parameter gap is irrelevant. Attention won with
{100 * (1 - ra['n_params'] / rk['n_params']):.0f}% fewer parameters, so the gap
runs against it and the result is conservative. Had it lost, this would need a
parameter-matched rerun.

That the remaining deficit is the embedding or the objective. Those are the two
candidates left standing; neither has been tested.

One seed, one width, one depth, one prior.

## Sources

`results.jsonl` rows `exp49` and `exp56`. exp56 is `experiments56.py`, identical
to exp49 but for `Cfg(mixer="softmax")`; the mixer is a `Cfg` field defaulted to
`"kda"`, so every row already written still rebuilds and the delta-rule path,
including its consumption of random numbers, is unchanged. Recall quality, cost,
item spacing and distance to the nearest stored item come from
`scripts/recall_quality.py`, whose `--as-domain` scores a synth-trained
checkpoint under `synth_to_mnist`, `synth_to_fashion_mnist` and `synth_to_chess`:
the A band stays synthetic training worlds and the B band becomes the real
dataset. Drawn panels from `scripts/recall_images.py`. Episodes are the project's
standard eval draw at M=16, context type `class`, scored on the first query.
Figures generated by `scripts/gen_report_23.py`.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_23", md))


if __name__ == "__main__":
    main()
