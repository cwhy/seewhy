"""Report 25: does more context help a bounded memory? Train length x test length.

The delta rule writes every context item into one fixed dk x dk matrix per head.
At d_model=512 with 8 heads of dk=64 that state is 32,768 floats, and context
content is 832 floats per item — so M=16 is comfortably under it and M=64 is
comfortably over. Two checkpoints trained at those two lengths, each scored at
both, gives the 2x2 the question needs.

Every cell is also scored on MNIST, Fashion-MNIST and chess, because exp57
showed the synthetic held-out band and real data can disagree.

Evaluation and figures only. The runs are `experiments58.py` and
`experiments59.py`.

Run on the GPU box:
    .venv/bin/python projects/recall-gen/scripts/gen_report_25.py
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT_DIR))

from shared_lib.typst_plot import bar_chart, cm, line_chart, long_form
from shared_lib.typst_report import save_figure
from shared_lib.report import save_report

from recall_quality import analyse
from recall_images import compare_grid
from rescore import rows as read_rows

PROJ = "recall-gen"
REPORT_MD_PATH = PROJECT_DIR / "reports" / "25-more-context.md"
V = "v3"
TRAIN = [("trained at M=16", "exp60"), ("trained at M=64", "exp61")]
# The 48000-step pair the finding was first drawn from. Kept because the pair of
# budgets is the evidence that the M=64 arm had finished learning.
SHORT = [("trained at M=16", "exp58"), ("trained at M=64", "exp59")]
TEST_M = [8, 16, 32, 64]
CELLS = (16, 64)
TRANSFER = [("MNIST", "synth_long_to_mnist"),
            ("Fashion-MNIST", "synth_long_to_fashion_mnist"),
            ("chess", "synth_long_to_chess")]
STATE = 32768


def main():
    by_exp = {r["experiment"]: r for r in read_rows()}
    R = {(e, m): analyse(e, by_exp[e], M=m) for _, e in TRAIN for m in TEST_M}
    T = {(e, d, m): analyse(e, by_exp[e], as_domain=d, M=m)
         for _, e in TRAIN for _, d in TRANSFER for m in CELLS}
    nov = lambda e, m: R[(e, m)]["B_novel_present"]
    tr = lambda e, d, m: T[(e, d, m)]["B_novel_present"]

    u_img, _ = compare_grid(
        [("trained at M=16", "exp60"), ("trained at M=64", "exp61")],
        [("Synthetic, worlds never seen — read at M=64", None, "B_novel_present", 64),
         ("MNIST — read at M=64", "synth_long_to_mnist", "B_novel_present", 64),
         ("MNIST — read at M=16", "synth_long_to_mnist", "B_novel_present", 16)],
        f"{PROJ}_r25_content_{V}")
    print("fig:", u_img)

    u1 = save_figure(line_chart(
        f"{PROJ}_r25_novel",
        long_form(TEST_M, {lab: [nov(e, m)["R"] for m in TEST_M] for lab, e in TRAIN},
                  x_name="test_M", y_name="R", series_name="training"),
        x="test_M", y="R", colour="training", points=True,
        title="More context lowers recall quality, whichever length it was trained at",
        subtitle=(f"Worlds never seen. The state holds {STATE:,} floats; content is "
                  "832 floats per item, so it passes the state between M=32 and M=64."),
        x_label="context items at test time", y_label="recall quality",
        y_limits=(0.0, 1.05),
        caption=("The short-context network is above the long-context one at every "
                 "length, including at 64, which is the length the other was "
                 "trained for."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r25_novel_{V}")

    absent = {lab: [R[(e, m)]["C_seen_absent"]["nmse"] for m in TEST_M]
              for lab, e in TRAIN}
    u2 = save_figure(line_chart(
        f"{PROJ}_r25_absent",
        long_form(TEST_M, absent, x_name="test_M", y_name="nmse",
                  series_name="training"),
        x="test_M", y="nmse", colour="training", points=True,
        title="With nothing to retrieve, more context helps — the same context, opposite sign",
        subtitle=("Answer removed from the context, so the network must predict the "
                  "hidden half rather than find it. Lower is better."),
        x_label="context items at test time", y_label="normalised error",
        caption=("More items is more evidence about the world and more interference "
                 "in the store. Prediction gets the first, recall pays the second."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r25_absent_{V}")

    cell_lab = lambda lab, m: f"{lab.replace('trained at ', 'train ')}, test M={m}"
    u3 = save_figure(bar_chart(
        f"{PROJ}_r25_transfer",
        long_form([d for d, _ in TRANSFER],
                  {cell_lab(lab, m): [tr(e, dom, m)["R"] for _, dom in TRANSFER]
                   for lab, e in TRAIN for m in CELLS},
                  x_name="dataset", y_name="R", series_name="cell"),
        x="dataset", y="R", fill="cell", x_order=[d for d, _ in TRANSFER],
        position="dodge",
        title="The same four cells on real data, where the effect is far larger",
        subtitle="Both networks trained on the synthetic prior alone.",
        x_label="", y_label="recall quality", y_limits=(0.0, 1.05),
        caption=("Every dataset orders the cells the same way, and the spread is "
                 "four to five times what the synthetic band shows."),
        width=cm(17), height=cm(9)), name=f"{PROJ}_r25_transfer_{V}")

    for u in (u1, u2, u3):
        print("fig:", u)

    # ── convergence: the same two cells at both budgets ─────────────────────
    curves = {}
    for budget, pairs in (("48k steps", SHORT), ("96k steps", TRAIN)):
        for lab, e in pairs:
            h = by_exp[e]["history"]
            curves[f"{lab.replace('trained at ', '')}, {budget}"] = (
                h["step"], h["nmse"]["B_novel_present"])
    steps96 = by_exp["exp61"]["history"]["step"]
    aligned = {k: (v[1] + [None] * (len(steps96) - len(v[1])))
               for k, v in curves.items()}
    u4 = save_figure(line_chart(
        f"{PROJ}_r25_convergence",
        long_form(steps96, aligned, x_name="step", y_name="nmse",
                  series_name="cell"),
        x="step", y="nmse", colour="cell", points=False,
        title="Was the long-context arm trained enough? Yes — it stops improving at 42000",
        subtitle=("Error on worlds never seen, through training. The two budgets "
                  "are separate runs, not a resumption."),
        x_label="training step", y_label="normalised error, novel worlds",
        caption=("Both M=64 runs reach their best at step 42000 and neither goes "
                 "lower. Both M=16 runs are still descending when they stop."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r25_convergence_{V}")
    print("fig:", u4)

    def conv(e):
        h = by_exp[e]["history"]
        B, st = h["nmse"]["B_novel_present"], h["step"]
        i = min(range(len(B)), key=lambda j: B[j])
        q = len(st) // 4
        return dict(final=B[-1], best=B[i], at=st[i], last=B[-1],
                    dq=100 * (B[-1] - B[-q]) / B[-q], steps=st[-1])
    C = {e: conv(e) for e in ("exp58", "exp59", "exp60", "exp61")}

    r58, r59 = by_exp["exp60"], by_exp["exp61"]
    best_syn = nov("exp60", 16)["R"]
    worst_syn = nov("exp61", 64)["R"]

    md = f"""# More context does not help a bounded memory. It costs it.

Two networks, identical but for the number of items they were trained to hold.
The delta rule writes every context item into one fixed dk x dk matrix per head:
at d_model=512 with 8 heads of dk=64 that state is {STATE:,} floats, and an item
is 832 numbers. Sixteen items is 13,312 floats of content, comfortably under.
Sixty-four is 53,248, comfortably over.

Each network scored at both lengths, on worlds drawn from the prior it has never
seen:

| | test M=16 | test M=64 |
|---|---|---|
| **trained at M=16** | **{nov('exp60', 16)['R']:.3f}** | {nov('exp60', 64)['R']:.3f} |
| **trained at M=64** | {nov('exp61', 16)['R']:.3f} | {nov('exp61', 64)['R']:.3f} |

Every direction away from the top-left cell is worse. Longer contexts cost
recall, and training on longer contexts costs more than testing on them.

The third number is the one worth pausing on. At M=64 — the length the second
network was built for — the network trained at M=16 scores
{nov('exp60', 64)['R']:.3f} and the one trained at M=64 scores
{nov('exp61', 64)['R']:.3f}. **If you want to run a bounded memory at sixty-four
items, train it at sixteen.**

## What is being measured

**Recall quality (R).** Take the item the network's output most resembles,
measure how far it is from the true answer, and compare that to what a uniformly
random context item would cost. R = 1 means the returned item is as good as the
right one; R = 0 means no better than guessing. It does not care which index came
back, which matters here because a world's items can be near-duplicates and
returning the duplicate is correct behaviour for a fuzzy memory. Report 22 sets
this out.

**Committed.** How far the output sits from the *nearest* context item, divided
by how far apart that episode's items are. Low means the output is sitting on
something actually in memory rather than floating between items.

**Present against absent.** The same queries scored with the answer among the
context items, and again with it removed. With it removed there is nothing to
retrieve and the network must predict the hidden half from the rest.

**M** is the number of context items. Training M and test M are independent: the
architecture has no length-specific parameters, so a checkpoint can be read at
any length.

Errors are squared error over hidden coordinates, divided by the error of
ignoring the input and drawing the average item.

## The curve, not just the corners

![Recall quality against context length]({u1})

Test length is free to vary, so the 2x2 is four points on a curve with eight.
Both networks decline monotonically past M=16, and the short-context network is
above the long-context one at every single length.

Read `committed` the same way and it says the same thing: at M=16 the
short-context network sits {nov('exp60', 16)['committed']:.3f} from the nearest
stored item and the long-context one sits {nov('exp61', 16)['committed']:.3f}.
A network trained on crowded contexts barely lands on a stored item even when the
context is uncrowded.

## Where more context does help

![The answer-absent condition]({u2})

| trained at M=64, answer removed | M=8 | M=16 | M=32 | M=64 |
|---|---|---|---|---|
| normalised error | {absent['trained at M=64'][0]:.3f} | {absent['trained at M=64'][1]:.3f} | {absent['trained at M=64'][2]:.3f} | {absent['trained at M=64'][3]:.3f} |

Monotonic improvement, over exactly the range where recall gets worse.

That is the whole finding in one line. **Context is evidence for inference and
interference for retrieval.** More items means more to infer a world's structure
from, and more written into a store that does not grow. Prediction collects the
first; recall pays the second. A bounded memory makes the two pull apart, and
which one you are measuring decides whether "more context" reads as a gain or a
loss.

## Scored on real data, where it is much worse

exp57 showed that the synthetic held-out band and real datasets can move in
opposite directions — early-stopping on the former cost 0.335 of recall quality
on Fashion-MNIST. So the ranking above is not allowed to stand on synthetic
worlds alone.

![The four cells on real data]({u3})

| | train 16 / test 16 | train 16 / test 64 | train 64 / test 16 | train 64 / test 64 |
|---|---|---|---|---|
| MNIST | **{tr('exp60', 'synth_long_to_mnist', 16)['R']:.3f}** | {tr('exp60', 'synth_long_to_mnist', 64)['R']:.3f} | {tr('exp61', 'synth_long_to_mnist', 16)['R']:.3f} | {tr('exp61', 'synth_long_to_mnist', 64)['R']:.3f} |
| Fashion-MNIST | **{tr('exp60', 'synth_long_to_fashion_mnist', 16)['R']:.3f}** | {tr('exp60', 'synth_long_to_fashion_mnist', 64)['R']:.3f} | {tr('exp61', 'synth_long_to_fashion_mnist', 16)['R']:.3f} | {tr('exp61', 'synth_long_to_fashion_mnist', 64)['R']:.3f} |
| chess | **{tr('exp60', 'synth_long_to_chess', 16)['R']:.3f}** | {tr('exp60', 'synth_long_to_chess', 64)['R']:.3f} | {tr('exp61', 'synth_long_to_chess', 16)['R']:.3f} | {tr('exp61', 'synth_long_to_chess', 64)['R']:.3f} |

This time the two agree on order, on all three datasets. They disagree on size.
The worst synthetic cell keeps {100 * worst_syn / best_syn:.0f}% of the best
cell's recall quality; on MNIST it keeps
{100 * tr('exp61', 'synth_long_to_mnist', 64)['R'] / tr('exp60', 'synth_long_to_mnist', 16)['R']:.0f}%
and on chess {100 * tr('exp61', 'synth_long_to_chess', 64)['R'] / tr('exp60', 'synth_long_to_chess', 16)['R']:.0f}%.

The synthetic band understates this by a factor of four to five. Anything ranked
on it alone should be re-checked on real data before it is believed — which is
the second time in two days that has been true.

## What comes back

![Both networks, read at the same length]({u_img})

Rows three and four are the two networks answering the same episodes. The first
two blocks are read at M=64 so the comparison is at one length; the third is the
same MNIST episodes read at M=16, where both do better and the gap narrows.

## What this changes

A bounded memory has an operating length, and it is shorter than the state
budget suggests. Content passes {STATE:,} floats somewhere between M=32 and
M=64, but recall on novel worlds is already falling by M=32
({nov('exp60', 32)['R']:.3f} against {nov('exp60', 16)['R']:.3f}). The degradation
starts before the arithmetic says the store is full.

Training length is not a free parameter to match to deployment length. The usual
instinct — train at the length you will run at — is wrong here in both directions
tested.

Any future long-context work on this project should carry the answer-absent
condition alongside the present one. On a single metric the two effects partly
cancel, and reporting only one of them would have made this look like either a
clean win or a clean loss instead of a trade.

## Was the long-context arm simply undertrained?

It is the first thing to check, because the finding is a comparison and a
comparison between one converged network and one unconverged network is not a
result. It was checked by running both cells again at twice the budget, as fresh
runs with a single learning-rate schedule over the full length rather than a
restart on top of the old weights.

![Novel-worlds error through training, both cells at both budgets]({u4})

| | best on novel worlds | at step | final | change over last quarter |
|---|---|---|---|---|
| M=16, 48000 steps | {C['exp58']['best']:.4f} | {C['exp58']['at']:,} | {C['exp58']['final']:.4f} | {C['exp58']['dq']:+.1f}% |
| M=16, 96000 steps | **{C['exp60']['best']:.4f}** | {C['exp60']['at']:,} | {C['exp60']['final']:.4f} | {C['exp60']['dq']:+.1f}% |
| M=64, 48000 steps | {C['exp59']['best']:.4f} | {C['exp59']['at']:,} | {C['exp59']['final']:.4f} | {C['exp59']['dq']:+.1f}% |
| M=64, 96000 steps | {C['exp61']['best']:.4f} | {C['exp61']['at']:,} | {C['exp61']['final']:.4f} | {C['exp61']['dq']:+.1f}% |

The M=64 arm is converged and the M=16 arm is not.

Two independent runs of the M=64 cell reach their best at **step
{C['exp59']['at']:,}** — the same step — at {C['exp59']['best']:.4f} and
{C['exp61']['best']:.4f}. Those two numbers differ by
{abs(C['exp61']['best'] - C['exp59']['best']):.4f}. The extra
{C['exp61']['steps'] - C['exp59']['steps']:,} steps bought nothing at all, and
training past 42000 makes it worse: the 96000-step run ends at
{C['exp61']['final']:.4f}, above where it started plateauing.

The M=16 arm, over the same doubling, went {C['exp58']['best']:.4f} to
{C['exp60']['best']:.4f} — a {100 * (1 - C['exp60']['best'] / C['exp58']['best']):.0f}%
improvement — and is still descending at 96000, with its best at step
{C['exp60']['at']:,} of {C['exp60']['steps']:,}.

So the honest conclusion is stronger than the one the 48000-step pair supported,
not weaker. At matched budget the gap grew from
{C['exp59']['best'] / C['exp58']['best']:.1f}x to
{C['exp61']['best'] / C['exp60']['best']:.1f}x, and the arm that has room left
to improve is the short-context one. The earlier numbers understated this.

Every figure and table above uses the 96000-step pair for this reason.

## What this does not establish

That the state size is the mechanism. M=64 exceeds {STATE:,} floats of state and
recall falls, which is consistent with saturation, but state size was not varied
here. A dk sweep at fixed M would separate "the store is full" from "long
contexts are harder to learn from".

Whether attention behaves the same way. Its state grows with the number of items,
so it should not pay the interference cost — untested, and the obvious next run.

Comparability with the earlier delta-rule numbers. Both cells use batch 128,
forced by the M=64 run: the scan keeps a `(batch, heads, dk, dk)` carry per token
for the backward pass and 256 asks for 21.53 GiB on a 24 GB card. exp58 lands at
{R[('exp60', 16)]['A_seen_present']['nmse']:.4f} on trained worlds against
exp49's 0.0723 at batch 256, so this 2x2 sits below the level reported elsewhere.
It is internally consistent; it is not a continuation of the earlier series.

`committed` at M=8 is high for both networks
({nov('exp60', 8)['committed']:.3f} and {nov('exp61', 8)['committed']:.3f}).
With eight items the nearest stored item is a weaker reference, so that column
is the metric getting noisy at small M rather than a result.

That the M=16 cell is at its own ceiling. It was still improving when it
stopped, at both budgets. Its numbers here are a lower bound on what the
short-context arm can do, which only widens the gap being reported.

One seed, one width, one prior, two training lengths.

## Sources

`results.jsonl` rows `exp60` and `exp61` — the delta rule at d_model=512, 8 heads
of dk=64, trained at M=16 and M=64 on `synth_long` for 96000 steps — with `exp58`
and `exp59`, the same two cells at 48000 steps, supplying the convergence check. That domain is the project's
synthetic prior with 96 items per world rather than 48: a world holds `per_world`
items and an episode draws M+Q without replacement, so 48 caps M at 44. Changing
that count changes the random stream that builds the worlds, so `synth_long`'s
worlds are not `synth`'s and both cells had to be trained rather than reusing
exp49. `_synthetic_pools` was also giving held-out worlds the standard item count
regardless of the base, which would have capped the novel band at M=44; it now
follows the base, which is a no-op for every previously registered domain.
Metrics from `scripts/recall_quality.py`, the grid from
`scripts/ctx_len_grid.py`, drawn panels from `scripts/recall_images.py`. Episodes
are the standard eval draw, context type `class`, scored on the first query.
Figures generated by `scripts/gen_report_25.py`. Training time: exp58
{r58['time_s']:.0f}s, exp59 {r59['time_s']:.0f}s.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_25", md))


if __name__ == "__main__":
    main()
