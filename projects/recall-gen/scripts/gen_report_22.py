"""Report 22: recall on the synthetic prior is not broken; the metric is.

Reports 19 and 20 headline `identification` — an argmin over the sixteen context
items, scoring 1 for the target and 0 for every other item including a
near-duplicate. On a prior whose worlds are drawn with a latent dimension as low
as 1, near-duplicates are the norm, so that metric charges the network for a
property of the data.

This report measures what a fuzzy recall system is actually for: whether the
content that comes back is the content that was stored. It also draws it, which
no report on the synthetic prior has done.

Evaluation and figures only. No training was run.

Run on the GPU box:
    .venv/bin/python projects/recall-gen/scripts/gen_report_22.py
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

from shared_lib.typst_plot import bar_chart, cm, line_chart, long_form
from shared_lib.typst_report import save_figure
from shared_lib.report import save_report

from recall_quality import analyse as quality
from recall_images import figure as content_figure
from rescore import rows as read_rows

PROJ = "recall-gen"
REPORT_MD_PATH = PROJECT_DIR / "reports" / "22-what-comes-back.md"
CHANCE = 1.0 / 16
V = "v1"
MAIN = "exp49"
RUNS = [("d256 4x64", "exp45", "4.06M"),
        ("d512 8x64", "exp49", "14.95M"),
        ("d512 4x128", "exp54", "14.94M")]


def main():
    by_exp = {r["experiment"]: r for r in read_rows()}
    Q = {e: quality(e, by_exp[e]) for _, e, _ in RUNS}
    m = Q[MAIN]
    A, B = m["A_seen_present"], m["B_novel_present"]
    pa = m["present_absent"]

    u_content, cs = content_figure(MAIN, name=f"{PROJ}_r22_content_{V}")
    print("fig:", u_content)

    # ── 1. the two metrics, quartile by quartile ─────────────────────────────
    qlab = [f"{q['margin']:.2f}" for q in cs["by_q"]["seen"]]
    d1 = long_form(
        qlab,
        {"identification (which index came back)": [q["id_acc"] for q in A["by_q"]],
         "recall quality (whether the content was right)": [q["R"] for q in A["by_q"]]},
        x_name="margin", y_name="value", series_name="metric")
    u1 = save_figure(bar_chart(
        f"{PROJ}_r22_two_metrics", d1, x="margin", y="value", fill="metric",
        x_order=qlab, position="dodge",
        title="The two metrics disagree by half, on the same outputs",
        subtitle=(f"{MAIN}, worlds seen in training, answer present. Episodes in "
                  "quartiles by how far the target sits from its nearest rival; "
                  "left group = the hardest."),
        x_label="distance from the target to its nearest rival", y_label="",
        y_limits=(0.0, 1.05),
        caption=("Recall quality asks how much the mistake costs: how far the "
                 "returned item sits from the true answer, against how far a "
                 "randomly grabbed context item would sit. 1.0 means the mistake "
                 "was free."),
        width=cm(17), height=cm(9)), name=f"{PROJ}_r22_two_metrics_{V}")

    # ── 2. where the real deficit is ─────────────────────────────────────────
    d2 = long_form(
        [i + 1 for i in range(4)],
        {"worlds seen in training": [q["ratio"] for q in cs["by_q"]["seen"]],
         "worlds never seen": [q["ratio"] for q in cs["by_q"]["novel"]]},
        x_name="quartile", y_name="ratio", series_name="pool")
    u2 = save_figure(line_chart(
        f"{PROJ}_r22_committed", d2, x="quartile", y="ratio", colour="pool",
        points=True,
        title="On a world it has never seen, the output stops landing on a stored item",
        subtitle=("Distance from the output to the NEAREST context item, divided by "
                  "how far apart that episode's items are. Low means the output is "
                  "sitting on something that is actually in memory."),
        x_label="margin quartile (1 = closest rivals)",
        y_label="distance to nearest stored item, at episode scale",
        y_limits=(0.0, 0.5),
        caption=("This is the deficit identification hides: in quartile 3 it reads "
                 f"{cs['by_q']['novel'][2]['id']:.3f} on novel worlds while the "
                 "output is further from any stored item than anywhere on the "
                 "trained pool."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r22_committed_{V}")

    # ── 3. is the context used at all ────────────────────────────────────────
    d3 = long_form(
        [lab for lab, _, _ in RUNS],
        {"answer present in the context":
            [Q[e]["present_absent"]["A_present"] for _, e, _ in RUNS],
         "answer removed":
            [Q[e]["present_absent"]["C_absent"] for _, e, _ in RUNS]},
        x_name="network", y_name="nmse", series_name="condition")
    u3 = save_figure(bar_chart(
        f"{PROJ}_r22_present_absent", d3, x="network", y="nmse", fill="condition",
        x_order=[lab for lab, _, _ in RUNS], position="dodge",
        title="The context is used, heavily",
        subtitle=("Same queries, same weights; the only difference is whether the "
                  "answer is among the sixteen. Lower is better."),
        x_label="", y_label="normalised error", hlines=[(1.0, "ignore the input")],
        caption=("The gap is the whole of recall. It is the measurement a fuzzy "
                 "memory system should be judged on first, and it does not depend "
                 "on any index being returned."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r22_present_absent_{V}")

    for u in (u1, u2, u3):
        print("fig:", u)

    sq, nq = cs["by_q"]["seen"], cs["by_q"]["novel"]
    r45, r54 = Q["exp45"], Q["exp54"]

    md = f"""# What comes back is right. The metric saying otherwise is the problem.

The network is shown sixteen items from one world and a seventeenth that is a
copy of one of them with half its coordinates erased. On worlds it trained on it
scores **{A['id_acc']:.3f}** at naming which of the sixteen the answer was.
Reports 19 and 20 treated that as a large unexplained shortfall, and report 20
spent two training runs on it.

Measured as content rather than as an index, the same outputs on the same
episodes score **{A['R']:.3f}**.

The item the network hands back sits **{A['cost']:.4f}** from the true answer.
A randomly grabbed context item would sit {A['cost_rand']:.4f} away. It is
returning a near-duplicate, and the metric was scoring that zero.

There is no recall failure on trained worlds. There is a real one on worlds it
has never seen, and identification hides that one instead.

## What is being measured, and why the old metric is wrong here

**The task.** A *world* is a generative process: items are a random linear map of
a latent code whose dimension is drawn as low as 1. Sixteen draws from a
one-dimensional world lie almost on a line. An *episode* is sixteen items from
one world — the *context* — followed by a seventeenth, the *query*, with about
half its coordinates erased. The erased ones are *hidden*, and the network
produces them. In these runs the query is always an exact copy of one of the
sixteen, so the answer is in memory and the task is to get it back.

**Identification**, the old metric. Take the output, find which of the sixteen
context items it is closest to on hidden coordinates, score 1 if that is the
item the query was copied from and 0 otherwise. Chance is {CHANCE:.3f}.

The problem is the 0. When two items sit {sq[0]['margin']:.3f} apart because the
world has one latent dimension, returning the other one is scored exactly as
badly as returning noise. That is a property of the prior, not a fault of the
network — and for a system whose purpose is fuzzy recall, returning the
near-duplicate is the correct behaviour.

**Recall quality (R)**, the metric this report uses instead. Take the item the
network returned. Measure how far it is from the true answer — call that the
*cost* of the mistake. Compare it to what a uniformly random context item would
cost. Then

    R  =  1  -  cost / cost of a random pick

R = 1 means the returned item is as good as the right one. R = 0 means no better
than guessing. It degrades smoothly, and it does not care which index came back.

**Item spacing**, and **distance to the nearest stored item**. Two items being
close is a property of the episode, so any distance has to be read against the
scale of that episode. *Spacing* is the mean distance from the query to the
sixteen context items. *Distance to the nearest stored item* is how far the
output sits from whichever context item it is closest to — small means the
output is sitting on something that is genuinely in memory, rather than floating
between items. The two are quoted as a ratio.

**Present against absent.** The same queries, scored once with the answer among
the sixteen and once with it removed. The gap is the direct test of whether the
context is used at all, and it involves no index.

All errors are squared error over hidden coordinates, divided by the error of
ignoring the input and drawing the average item, so 1.0 is the do-nothing score.

## What actually comes back

![What the network returns, drawn]({u_content})

Ten episodes, none cherry-picked: the columns are the 10th to 90th percentile of
the quantity each block is about.

The top block is worlds seen in training, the closest-rival quartile, and
**only episodes identification scored 0**. Rows two, three and four are the same
item to the eye. The returned item is {cs['cost_seen']:.3f} from the true answer
while the rivals are {cs['margin_seen']:.3f} apart. The right content came back
under the wrong name.

The bottom block is worlds never seen, and **only episodes identification got
right** — it scores {cs['id_novel_q2']:.3f} on that quartile. Row three is
visibly not row two or row four. The output sits {cs['ratio_novel']:.2f} of the
episode's own item spacing from the nearest stored item, against
{cs['ratio_seen']:.2f} in the block above. It is in the right neighbourhood
without being anything that is actually in memory.

Those are the two failure modes of the metric, in one picture.

## Identification is wrong in both directions

![The two metrics side by side]({u1})

| margin quartile | {qlab[0]} | {qlab[1]} | {qlab[2]} | {qlab[3]} |
|---|---|---|---|---|
| identification, seen worlds | {A['by_q'][0]['id_acc']:.3f} | {A['by_q'][1]['id_acc']:.3f} | {A['by_q'][2]['id_acc']:.3f} | {A['by_q'][3]['id_acc']:.3f} |
| **recall quality, seen worlds** | **{A['by_q'][0]['R']:.3f}** | **{A['by_q'][1]['R']:.3f}** | **{A['by_q'][2]['R']:.3f}** | **{A['by_q'][3]['R']:.3f}** |
| cost of the returned item | {A['by_q'][0]['cost']:.4f} | {A['by_q'][1]['cost']:.4f} | {A['by_q'][2]['cost']:.4f} | {A['by_q'][3]['cost']:.4f} |
| cost of a random item | {A['by_q'][0]['cost_rand']:.4f} | {A['by_q'][1]['cost_rand']:.4f} | {A['by_q'][2]['cost_rand']:.4f} | {A['by_q'][3]['cost_rand']:.4f} |

Where identification reads {A['by_q'][0]['id_acc']:.3f}, recall quality reads
{A['by_q'][0]['R']:.3f}. The returned item costs {A['by_q'][0]['cost']:.4f}
against {A['by_q'][0]['cost_rand']:.4f} for a random one. Nothing is broken
there.

The other direction is the one worth keeping. On novel worlds:

| margin quartile | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| identification | {nq[0]['id']:.3f} | {nq[1]['id']:.3f} | {nq[2]['id']:.3f} | {nq[3]['id']:.3f} |
| distance to nearest stored item, at episode scale | {nq[0]['ratio']:.3f} | {nq[1]['ratio']:.3f} | {nq[2]['ratio']:.3f} | {nq[3]['ratio']:.3f} |
| the same, on seen worlds | {sq[0]['ratio']:.3f} | {sq[1]['ratio']:.3f} | {sq[2]['ratio']:.3f} | {sq[3]['ratio']:.3f} |

In quartile 3 identification reads {nq[2]['id']:.3f} — near perfect — while the
output sits {nq[2]['ratio']:.3f} of the item spacing from anything in memory,
against {sq[2]['ratio']:.3f} on the trained pool. When items are far apart you
can be a long way from the right one and still be nearest it. Identification
cannot see that, and it is exactly what a fuzzy recall system must not do.

## Where the real deficit is

![Distance to the nearest stored item]({u2})

On trained worlds the output lands on a stored item and lands harder as items
separate: {sq[0]['ratio']:.3f}, {sq[1]['ratio']:.3f}, {sq[2]['ratio']:.3f},
{sq[3]['ratio']:.3f} across the quartiles. On novel worlds it does not:
{nq[0]['ratio']:.3f}, {nq[1]['ratio']:.3f}, {nq[2]['ratio']:.3f},
{nq[3]['ratio']:.3f} — two to five times further, at every margin.

That is the finding worth acting on. The network has learned to retrieve within
the worlds it was trained on and has not learned to retrieve in a world whose
structure is new. It is a transfer problem, not a capacity problem, and it was
invisible under the old metric because the old metric reads
{nq[3]['id']:.3f} there.

## The context is used

![Present against absent]({u3})

| | answer present | answer removed | ratio |
|---|---|---|---|
| {RUNS[0][0]}, {RUNS[0][2]} | {r45['present_absent']['A_present']:.4f} | {r45['present_absent']['C_absent']:.4f} | {r45['present_absent']['C_absent']/r45['present_absent']['A_present']:.1f}x |
| {RUNS[1][0]}, {RUNS[1][2]} | {pa['A_present']:.4f} | {pa['C_absent']:.4f} | {pa['C_absent']/pa['A_present']:.1f}x |
| {RUNS[2][0]}, {RUNS[2][2]} | {r54['present_absent']['A_present']:.4f} | {r54['present_absent']['C_absent']:.4f} | {r54['present_absent']['C_absent']/r54['present_absent']['A_present']:.1f}x |

Putting the answer in the context makes {MAIN} {pa['C_absent']/pa['A_present']:.1f}
times better. This is the first number a fuzzy memory system should be judged on,
it needs no index, and no report on this prior has led with it.

Read this way the capacity comparison also says something it could not say
before. Going from {RUNS[0][2]} to {RUNS[1][2]} parameters moved the distance to
the nearest stored item on seen worlds from {r45['A_seen_present']['d_ctx']:.3f}
to {A['d_ctx']:.3f}. The larger network commits to stored content where the
smaller one hedges between items. That is a real capacity effect, and
identification could not separate it from the ambiguity of the data.

## What this retracts

Reports 19 and 20 headline identification on the synthetic prior. Those headlines
describe the metric, not the network. Report 20's central puzzle —
"{A['id_acc']:.3f} where it should be near-perfect" — dissolves: measured as
content it is {A['R']:.3f}.

Report 21 is mine and it is half wrong. Its mechanics hold: identification
depends on the direction of the error rather than its size, the error leans
toward the average of the context, and the memory addresses the right slot more
often than the output names it. But it accepted recovering identification as the
goal, and its de-shrink correction recovers a number that did not need
recovering. The finding to keep from it is the diagnosis of why argmin over
near-duplicates is unstable — which is an argument for not using the metric, not
for fixing the network.

`lib/domains.py` should stop treating low latent dimension as a defect. The
`synth_k8` variant exists to raise the floor so that "recall is well-posed", and
its own comment records the price: it removes the easy-completion regime. That is
deforming the prior to satisfy a metric. Similar items are what a fuzzy memory
system exists to handle.

## What this does not establish

That R is the final metric. It rewards returning a near-duplicate, which is right
for this project and wrong for anything that needs the identity. Any system that
must distinguish two similar memories should keep identification, with its
ambiguity ceiling stated.

That the novel-world deficit is understood. It is measured here, not explained.
The transfer question — why retrieval learned on one family of worlds does not
carry to a new one — is untouched.

That the pictures generalise. They are ten episodes from one checkpoint, chosen
at fixed percentiles so they are a spread rather than a selection, but the
numbers are the evidence and the pictures are the illustration.

Anything about completion. Every number is from conditions where the answer is
present. What the network does when the thing is genuinely not in memory is a
different report.

## Sources

`results.jsonl` rows `exp45`, `exp49` and `exp54` — recall training on the
synthetic prior, the same three checkpoints reports 20 and 21 used. Recall
quality, cost, item spacing and distance to the nearest stored item are computed
by `scripts/recall_quality.py`; the drawn panels by `scripts/recall_images.py`,
which renders the raw 8 x 8 x 13 vector as an 8 x 104 heatmap rather than through
`domains.draw`, whose chess glyphs are wrong for a continuous synthetic item.
Present-against-absent numbers are read from the `final` block of each row.
Episodes are the project's standard eval draw at M=16, context type `class`,
scored on the first query. Figures generated by `scripts/gen_report_22.py`. No
training was run for this report.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_22", md))


if __name__ == "__main__":
    main()
