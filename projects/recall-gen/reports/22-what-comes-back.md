# What comes back is right. The metric saying otherwise is the problem.

The network is shown sixteen items from one world and a seventeenth that is a
copy of one of them with half its coordinates erased. On worlds it trained on it
scores **0.756** at naming which of the sixteen the answer was.
Reports 19 and 20 treated that as a large unexplained shortfall, and report 20
spent two training runs on it.

Measured as content rather than as an index, the same outputs on the same
episodes score **0.987**.

The item the network hands back sits **0.0115** from the true answer.
A randomly grabbed context item would sit 0.8782 away. It is
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
item the query was copied from and 0 otherwise. Chance is 0.062.

The problem is the 0. When two items sit 0.013 apart because the
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

![What the network returns, drawn](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r22_content_v1.png)

Ten episodes, none cherry-picked: the columns are the 10th to 90th percentile of
the quantity each block is about.

The top block is worlds seen in training, the closest-rival quartile, and
**only episodes identification scored 0**. Rows two, three and four are the same
item to the eye. The returned item is 0.018 from the true answer
while the rivals are 0.009 apart. The right content came back
under the wrong name.

The bottom block is worlds never seen, and **only episodes identification got
right** — it scores 0.901 on that quartile. Row three is
visibly not row two or row four. The output sits 0.36 of the
episode's own item spacing from the nearest stored item, against
0.22 in the block above. It is in the right neighbourhood
without being anything that is actually in memory.

Those are the two failure modes of the metric, in one picture.

## Identification is wrong in both directions

![The two metrics side by side](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r22_two_metrics_v1.svg)

| margin quartile | 0.01 | 0.13 | 0.60 | 1.26 |
|---|---|---|---|---|
| identification, seen worlds | 0.367 | 0.664 | 0.992 | 1.000 |
| **recall quality, seen worlds** | **0.936** | **0.948** | **0.998** | **1.000** |
| cost of the returned item | 0.0125 | 0.0309 | 0.0028 | 0.0000 |
| cost of a random item | 0.1954 | 0.5936 | 1.1894 | 1.5293 |

Where identification reads 0.367, recall quality reads
0.936. The returned item costs 0.0125
against 0.1954 for a random one. Nothing is broken
there.

The other direction is the one worth keeping. On novel worlds:

| margin quartile | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| identification | 0.188 | 0.424 | 0.901 | 0.984 |
| distance to nearest stored item, at episode scale | 0.318 | 0.391 | 0.366 | 0.258 |
| the same, on seen worlds | 0.189 | 0.189 | 0.075 | 0.047 |

In quartile 3 identification reads 0.901 — near perfect — while the
output sits 0.366 of the item spacing from anything in memory,
against 0.075 on the trained pool. When items are far apart you
can be a long way from the right one and still be nearest it. Identification
cannot see that, and it is exactly what a fuzzy recall system must not do.

## Where the real deficit is

![Distance to the nearest stored item](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r22_committed_v1.svg)

On trained worlds the output lands on a stored item and lands harder as items
separate: 0.189, 0.189, 0.075,
0.047 across the quartiles. On novel worlds it does not:
0.318, 0.391, 0.366,
0.258 — two to five times further, at every margin.

That is the finding worth acting on. The network has learned to retrieve within
the worlds it was trained on and has not learned to retrieve in a world whose
structure is new. It is a transfer problem, not a capacity problem, and it was
invisible under the old metric because the old metric reads
0.984 there.

## The context is used

![Present against absent](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r22_present_absent_v1.svg)

| | answer present | answer removed | ratio |
|---|---|---|---|
| d256 4x64, 4.06M | 0.2477 | 0.4622 | 1.9x |
| d512 8x64, 14.95M | 0.0723 | 0.4106 | 5.7x |
| d512 4x128, 14.94M | 0.0677 | 0.4150 | 6.1x |

Putting the answer in the context makes exp49 5.7
times better. This is the first number a fuzzy memory system should be judged on,
it needs no index, and no report on this prior has led with it.

Read this way the capacity comparison also says something it could not say
before. Going from 4.06M to 14.95M parameters moved the distance to
the nearest stored item on seen worlds from 0.217
to 0.062. The larger network commits to stored content where the
smaller one hedges between items. That is a real capacity effect, and
identification could not separate it from the ambiguity of the data.

## What this retracts

Reports 19 and 20 headline identification on the synthetic prior. Those headlines
describe the metric, not the network. Report 20's central puzzle —
"0.756 where it should be near-perfect" — dissolves: measured as
content it is 0.987.

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
