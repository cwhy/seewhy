# Attention roughly doubles recall on real data, with fewer parameters

Two networks, identical in width, depth, heads, steps, learning rate, seed and
training episodes. They differ in one thing: how the sixteen context items reach
the query. One compresses them into a fixed matrix with the delta rule. The other
attends to them.

Trained on the synthetic prior alone and then asked to recall from datasets
neither has seen:

| recalling from | delta rule | attention | |
|---|---|---|---|
| synthetic worlds it never saw | 0.946 | **0.980** | |
| MNIST | 0.405 | **0.766** | +89% |
| Fashion-MNIST | 0.342 | **0.743** | +117% |
| chess | 0.898 | **0.969** | |

Attention did this with **13.88M parameters against
14.95M** — 7%
fewer — and trained in **454 seconds against 2715**.

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
sixteen items of 832 numbers are compressed into
32,768 floats. Attention keeps every item and picks among them
with a softmax, which can be arbitrarily sharp. Its cost is quadratic in sequence
length, which is irrelevant at seventeen tokens.

The two are held to the same information channel. Under the delta rule query
tokens never write, so nothing can read them; under attention they are masked out
as keys. Causal masking is kept because a scan cannot see the future.

## What comes back

![Every pool, both architectures](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r23_allpools_v1.png)

Five pools, five episodes each, both networks answering the same episodes. The
columns are the 10th to 90th percentile of how far the **delta rule's** output
sits from the nearest stored item, so the episodes are chosen by the baseline
rather than by the comparison.

Read the top block against the bottom three. On worlds it trained on, both rows
of output are the stored item. On MNIST and Fashion-MNIST neither is, and
attention's row is the closer of the two without being the answer.

## Recall quality across four pools

![Recall quality](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r23_quality_v1.svg)

The synthetic column is worth reading first: 0.946 to 0.980 on
worlds drawn from the same prior the network trained on but never seen. That gap
is small because the delta rule was already close to solving it. The real-data
columns are where the mixers separate.

Chess moves least in relative terms, from 0.898 to 0.969, and
the reason is in the prior: 40% of the synthetic worlds are drawn in simplex mode,
one active coordinate per group of thirteen, which is exactly a chess piece plane.
Chess sits nearly inside the prior's support. MNIST and Fashion-MNIST do not, and
they are where the change is worth 89% and
117%.

## The context is worth more to attention

![Present against absent](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r23_present_absent_v1.svg)

| | answer present | answer removed | ratio |
|---|---|---|---|
| delta rule | 0.0723 | 0.4106 | 5.7x |
| attention | 0.0306 | 0.5642 | **18.4x** |

Both numbers move in the right direction: attention is better with the answer
present and *worse* with it absent. That is the signature of a network leaning
harder on retrieval and less on a learned prior, which is what a memory system
should do.

## What attention does not fix

![Distance to the nearest stored item](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r23_committed_v1.svg)

| distance to nearest stored item, at episode scale | delta rule | attention |
|---|---|---|
| synthetic, worlds seen in training | 0.125 | **0.079** |
| synthetic, worlds never seen | 0.333 | 0.291 |
| MNIST | 0.672 | 0.578 |
| Fashion-MNIST | 0.872 | 0.798 |
| chess | 0.799 | 0.618 |

Every pool improves and none of them closes. Against
0.079 on the worlds it trained on, attention still reads
0.578 on MNIST and 0.798 on Fashion-MNIST —
five to ten times further from anything actually in memory.

So the compression was a real limiter and it was not the only one. On a genuinely
new distribution both networks produce something in the right neighbourhood
rather than the stored item itself. Sharper addressing raised the quality of the
neighbourhood; it did not make the network commit.

Two suspects remain, and they are the ones this experiment was designed to
isolate down to. A single `W_pix` maps 832 raw numbers to the
embedding before any mixing, so items differing in ways the synthetic prior never
varied can be collapsed before a mixer ever sees them — which would look exactly
like this. And squared error still pays for hedging between candidates, which is
what `committed` measures; attention made that hedge cheaper to avoid without
making it unrewarded.

## What this changes

The architecture question that reports 19 and 20 answered with capacity has a
much larger answer in the mixer. Nothing in the capacity sweep approached
89% on transfer, and this run cost
8 minutes.

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
7% fewer parameters, so the gap
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
