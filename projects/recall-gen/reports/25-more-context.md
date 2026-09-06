# More context does not help a bounded memory. It costs it.

Two networks, identical but for the number of items they were trained to hold.
The delta rule writes every context item into one fixed dk x dk matrix per head:
at d_model=512 with 8 heads of dk=64 that state is 32,768 floats, and an item
is 832 numbers. Sixteen items is 13,312 floats of content, comfortably under.
Sixty-four is 53,248, comfortably over.

Each network scored at both lengths, on worlds drawn from the prior it has never
seen:

| | test M=16 | test M=64 |
|---|---|---|
| **trained at M=16** | **0.967** | 0.892 |
| **trained at M=64** | 0.858 | 0.803 |

Every direction away from the top-left cell is worse. Longer contexts cost
recall, and training on longer contexts costs more than testing on them.

The third number is the one worth pausing on. At M=64 — the length the second
network was built for — the network trained at M=16 scores
0.892 and the one trained at M=64 scores
0.803. **If you want to run a bounded memory at sixty-four
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

![Recall quality against context length](https://media.tanh.xyz/seewhy/26-09-06/recall-gen_r25_novel_v3.svg)

Test length is free to vary, so the 2x2 is four points on a curve with eight.
Both networks decline monotonically past M=16, and the short-context network is
above the long-context one at every single length.

Read `committed` the same way and it says the same thing: at M=16 the
short-context network sits 0.202 from the nearest
stored item and the long-context one sits 1.591.
A network trained on crowded contexts barely lands on a stored item even when the
context is uncrowded.

## Where more context does help

![The answer-absent condition](https://media.tanh.xyz/seewhy/26-09-06/recall-gen_r25_absent_v3.svg)

| trained at M=64, answer removed | M=8 | M=16 | M=32 | M=64 |
|---|---|---|---|---|
| normalised error | 0.857 | 0.666 | 0.477 | 0.364 |

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

![The four cells on real data](https://media.tanh.xyz/seewhy/26-09-06/recall-gen_r25_transfer_v3.svg)

| | train 16 / test 16 | train 16 / test 64 | train 64 / test 16 | train 64 / test 64 |
|---|---|---|---|---|
| MNIST | **0.480** | 0.088 | 0.170 | 0.116 |
| Fashion-MNIST | **0.357** | 0.181 | 0.233 | 0.200 |
| chess | **0.949** | 0.491 | 0.335 | 0.202 |

This time the two agree on order, on all three datasets. They disagree on size.
The worst synthetic cell keeps 83% of the best
cell's recall quality; on MNIST it keeps
24%
and on chess 21%.

The synthetic band understates this by a factor of four to five. Anything ranked
on it alone should be re-checked on real data before it is believed — which is
the second time in two days that has been true.

## What comes back

![Both networks, read at the same length](https://media.tanh.xyz/seewhy/26-09-06/recall-gen_r25_content_v3.png)

Rows three and four are the two networks answering the same episodes. The first
two blocks are read at M=64 so the comparison is at one length; the third is the
same MNIST episodes read at M=16, where both do better and the gap narrows.

## What this changes

A bounded memory has an operating length, and it is shorter than the state
budget suggests. Content passes 32,768 floats somewhere between M=32 and
M=64, but recall on novel worlds is already falling by M=32
(0.944 against 0.967). The degradation
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

![Novel-worlds error through training, both cells at both budgets](https://media.tanh.xyz/seewhy/26-09-06/recall-gen_r25_convergence_v3.svg)

| | best on novel worlds | at step | final | change over last quarter |
|---|---|---|---|---|
| M=16, 48000 steps | 0.2316 | 48,000 | 0.2316 | -3.6% |
| M=16, 96000 steps | **0.1797** | 94,000 | 0.1801 | -2.0% |
| M=64, 48000 steps | 0.5018 | 42,000 | 0.5041 | +0.3% |
| M=64, 96000 steps | 0.5027 | 42,000 | 0.5327 | -1.0% |

The M=64 arm is converged and the M=16 arm is not.

Two independent runs of the M=64 cell reach their best at **step
42,000** — the same step — at 0.5018 and
0.5027. Those two numbers differ by
0.0009. The extra
48,000 steps bought nothing at all, and
training past 42000 makes it worse: the 96000-step run ends at
0.5327, above where it started plateauing.

The M=16 arm, over the same doubling, went 0.2316 to
0.1797 — a 22%
improvement — and is still descending at 96000, with its best at step
94,000 of 96,000.

So the honest conclusion is stronger than the one the 48000-step pair supported,
not weaker. At matched budget the gap grew from
2.2x to
2.8x, and the arm that has room left
to improve is the short-context one. The earlier numbers understated this.

Every figure and table above uses the 96000-step pair for this reason.

## What this does not establish

That the state size is the mechanism. M=64 exceeds 32,768 floats of state and
recall falls, which is consistent with saturation, but state size was not varied
here. A dk sweep at fixed M would separate "the store is full" from "long
contexts are harder to learn from".

Whether attention behaves the same way. Its state grows with the number of items,
so it should not pay the interference cost — untested, and the obvious next run.

Comparability with the earlier delta-rule numbers. Both cells use batch 128,
forced by the M=64 run: the scan keeps a `(batch, heads, dk, dk)` carry per token
for the backward pass and 256 asks for 21.53 GiB on a 24 GB card. exp58 lands at
0.0636 on trained worlds against
exp49's 0.0723 at batch 256, so this 2x2 sits below the level reported elsewhere.
It is internally consistent; it is not a continuation of the earlier series.

`committed` at M=8 is high for both networks
(0.228 and 2.330).
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
2476s, exp59 7962s.
