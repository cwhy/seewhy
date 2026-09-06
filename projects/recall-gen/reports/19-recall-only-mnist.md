# On MNIST, capacity buys retrieval and not completion

A network was trained on a synthetic prior — random low-dimensional worlds, a
fresh one every episode — and never shown an MNIST digit. It was then asked to
do two things with real MNIST: find a digit sitting in its context, and predict
one that is not there.

Trained at four sizes, those two abilities go in opposite directions.

| | finding a digit | predicting a digit |
|---|---|---|
| 12k steps, d=256 | 0.146 | 1.576 |
| 48k steps, d=256 | 0.248 | 1.584 |
| 192k steps, d=256 | 0.331 | 1.625 |
| 48k steps, d=512 | **0.394** | **1.711** |
| chance / do-nothing | 0.063 | 1.000 |

Finding nearly triples. Predicting gets slightly worse, and ends above the score
for ignoring the input entirely.

## The terms used here

**An episode.** The network is shown sixteen complete items, one per token, then
a seventeenth with part of it erased. It has to produce the erased part. For a
MNIST item the erased part is the bottom fourteen of twenty-eight rows.

**The context** is the sixteen items. **The query** is the seventeenth. Two kinds
of episode matter throughout, and they are the two columns of every figure below.
In one the query is a copy of one of the sixteen, so the answer is *present* and
the network can succeed by finding it. In the other the query is not among them,
so the answer is *absent* and the missing part has to be worked out.

**The three networks** are identical in size and shape — 14.95M numbers, four
layers, d_model 512 — and differ only in the episodes they were trained on.

- **Recall-trained**: its answer was always one of the sixteen. Copying always
  worked, so it never had to learn to predict anything.
- **Completion-trained**: its answer was never among the sixteen. Copying was
  never available, so it could only ever predict.
- **Frozen layers**: trained like the recall network, but its four mixing layers
  keep their random starting values forever. Only the input embedding and the
  output head learn — a small fraction of the total.

**The synthetic prior** is what all three were trained on instead of real data.
Each episode samples a fresh **world** — a random low-dimensional generative
model — and draws its sixteen items and its query from that one world. The
network therefore cannot memorise any particular world; it has to work out what
this world is from the sixteen items in front of it. That is the whole idea, and
it is what TabPFN does with tables.

A world is drawn in one of two modes. A **continuous** world produces items whose
coordinates take any value in [0, 1]. A **simplex** world puts exactly one active
coordinate in each group of thirteen, so its items are strictly binary. Forty per
cent of worlds are simplex. That number matters later, because a chess board in
this project *is* sixty-four groups of thirteen with one active coordinate each.

**Three bands** appear in every figure, increasing in novelty downward: items from
worlds seen during training, items from worlds never seen, and real MNIST — which
the network has never been shown in any form.

The two synthetic bands are scored on **single-world episodes**, matching how
these networks were trained: all sixteen context items come from one world, which
is what makes "infer this world" a question at all. The real MNIST band is scored
on unrelated items, because a real dataset has no world structure to respect.
Scoring the synthetic bands the second way asks a question the networks were
never trained for and reads 1.35 where their own task reads 0.41; an earlier
version of these reports did exactly that.

**Normalised error** is squared error over the erased coordinates, divided by the
error of a fixed reference so that 1.0 means "no better than that reference".
Which reference is stated on each figure. For MNIST the reference used here is the
average real MNIST item, whose raw error is 0.0713.

**Identification accuracy** asks which of the sixteen context items the network's
output most resembles, measured on the erased coordinates only — so a network
that merely copies the visible part cannot score. It is 1/16 = 0.063 at chance.
Its ceiling is 1.000 unless two context items share an erased half, in which case
even a perfect answer can lose the tie-break; the ceiling is reported where it
is not 1.000.

**Two references involve no trained network at all.** *Ridge* is a linear map from
the visible coordinates to the erased ones, fitted on the training pool and
applied blind — it never looks at the context. *Soft look-up* is a
similarity-weighted blend of the sixteen context items, which is the shape of
computation linear attention can actually perform, and is therefore the bar for
whether the context is being used at all.

Only the recall-trained network appears below. The completion-trained and
frozen networks are covered in report 17; both sit at chance on MNIST at every
size tried, which is a short story and not this one.

## What scales

![Capacity against the two abilities](https://media.tanh.xyz/seewhy/26-09-03/recall-gen_r19_capacity_v7.svg)

The left panel splits finding by novelty band. All three climb together and none
has flattened.

On worlds the network trained on, identification goes from
0.460 to 0.705. On worlds from the same prior that it
has never seen, 0.451 to 0.631. On real MNIST,
0.146 to 0.394.

Two things are worth reading off that. The first is that the middle band is the
one that says whether the prior works at all, and at the smallest size it does
not — 0.451 against chance of 0.063 is barely a signal. The
capability appears with capacity, and an earlier version of this work concluded
the prior had failed on the strength of the small model alone.

The second is the widening gap between the first two bands. At the smallest size
the network is 0.009 better on worlds it trained on
than on fresh ones; at the largest, 0.074. Bigger models
memorise more of the training worlds — and still generalise better in absolute
terms.

## What does not

The right panel is prediction: 1.576, 1.584,
1.625, 1.711, where 1.0 is the error of drawing the average
real digit. Higher is worse, so it drifts the wrong way. Sixteen times the
compute, spent two different ways, does not move it.

The x axis is compute — steps times parameters — which makes the last two points
a controlled comparison rather than two rungs of one ladder. 192 000 steps at
4.06M parameters and 48 000 at 14.95M land within 8% of the same compute, and the
wider model is higher on every finding series.

Both references sit where they sat. Ridge, a linear map fitted on the prior that
never looks at the context, scores 1.419. The soft look-up, which
does nothing but average the sixteen context images by similarity and involves no
training at all, scores 0.993 — better than every network on this
list.

So the context contains something usable for prediction, and more capacity does
not make the network any better at using it. Whatever retrieval is learning from
scale, prediction is not learning it too.

![What the largest network produces](https://media.tanh.xyz/seewhy/26-09-03/recall-gen_r19_grid_v5.svg)

## The context matters more than the network

Everything above uses sixteen unrelated images as context. Replace them with the
query's own sixteen nearest neighbours and both numbers move — in opposite
directions again.

![The same network, two kinds of context](https://media.tanh.xyz/seewhy/26-09-03/recall-gen_r19_context_v3.svg)

Identification falls from 0.394 to 0.281.
Prediction improves from 1.711 to 1.464.

This is not a paradox, and the pool diagnostics say why. A neighbour context is
assembled *from* the query's closest matches, so it is by construction a
low-margin context: the median distance from a target to its nearest rival falls
from 0.63 to 0.36. The answer becomes easier to
reconstruct and harder to name, because the sixteen candidates are now nearly the
same picture.

That is the same mechanism report 15 measured on real training data, reproduced
here on a network that has only ever seen synthetic worlds.

## What this establishes

That retrieval from a synthetic prior transfers to MNIST, partially, and scales
with capacity across every size tried — 0.146 to
0.394, with no sign of a ceiling.

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
their `_stdeval_synth_to_mnist` rows from `scripts/standard_eval.py`, which carry both
context types, the identification ceilings and the nearest-rival margins.
`baselines_synth_to_mnist_M16_r14_split` and its `_knn` counterpart hold the ridge, soft
look-up and average-item references. Completion is divided by the error of the
average real MNIST item (0.0713) rather than the average training item,
because the training pool here is synthetic and would not be a meaningful
reference for a real digit. Figures generated by `scripts/gen_report_19.py`. No
training was run for this report.
