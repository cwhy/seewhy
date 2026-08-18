# Which completion is "better" depends on which metric is asked, and the two reasonable metrics disagree

Squared error is minimised by the conditional mean, so it rewards a blurred
average over plausible completions of a missing image region. A metric that
asks whether the completion looks like a real digit, or like the *right*
digit, rewards a sharp copy of one plausible context image instead. These are
not two qualities of one good solution — they are opposite ends of a single
knob, and the objective a model is scored on decides which end it lands at.
The clearest demonstration needs no trained model at all: a weighted average
over 16 candidate images, with the averaging weight set by one temperature,
inverts its own ranking depending only on which metric reads the result.

| temperature | squared error (lower better) | realism (lower better) | digit identity (higher better) |
|---|---|---|---|
| tau=0.003 (near-copy) | 0.672 | 0.0154 | 0.805 |
| tau=0.03 (blend) | 0.553 | 0.0180 | 0.756 |

The blend wins on squared error (0.553 vs 0.672); the near-copy wins on
realism (0.0154 vs 0.0180) and digit identity (0.805 vs 0.756). Same
computation, one parameter, and the ranking inverts.

## Setup

Each example is a sequence of 16 MNIST images, 28x28 pixels flattened to 784
(the context), followed by a query image with its bottom 14 rows hidden (392
of 784 pixels scored). The task is to fill in the hidden half; the context is
built from the query's 16 nearest neighbours by pixel distance on the visible
half, and results below are restricted to the case where the query's true
image is *not* one of the 16 — the only case where "completion quality" means
anything, since when the answer is present the best move is simply to find it.

![the task: 16 context images write into a fixed-size state, the query reads it, and only the greyed region is scored](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_task_diagram.svg)

Two computations are scored throughout. One is a plain weighted average of the
16 context images — no learned parameters — with a temperature `tau`
controlling how concentrated the weighting is:

```
d_i    = || visible(query) - visible(context_i) ||^2 / n_visible_pixels
w_i    = softmax(-d / tau)_i
output = sum_i  w_i * context_i
```

At low temperature (tau=0.003) almost all the weight lands on the single
closest-matching context image, so the output is nearly a copy of it. At
higher temperature (tau=0.03) the weight spreads across several near
neighbours and the output is their blend. The other is a trained recurrent
network (4-layer linear-attention / delta-rule architecture, 4.03M
parameters) that writes the 16 context images into a fixed-size state and
reads a prediction from it after the query is presented, either left free to
train throughout, or trained only in its input and output layers with the
recurrent layers frozen at random initialisation.

Four metrics are computed on the same 512 held-out episodes:

- **squared error** — masked MSE over the 392 hidden pixels, divided by the
  masked MSE of always predicting the training-set mean image, so 1.0 = no
  better than the average digit and 0.0 = perfect.
- **realism** — per-pixel distance from the predicted hidden half to the
  *nearest real hidden half* in the training pool. No classifier, nothing
  tuned on the models being judged: it asks "does this look like some real
  digit's bottom half", not "is it the right one".
- **digit identity** — the label of that nearest real training image,
  compared against the query's true label. Answers the question realism
  cannot: whether the completion resembles the right digit, not just any
  digit.
- **classifier** — an independently trained MLP (97.99% held-out accuracy)
  classifying the composited completion (true visible half + predicted hidden
  half) against the true label.

## Full comparison, 512 held-out episodes

| | squared error | realism | digit identity | classifier |
|---|---|---|---|---|
| true image (ceiling) | 0.000 | 0.0161 | 0.869 | 0.988 |
| mean image (blur floor) | 1.000 | 0.0360 | 0.113 | 0.641 |
| look-up, tau=0.003 (near-copy) | 0.672 | 0.0154 | 0.805 | 0.912 |
| look-up, tau=0.03 (blend) | 0.553 | 0.0180 | 0.756 | 0.918 |
| trained network, best checkpoint | 0.505 | 0.0169 | 0.729 | 0.930 |
| trained network, end of training | 0.666 | 0.0172 | 0.758 | 0.908 |
| frozen-layer network, best checkpoint | 0.471 | 0.0166 | 0.744 | 0.930 |
| frozen-layer network, end of training | 0.474 | 0.0165 | 0.738 | 0.928 |

![the inversion: the two look-up temperatures on all three metrics](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_inversion.svg)

Three things follow from this table.

**The metric decides the winner.** The two look-up rows swap places between
squared error and the other two columns, as already shown above.

**No trained network beats the model-free near-copy on realism or digit
identity.** Every trained-network row scores worse realism than 0.0154 and
worse digit identity than 0.805 — and 0.805 is close to the 0.869 ceiling set
by the true image itself. On the metrics that do not reward blur, the best
strategy measured here is "copy the nearest context image", and every network
in this table underperforms it. This holds across all four network rows, best
and final checkpoints of both the trained and the frozen-layer models — it is
not one unlucky checkpoint.

**The frozen-layer network still beats the fully-trained one.** Comparing
best checkpoints, frozen beats trained on squared error (0.471 vs 0.505),
realism (0.0166 vs 0.0169) and the classifier (0.930 vs 0.930, tied);
comparing end-of-training checkpoints, frozen beats trained on squared error
(0.474 vs 0.666), realism (0.0165 vs 0.0172) and the classifier (0.928 vs
0.908), losing narrowly only on digit identity (0.738 vs 0.758). So that
comparison is not an artefact of which metric is used.

![realism against squared error, every row in the table](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_frontier.svg)

Two entries need a second look because they run against the intuitive
ordering. The near-copy look-up scores *better* realism (0.0154) than the
true image itself (0.0161) — not a measurement error: a copied training image
is, by construction, close to a training image, whereas a held-out query
image's own nearest training neighbour sits slightly further away on average.
The same effect caps digit identity below 1.0 even for the true image
(0.869): the nearest training image by hidden-half distance sometimes carries
a different label than the query, so "the closest match" and "the same digit"
are not always the same training image.

## What the classifier column does not add

The classifier spans a narrow range, 0.908 to 0.930, across every network
row, and this narrowness is not model quality — it is the instrument. The
true visible half already carries most of the class signal on its own: even
the mean-image row, with no information about the hidden half at all, scores
0.641. Realism and digit identity, which need no classifier and are not
trained on anything being judged, are the informative columns here; the
classifier corroborates at best.

## What this looks like in pixels

![six completions of the same six queries, chosen at fixed percentiles of a model-free difficulty measure, true visible half composited back in, each labelled with its squared error (green, e) and realism (cyan, r)](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_completions_v2.png)

Columns are six queries at fixed percentiles (p5 through p95) of the blend
look-up's own per-sample error — a model-free difficulty ordering, so no
network's own error picked which examples are shown. Rows: the true image;
the mean image; the near-copy look-up; the blend look-up; the fully-trained
network at the end of training; the frozen-layer network at its best
checkpoint. No single completion is barred from scoring well on both labels —
several panels do — but the two metrics rank the *strategies* differently:
averaged over the full 512-episode set (the table above), blend wins squared
error and near-copy wins realism and digit identity, and that aggregate
disagreement is what the per-panel labels here are showing example by
example, not a per-image trade-off.

## What this changes

A single-number comparison of completion quality on this task is not
meaningful without saying which metric produced it — the ranking of two
otherwise identical computations inverts depending on whether squared error
or realism/digit-identity is read. The temperature of the look-up computation
is better understood as trading between two real objectives than as a knob
toward a generically "better" model, and the same applies to any property of
a trained network's output that moves it toward or away from a copy of one
context image. Nothing here says which of these metrics should be preferred
for this task — that depends on what the completion is being produced for,
and is not answered by the data in this report.

## Sources

`results.jsonl` rows `metrics_beyond_mse`, `baselines_M16_r14_knn_Q1`,
`exp20`, `exp24`. Metric computation: `projects/recall-gen/scripts/metrics_beyond_mse.py`.
