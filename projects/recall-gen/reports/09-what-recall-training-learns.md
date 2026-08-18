# Generalisation appears only where the network is kept from resembling a copy of its single closest match

Two computations appear throughout this report. One is a trained recurrent
network (architecture below). The other is a **reference computation** with no
learned parameters at all — a plain weighted average of the 16 context images,
weight set by a temperature — used both as a baseline score and as a ruler for
describing the network. Only the reference computation literally contains a
temperature: at low temperature almost all its weight lands on the single
closest-matching context image, so its output is a copy of that image; at high
temperature its weight spreads across several near neighbours and its output
is their blend. The network has no such mechanism built into it and so has no
temperature of its own — but it can be *described* by one: take a checkpoint's
own output and ask which temperature of the reference computation would have
produced something closest to it. That fitted value is what "the network's
temperature" means everywhere below.

Left free to train, the fitted value describing the network keeps falling —
its output increasingly resembles a copy of the single closest-matching
context image — because the training signal only ever rewards exact copying,
never blending. The cost is generalisation: on held-out predictions where the
answer is not in the context, the fully-trained network's best result is
**0.505** normalised error (0.0 would be perfect, 1.0 is no better than
predicting the average image), degrading to 0.666 by the end of training as
its output moves further toward a copy of one image. A version of the same
network with its recurrent layers left at random initialisation — only the
input and output layers trained — never comes to resemble a copy of a single
image, and reaches **0.471**: better than the fully-trained network ever gets,
better than the best the reference computation can do at any fixed temperature
over the same context (0.552), and better than a ridge regression baseline
that ignores the context entirely (0.631).

## Setup

Each example is a sequence of 16 MNIST images, 28x28 pixels flattened to 784
(the context), followed by a query image with its bottom 14 rows hidden (392
of 784 pixels scored). The network predicts the missing pixels.

![the task: 16 context images write into a fixed-size state, the query reads it, and only the greyed region is scored](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_task_diagram.svg)

The network is a 4-layer linear-attention recurrent network (a delta-rule /
KDA-style layer), d_model 256, 4 heads of dimension 64, about 4.03M parameters
total. Context tokens write into a matrix-valued state per layer, one 64x64
matrix per head; the query token never writes, and reads only the finished
state after all 16 context tokens have written to it:

```
S <- S * diag(alpha_t)              # per-channel forgetting
vhat = S k_t                        # what the state currently holds at this key
e = beta_t * (v_t - vhat)           # correction toward the true value
S <- S + e k_t^T                    # write
o_t = S q_t / sqrt(d_k)             # read
```

The state holds 4 x 64 x 64 = 16,384 numbers, regardless of how many images
were written into it — noticeably fewer than the 16 x 784 = 12,544 numbers of
raw context content, so the state is doing compression, not storing the
context verbatim. Pixel predictions come out of an MLP head reading the
state's output; nothing in this pipeline normalises a distribution over the 16
context images.

Error is masked MSE, normalised by the masked MSE of always predicting the
training-set average image:

```
score = mean((pred - truth)^2 over hidden pixels)
       / mean((train_mean_image - truth)^2 over hidden pixels)
```

Two things vary in what follows: whether the query's true image happens to be
one of the 16 context images (the answer is present — exact copying is
possible) or not (the answer is absent — it must be predicted from
similar-looking neighbours); and how the context is built, either 16 images
unrelated to the query or the query's 16 nearest neighbours by pixel distance
on the visible half.

When the answer is present, the fully-trained network does not approximate
it, it reproduces it: on six example queries (chosen at fixed percentiles of
a model-free difficulty measure, not by hand), its squared error on the
hidden half is 0.00 in every one.

![Six queries whose true image is one of the 16 unrelated context images. Rows, top to bottom: the true image; what the network is given (bottom half hidden); always predicting the mean training image; a model-free nearest-match look-up at temperature 0.03; the fully-trained network; the frozen-layer network. Each panel shows the true visible half composited with that method's predicted hidden half, labelled with its own raw squared error on the hidden half only.](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_recon_present.png)

The reference computation, used as both a baseline and a ruler, is:

```
d_i    = || visible(query) - visible(context_i) ||^2 / n_visible_pixels
w_i    = softmax(-d / tau)_i
output = sum_i  w_i * context_i
```

and the fitted temperature used to describe a network checkpoint is the tau
that makes this reference computation's output closest to that checkpoint's
own output on the same batch:

```
tau*  =  argmin_tau   masked_mse( reference_output(context, query, tau),
                                   network_output(context, query) )
```

## The trade-off exists in the reference computation alone, with no network involved

Sweep the reference computation's temperature with no network in the loop. On
nearest-neighbour contexts, moving its weight almost entirely onto the single
nearest neighbour (tau=0.03 down to tau=0.003) lowers copying error from 0.372
to 0.013 (a gain of 0.359) but raises prediction error from 0.553 to 0.672 (a
cost of 0.119). The two objectives have opposite optimal temperatures. This is
a property of the task and the data, not of anything a network learns.

![the temperature trade-off, no network involved](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_tradeoff.svg)

## Trained networks come to resemble a copy of one image, and held-out accuracy falls as they do

For each training checkpoint, the fitted temperature — which value of tau
makes the reference computation's output closest to the network's own output,
defined above — falls as training proceeds. On nearest-neighbour contexts it
starts at 0.03 early in training, where held-out prediction error is at its
best value measured for this network, 0.505, and falls to 0.00053 by the end
of training — meaning the network's own output has become nearly
indistinguishable from a copy of a single context image — where the same
error has risen to 0.666. On unrelated-image contexts the network's output
moves even closer to a copy of one image, fitted temperature 0.0017, and
prediction error rises further still, to 0.843.

![fitted temperature against held-out prediction error, trained and frozen checkpoints](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_tau_vs_error.svg)

## Preventing that resemblance prevents the degradation

Training only the input embedding and output layers (0.60M of 4.03M
parameters) and leaving the four recurrent layers at random initialisation
means training can change how a token is embedded and how the state's output
is read out into pixels, but not how the state combines context tokens in the
first place. The fitted temperature describing this network stays at 0.03 for
the entire run — its output stays close to what the reference computation
would blend from several context images, never drifting toward a copy of one.
Held-out prediction error falls monotonically to 0.471 — the best number in
this comparison, ahead of the fully-trained network's best checkpoint (0.505),
the best the reference computation can do at any fixed temperature over the
same context (0.552), and a ridge regression baseline with no context at all
(0.631). A control that swaps in a different query's neighbours drops this
network's accuracy to 0.763, so the result is not simply memorising a prior
over digit shapes — the frozen network is reading the context it is given.

The same contrast, shown rather than scored, on the harder unrelated-image
context: with the answer absent, the fully-trained network does not blur its
guess toward the mean, it commits to a specific, confident, wrong completion
— a plausible 9 at p23, a curled tail turning a 7 into something 9-like at
p41, a doubled stroke on a 2 at p95. The frozen network is visibly blurrier
and scores lower squared error at p5, p23 and p41 (0.04 vs 0.05, 0.04 vs
0.06, 0.04 vs 0.05); at p95, the hardest of the six columns, the ordering
reverses (0.11 against 0.10). The look-up row shows what this context is
worth: smeared, overlapping strokes, scoring worse than the mean image at
the two hardest columns — a context carrying no information about an absent
target.

![The same six queries and row order as above, but the true image is now not in the context — the answer must be predicted rather than copied.](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_recon_absent.png)

## A second way to make the network resemble a copy of one image, with no training at all

The same frozen network, given contexts built from progressively more distant
neighbours (the nearest neighbour, the 64th-nearest, the 512th-nearest, then
unrelated images), moves through exact-match accuracy 0.295, 0.527, 0.889,
0.988 — even though its fitted temperature never changes from 0.03. Held-out
prediction accuracy degrades over training more at each step, in the same
order: +0.002, +0.007, +0.094, +0.208. This matches how the reference
computation itself would behave if held at a fixed temperature while only the
distances changed: when one context item sits much closer than the rest, even
a fixed-temperature weighted average puts most of its weight on that one item.
Geometry alone reproduces the effect of a lower temperature, with nothing
inside the network having to change. Resemblance to a copy of one image has
two independent causes here — the training-driven one above, and this purely
geometric one — and either is enough to trade prediction for copying.

![exact-match accuracy and prediction degradation as distractor distance grows](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_geometry.svg)

## What this does not establish

The frozen network trains 0.60M parameters against the fully-trained
network's 4.03M, so this comparison confounds "kept from resembling a copy of
one image" with "fewer trainable parameters" — a frozen network matched in
trainable capacity has not been run. One configuration also does not fit the
account above: the frozen network on unrelated-image contexts never comes to
resemble a copy of one image (fitted temperature stays at 0.03) yet its
held-out accuracy still degrades over training, 0.570 to 0.778 — the largest
degradation of any frozen run measured. Coming to resemble a copy of one image
is sufficient to cause the degradation seen in the trained networks, but it is
evidently not the only mechanism, and the second one is not identified here.
All of this is one dataset (MNIST), one architecture, one masking pattern.

## Sources

`results.jsonl` rows `exp1`, `exp20`, `exp24`, `exp26`, `exp27`, `exp28`,
`baselines_M16_r14_knn_Q1`, `effective_tau2`, `ctx_ablation2`. Temperature
fitting: `projects/recall-gen/scripts/effective_tau.py`. Context construction:
`knn_offset` parameter in `projects/recall-gen/lib/train.py` and
`lib/evalsets.py`.
