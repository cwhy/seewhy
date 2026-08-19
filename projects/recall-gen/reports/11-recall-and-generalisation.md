# Copying costs generalisation

This report answers one question. A network is trained to fill in a missing
part of an image by looking at sixteen example images. What does it actually
learn to do, and when does that learning help it handle an image it has
never seen before?

The training signal used here rewards only one thing: reproducing an answer
that is already sitting in the sixteen examples. A network that gets very
good at that reward becomes worse, not better, at the harder case. In that
harder case the answer is not sitting there at all, and has to be worked out
instead. A version of the same network is kept from specialising in that
reward, by leaving most of its parameters frozen at their random starting
values. That frozen version ends up doing the working-out part better than
any version of the network that was allowed to train freely.

Put in numbers: a fully-trained network's best measured performance on
held-out examples where the answer must be worked out is a normalised error
of **0.505**. (0.0 is a perfect prediction. 1.0 is no better than always
guessing the average training image. So 0.505 is roughly halfway between
useless and perfect.) Left training longer, the same network's error on
those same held-out examples rises to 0.666 — it gets worse with more
training, not better. A version of the network with its context-processing
layers frozen at random initialisation reaches **0.471**. That is better
than the fully-trained network manages at its best. It is also better than a
reference computation with no learned parameters at all, which simply
averages the sixteen example images and reaches 0.552 at its best setting.
And it is better than a ridge-regression baseline that is not shown the
sixteen examples at all, which reaches 0.631. The rest of this report builds
these numbers up from the task itself, so that by the end each one is a
plain measurement rather than a claim to be taken on trust.

## The task

Every example given to the network is a short sequence. First come sixteen
images of handwritten digits from the MNIST dataset, each a 28-by-28 grid of
pixels flattened into a list of 784 numbers. Then comes a seventeenth image
— the query — with its bottom fourteen rows (392 of its 784 pixels) replaced
by a neutral grey and hidden from the network. The network's job is to
output values for those 392 hidden pixels. Its answer is scored only on
that hidden region; the visible top half of the query is given for free and
is not part of the prediction task.

![the task: 16 context images write into a fixed-size state, the query reads it, and only the greyed region is scored](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_task_diagram.svg)

The task is worth building this way because it can be set up in two
different ways that look identical from the network's point of view, but
call for opposite strategies. In one setup, the query's true, complete image
is literally one of the sixteen context images. The network has already
been shown the answer, and the correct move is to find and reproduce it. In
the other setup, the query's true image is not among the sixteen — nothing
in the context is a copy of the answer. The best a network can do is notice
which context images look similar to the visible top half of the query.
It can use them to make an informed guess about what the hidden bottom half
probably looks like. The first setup tests whether a network can look
something up. The second tests whether it can work something out from
related but non-identical examples. A single architecture, trained one way,
is asked to do both. The interesting question is whether getting good at the
first makes a network better or worse at the second.

## A worked example, from raw pixels to five guesses

Before any aggregate numbers, it helps to see one full episode end to end,
with nothing summarised away. The figure below shows a single held-out
example where the query's true image is not among its sixteen context
images. This is the harder of the two setups described above, and the one
this report cares about most. The example shown is not cherry-picked: it is
the one sitting at the midpoint (the 50th percentile) of a difficulty
ranking built without reference to any trained network. Specifically, the
ranking uses the per-example error of the blend reference computation
defined in the next section, so this example is neither an easy win nor a
worst case.

![One held-out episode, shown completely. Top: all 16 context images (the query's 16 nearest neighbours by pixel distance on its visible half; the true query image is not among them) and the query with its bottom half hidden. Bottom: the true image, then five methods' predictions for the missing 392 pixels, each with the true visible half composited back in and its own raw squared error on the hidden half printed underneath. The episode is the median-difficulty example (p50) by a model-free difficulty ranking — the blend look-up's own per-sample error — not hand-picked and not chosen by any trained network's error.](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r11_worked_episode_v3.png)

Reading left to right along the bottom row: always predicting the average
training-set image ignores the sixteen context images entirely. It produces
a uniform grey smear, with the highest error of the five methods on this
example. The near-copy look-up puts almost all its weight on whichever
single context image looks most similar to the query's visible top half.
It produces a sharp, digit-like completion, because it commits fully to one
neighbour's answer. The blend look-up spreads its weight across several
similar-looking neighbours instead, and produces a softer, more averaged
completion. The fully-trained network and the frozen-layer network each
produce their own completions. Their errors on this one example should be
read as one data point, not as the pattern. The aggregate comparisons later
in this report, over 512 such episodes, are what the rest of the argument
rests on. What this figure is for is showing, concretely, what "guessing the
hidden half of a digit from sixteen similar-looking digits" actually looks
like as pixels. That is before any of it gets reduced to a single number.

## How the network works

The network is a four-layer recurrent architecture built around
linear-attention layers of a delta-rule / KDA type. It keeps a
matrix-valued internal state that context images write into one at a time,
and that the query then reads from once, after all sixteen context images
have been processed. It has about 4.03 million parameters in total, arranged
as d_model 256 with 4 attention heads of dimension 64 each.

Each layer's state update, for every context token in turn, is:

```
S <- S * diag(alpha_t)              # per-channel forgetting
vhat = S k_t                        # what the state currently holds at this key
e = beta_t * (v_t - vhat)           # correction toward the true value
S <- S + e k_t^T                    # write
o_t = S q_t / sqrt(d_k)             # read
```

`S` is the state, one 64-by-64 matrix per head, per layer. Each context
token computes a key `k_t` and value `v_t`. It checks what the state
currently predicts for that key (`vhat`), and writes a correction toward the
token's true value, scaled by a learned gate `beta_t`. The query token never
writes — it only performs the final read, `o_t = S q_t`, once the state has
finished absorbing all sixteen context images.

The state's total size is fixed regardless of how many images were written
into it: 4 heads times 64 times 64 numbers per layer, which is 16,384
numbers per layer. That is fewer than the 16 images times 784 pixels —
12,544 numbers — that make up the raw context. So the state cannot simply be
a lossless copy of the sixteen images. Whatever it captures about them has
to be a compressed summary, not a verbatim record.

Pixel predictions are produced by a small output network reading off the
state after the query's read. It is worth being explicit about one thing
that is absent from this architecture: there is no softmax anywhere in it.
Nothing in the pipeline computes an explicit weighted average over the
sixteen context images, the way a conventional attention mechanism would.
Any tendency for the network's output to resemble one particular context
image, or to resemble a blend of several, has to be an emergent effect of
training. It is something the delta-rule state update above produces on its
own, not a mechanism built in the way it is in the reference computation
described next. This distinction matters later. Everywhere below that a
"temperature" is used to describe the network, it is a description fitted
after the fact, not something the network has any explicit control over.

## The four conditions

Every evaluation below sits in one of four conditions. These are formed by
crossing two independent choices. The first is whether the query's true
image is present in the sixteen-image context or absent from it. The second is whether the query image itself comes from the pool of images
the network's context-processing layers were exposed to during training
(seen). The alternative is that it comes from MNIST's held-out test split,
never seen during training in any role (novel). The figure below shows one
real episode from each of the four cells.

![The four conditions, drawn from one real held-out episode per cell rather than described in words. Columns: whether the query's true image is one of the 16 context images (present, left column) or not (absent, right column). Rows: whether the query is from the pool the network's context-reading layers were trained on (seen, top row) or from MNIST's held-out test split (novel, bottom row). Each cell shows 4 of that episode's 16 context images (a green border marks the one that is the query's own image, present cells only) and the query with its bottom half hidden.](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r11_conditions_matrix_v2.png)

Condition A, seen pool with the target present, is the condition the
training loss is actually computed on. The network is directly optimised to
do well here. Condition B, novel pool with the target present, asks a
different question. Does whatever lets the network solve A also work on an
image it never trained on, when the answer is still sitting in the context?
Condition C, seen pool with the target absent, asks what the network does
when the context carries no usable answer but the query itself is a
familiar kind of image. Condition D, novel pool with the target absent, is
the hardest and most important cell in this matrix. The query is
unfamiliar, and the context contains no copy of the answer, so a good score
here can only come from generalising — using the context's
similar-but-not-identical images to make a sound guess. Every headline
number in this report, including 0.505, 0.666 and 0.471, is a Condition D
number.

The figure below adds the piece those four cells describe in words but do
not yet show: what the network actually predicts, in each of the four
conditions, next to the true image.

![Completions across all four conditions, one block per cell (A top-left, B top-right, C bottom-left, D bottom-right), columns chosen at fixed percentiles (10th, 35th, 65th, 90th) of the blend look-up's own per-sample error within that condition — a model-free ranking, not file order. Rows within each block: the true image; the fully-trained network's prediction; the frozen-layer network's prediction; each predicted panel composites the true visible top half back in and is labelled with its own squared error on the hidden half.](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r11_four_condition_completions_v2.png)

The present conditions (A and B, left column of blocks) are near-exact for
the fully-trained network across every column shown: the answer is sitting
in the context, and the network reproduces it. The absent conditions (C and
D, right column of blocks) are visibly not exact for either network, and the
frozen network's completions there look softer and less committed than the
trained network's. Seen versus novel makes far less visible difference than
present versus absent does. Blocks A and B look similar to each other, and
so do blocks C and D, while A and C (or B and D) look very different. That
ordering — present/absent dominating, seen/novel barely mattering — is the
same ordering the aggregate numbers in the rest of this report describe.

## Three reference computations

Before judging any trained network, it helps to have computations that are
not trained at all. Then a network's score can be read against something
concrete, rather than in isolation. Three are used throughout.

The first is simply predicting the training-set average image for every
query, ignoring the sixteen context images entirely. This is the definition
of "no better than chance" used everywhere in this report: every error
number below is normalised so that this computation scores exactly 1.0. A
network beating 1.0 is doing something with the context that plain averaging
does not.

The second is a ridge regression that maps the query's visible top half to a
prediction of its hidden bottom half. It is fitted on the training pool but,
like the mean-image computation, never shown the sixteen context images at
evaluation time. It scores 0.631 on Condition D. Because it ignores the
context, this baseline draws a line. It shows how much of a network's
ability to predict a hidden region comes purely from knowing what digits
generally look like, independent of the sixteen context images. A network
that cannot beat 0.631 is not making effective use of its context at all.

The third is the reference computation used most throughout this report: a
plain weighted average of the sixteen context images, with no learned
parameters. The weight given to each context image depends on how close it
is, pixel by pixel, to the visible top half of the query:

```
d_i    = || visible(query) - visible(context_i) ||^2 / n_visible_pixels
w_i    = softmax(-d / tau)_i
output = sum_i  w_i * context_i
```

The single parameter `tau` (temperature) controls how concentrated this
weighting is. At a low temperature such as 0.003, almost all the weight
lands on whichever single context image is closest to the query, so the
output is nearly an exact copy of that one image. At a higher temperature
such as 0.03, the weight spreads out over several near neighbours, and the
output becomes their blend. This reference computation matters for two
separate reasons. It is itself a baseline a trained network can be compared
against. It is also the only computation here that literally contains a
temperature, so it doubles as a ruler. Given a trained network's actual
output on a batch of episodes, it is possible to ask which value of `tau`
would have made this reference computation's output closest to the
network's own output. That fitted value is what "the network's temperature"
means every time it is used below. The network itself has no such parameter
and no explicit averaging step.

## What the training objective actually rewards

Before looking at what trained networks do, it is worth asking what the
objective they are trained on would reward if it could be optimised
perfectly, with no network in the way at all. This can be measured directly
by sweeping the reference computation's temperature. It is scored on both
Condition A (target present, the actual training condition) and Condition D
(target absent, the generalisation test), with no trained model involved.

![the temperature trade-off, no network involved](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_tradeoff.svg)

Lowering `tau` from 0.03 to 0.003 moves the reference computation's weight
almost entirely onto the single nearest context image. That lowers its
error on the present condition from 0.372 to 0.013, a large gain. The gain
makes sense: a target that is literally present is best served by copying
it exactly. But the same change raises its error on the absent condition from
0.553 to 0.672, a real cost. A specific, confident, copied answer is
frequently the wrong digit when the true answer was never in the context to
copy. These two curves have opposite optimal temperatures. This is a
mathematical property of squared-error scoring applied to this data, not
something any trained network invents. Squared error is minimised, in
expectation, by the average of every plausible answer. So on the absent
condition it structurally prefers a blurred blend over a sharp, specific,
possibly-wrong guess, even though a plausible digit is a perfectly
reasonable thing to output. A confident, correctly-shaped digit that happens
to be the wrong digit is punished harder under squared error than a blur
that is wrong about everything equally. The training objective used
throughout this project is exactly this squared-error loss, computed only on
Condition A episodes. So the objective a network is trained on has an
unambiguous preference for sharp copying. It has no built-in reason to
prefer blending, because Condition D episodes never appear in the loss it
is optimising.

## What trained networks actually do

The reference computation above shows what an idealised, tau-controlled
averaging process would trade off. The next question is whether a trained
network — which has no explicit temperature and no explicit averaging
mechanism at all — moves along the same trade-off as it trains.

To answer that, each training checkpoint's output is fitted to the closest
matching temperature of the reference computation, using the same procedure
described above. Take the checkpoint's actual output on a batch of Condition
B (novel pool, target present) episodes, and find the `tau` whose reference
output is closest to it in squared error. This fitting is done on Condition
B rather than Condition D for a specific reason: the reference computation's
behaviour is only well defined when a target is actually present to copy or
not copy. Fitting a temperature to Condition D output does not give a stable
estimate the same way, because on Condition D the reference computation is
already just doing prediction, not exhibiting a clean copying/blending
trade-off.

![fitted temperature against held-out prediction error, trained and frozen checkpoints](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_tau_vs_error.svg)

Early in training, the fully-trained network's fitted temperature is 0.03,
comparable to the blend end of the reference sweep. Its Condition D error is
at 0.505, the best value measured for this network at any point in
training. As training continues, the fitted temperature keeps falling: to
0.00053 by the end of training. That means the network's own output has
become nearly indistinguishable from copying a single context image, even
on episodes where doing so is the wrong move. Over the same stretch of
training, Condition D error rises from 0.505 to 0.666. On a version of the
task where the context is sixteen images unrelated to the query, rather than
nearest neighbours, the same drift goes further still: fitted temperature
0.0017, Condition D error 0.843. The direction is consistent across every
trained checkpoint measured. As training proceeds, the network's output
comes to resemble a copy of a single context image more and more closely.
Its ability to handle episodes where the answer is genuinely absent gets
worse, not better.

The frozen-layer network gives the same measurement a very different
answer. Training only the input embedding and the output layers — 0.60
million of the network's 4.03 million parameters — leaves the four
recurrent, context-processing layers at their random initialisation.
Training can still change how a pixel is turned into a token, and how the
state's final read is turned back into pixels. But it cannot change how the
state combines information from the sixteen context tokens in the first
place. Across every checkpoint of this frozen-layer run, the fitted
temperature stays at 0.03. It never drifts toward copying a single image,
because the part of the network that would have to drift is not being
trained. Condition D error under this run falls monotonically over training,
reaching 0.471 at its best checkpoint. That is below the fully-trained
network's best value (0.505). It is also below the best the reference
computation can reach at any fixed temperature over the same context
(0.552), and below the ridge baseline that ignores context entirely (0.631).

A frozen network that never drifts toward copying could, in principle, be
doing something much simpler than actually reading its context. For
instance, it could have learned a generic prior over what digits look like,
and be ignoring the sixteen context images altogether — that too would
produce a temperature that never moves. To rule this out, a control swaps in
a different, unrelated query's sixteen nearest neighbours as the context,
while keeping the original query fixed. If the frozen network were ignoring
its context, this swap should not matter. In fact it drops the frozen
network's Condition D accuracy from 0.471 to 0.763, close to the level of
having no useful context at all. The frozen network is reading the context
it is actually given, not falling back on a generic prior independent of it.

Two things have now been established. The training objective itself rewards
sharpening toward a copy of one context image, and punishes that same
sharpening on the harder, target-absent case. A network free to train all
its layers follows exactly that gradient. A network prevented from making
the same drift never suffers the same degradation.

## What the completions actually look like

The temperature and error numbers above describe an aggregate tendency
across hundreds of held-out episodes. This section shows what that tendency
looks like in individual predictions, first on the easier setup where the
answer is present, then on the harder one where it is absent. Both figures
below use the query's sixteen nearest neighbours as context, and the same
six queries appear in both. The queries are chosen at fixed percentiles
(5th, 23rd, 41st, 59th, 77th, 95th) of a model-free difficulty ranking, not
by hand and not by any trained network's own error. Every predicted panel
composites the network's prediction for the hidden 392 pixels back onto the
query's true visible top half. This is needed because the network also
emits values for the visible pixels, which are never scored and would be
misleading to show unmodified.

![Six queries whose true image is one of the 16 unrelated context images. Rows, top to bottom: the true image; what the network is given (bottom half hidden); always predicting the mean training image; a model-free nearest-match look-up at temperature 0.03; the fully-trained network; the frozen-layer network. Each panel shows the true visible half composited with that method's predicted hidden half, labelled with its own raw squared error on the hidden half only.](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_recon_present.png)

When the answer is present in the context, the fully-trained network does
not merely approximate it. Across all six example columns its squared error
on the hidden half is 0.00. This is exactly the copying behaviour the
training objective rewards, executed essentially perfectly.

![The same six queries and row order as above, but the true image is now not in the context — the answer must be predicted rather than copied.](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r9_recon_absent.png)

When the answer is absent, the two networks visibly diverge in style. The
fully-trained network does not blur its guess toward an average digit
shape. It commits to a specific, confident, and often wrong completion.
Examples seen: a plausible-looking 9 where the true digit was something
else, a curled tail that turns a 7 into something 9-like, and a doubled
stroke that changes what a 2 looks like. The frozen-layer network's
completions are visibly softer and
less committed, and score a lower squared error at four of the six example
columns. At the single hardest column of the six, the ordering reverses and
the fully-trained network scores marginally lower error there. The look-up
row on this figure is also informative in its own right: at the two hardest
columns it scores worse than simply predicting the average image. That means
the sixteen nearest-neighbour context images, for those two particular
queries, happened to carry essentially no usable information about the true
hidden region.

## Which metric decides the winner

Everything up to this point has scored predictions with squared error. It is
worth asking whether a different, reasonable way of judging a completion's
quality would agree with that verdict. Squared error's structural preference
for blur is established above, and it means a low squared-error score is not
automatically the same thing as a completion that looks like a real,
correctly-identified digit.

Two further metrics are introduced for this purpose, and each needs its own
definition before its numbers mean anything. **Realism** measures, for a
predicted hidden half, its per-pixel distance to the single closest real
hidden half found anywhere in the training pool. It asks only "does this
look like some real digit's bottom half", never asking whether it is the
right digit. Lower is more realistic, and there is no upper bound other
than whatever the least realistic completion in a given batch happens to
score. **Digit identity** takes that same nearest real training match and
checks whether its label agrees with the query's own true label. It asks the
question realism cannot, namely whether the completion resembles the correct
digit rather than merely some digit. Its natural ceiling is set by the true
image itself: even the true, unaltered hidden half is not always closest to
a training example sharing its own label. Its closest match sometimes
belongs to a different digit that happens to look similar, so the true image
scores 0.869 rather than 1.000. That 0.869 is the honest ceiling this metric
can reach, not a network failing to hit 1.0.

Squared error is minimised by an average over plausible answers, established
earlier in this report. Realism and digit identity are not. A metric based
on nearest-neighbour distance to real images has no preference for
blending. A metric based on matching a discrete label has no preference for
hedging between several possible digits. Running the two look-up
temperatures used throughout this report through all three metrics shows the
predicted disagreement directly:

| temperature | squared error (lower better) | realism (lower better) | digit identity (higher better, ceiling 0.869) |
|---|---|---|---|
| tau=0.003 (near-copy) | 0.672 | 0.0154 | 0.805 |
| tau=0.03 (blend) | 0.553 | 0.0180 | 0.756 |

![the inversion: the two look-up temperatures on all three metrics](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_inversion.svg)

The blend wins on squared error, 0.553 against 0.672. The near-copy wins on
both of the other two metrics: realism 0.0154 against 0.0180, and digit
identity 0.805 against 0.756. This is the same underlying computation, with
one parameter changed. Its ranking against the alternative fully inverts,
depending only on which metric is used to read the result.

The same three metrics, computed on the trained and frozen networks over 512
held-out Condition D episodes, extend the table:

| | squared error | realism | digit identity (ceiling 0.869) |
|---|---|---|---|
| true image (ceiling) | 0.000 | 0.0161 | 0.869 |
| mean image (blur floor) | 1.000 | 0.0360 | 0.113 |
| look-up, tau=0.003 (near-copy) | 0.672 | 0.0154 | 0.805 |
| look-up, tau=0.03 (blend) | 0.553 | 0.0180 | 0.756 |
| trained network, best checkpoint | 0.505 | 0.0169 | 0.729 |
| trained network, end of training | 0.666 | 0.0172 | 0.758 |
| frozen-layer network, best checkpoint | 0.471 | 0.0166 | 0.744 |
| frozen-layer network, end of training | 0.474 | 0.0165 | 0.738 |

![realism against squared error, every row in the table](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_frontier.svg)

The finding that matters most in this table is this: no trained network, in
any row, beats the model-free near-copy look-up on either of the two
metrics that do not structurally reward blur. That look-up's own scores are
realism 0.0154 and digit identity 0.805.
This holds for the best and the final checkpoints of both the
fully-trained and the frozen-layer networks. That is four network rows, all
four worse on realism, all four worse on digit identity, than simply copying
the single most similar context image. On squared error alone, the frozen
network's best checkpoint (0.471) is the best result in the whole table,
including the near-copy look-up (0.672) and the blend look-up (0.553). On
realism and digit identity, a computation with no learned parameters at all
still wins.

One entry in this table looks backwards at first glance and is worth
addressing directly: the near-copy look-up scores a better realism (0.0154)
than the true image itself (0.0161). This is not a measurement error. A
copied training image is, by construction, a training image, so its
distance to the nearest training image is at or near zero. A held-out
query's own true hidden half is a real digit too, but it is drawn from
MNIST's test split, not the training pool. Its single nearest match among
training images sits, on average, slightly further away than an exact copy
would. The same effect is what keeps digit identity below a perfect 1.0
even for the true image, as already noted above.

Finally, six completions of the same query set are shown below in pixels,
this time with both squared error and realism printed under each panel. That
way the disagreement between the two metrics is visible example by example,
rather than only in the aggregate table.

![six completions of the same six queries, chosen at fixed percentiles of a model-free difficulty measure, true visible half composited back in, each labelled with its squared error (green, e) and realism (cyan, r)](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r10_completions_v2.png)

No individual panel is barred from scoring well on both measures at once —
several do. But the aggregate table above is what decides which *strategy*
wins. Averaged over the full 512-episode set, the blend look-up wins on
squared error, and the near-copy look-up wins on realism and digit
identity. That is a statement about the two strategies as a whole, not a
claim about
any single panel shown here.

## A geometric route to the same failure

Training is not the only thing that can push a computation toward
resembling a copy of one context image. The frozen-layer network's context
is normally built from the query's genuinely nearest neighbours. A separate
set of runs instead builds the context from progressively more distant
neighbours: the nearest, the 64th-nearest, the 512th-nearest, and finally
sixteen entirely unrelated images. The network's weights and its fitted
temperature (0.03 throughout, since these are all frozen-layer runs) never
change at all across this sweep.

![The dial: all four frozen runs. Left panel shows exact-match accuracy on the context item rising as the true image is deliberately placed further from its distractors; right panel shows the resulting degradation in held-out prediction accuracy over training, largest for the most separated context.](https://media.tanh.xyz/seewhy/26-08-18/recall-gen_dial.svg)

As the distractors are moved further away, exact-match identification
accuracy on which context item is the target rises steadily. It goes 0.295
at the nearest-neighbour setting, 0.527 at the 64th-nearest, 0.889 at the
512th-nearest, and 0.988 with unrelated images. This matches how the reference
computation itself would behave at a fixed temperature if only the distances
changed. When one context item sits much closer than the rest, even a
fixed-temperature weighted average ends up putting most of its weight on
that one item. That happens simply because the distances driving the
softmax have become more separated, not because the temperature has moved.
In the same order, held-out prediction accuracy degrades progressively more
over training: +0.002, +0.007, +0.094, +0.208 (each figure is the rise in
Condition D error from the run's best checkpoint to its final one). Moving
toward resembling a copy of one image, and the generalisation cost that
follows it, can therefore be produced two different ways in this project. It
can come from training, as shown earlier, or from context geometry alone,
with no weights changing at all.

## What this does not establish

**A capacity confound.** The frozen-layer network trains 0.60 million
parameters; the fully-trained network trains all 4.03 million. Every
comparison in this report between "frozen" and "fully-trained" is therefore
also a comparison between fewer and more trainable parameters. This report
cannot separate "kept from resembling a copy of one image" from "simply has
less capacity to overfit" as the explanation for the frozen network's
advantage. Settling this would need a frozen-layer network whose trainable
parameter count is matched to the fully-trained one some other way. One
option is freezing a different subset of layers of the same total size —
which has not been run.

**An unexplained residual.** One configuration does not fit the account
built up across this report: the frozen-layer network run on entirely
unrelated-image context never develops a fitted temperature away from 0.03.
By the account above, it should therefore not degrade — yet its held-out
error still rises from 0.570 to 0.778 over training, the largest degradation
of any frozen run measured. Coming to resemble a copy of one image is
evidently sufficient to cause degradation, since preventing it prevents the
degradation seen in the fully-trained network. But this one result shows it
is not the only mechanism that can cause it, and no alternative mechanism is
identified here. Settling this would need instrumenting what else changes
over training in this particular frozen run, which has not been done.

**Scope.** All of this is one dataset (MNIST), one architecture (a
four-layer delta-rule linear-attention network), and one masking pattern
(the bottom half of a 28-by-28 image). Nothing here has been checked against
a different dataset, a different recurrent architecture, or a different
missing-region shape. None of the numbers in this report should be assumed
to transfer to those settings without rerunning it.

## Sources

`results.jsonl` rows `exp1`, `exp20`, `exp24`, `exp26`, `exp27`, `exp28`,
`baselines_M16_r14`, `baselines_M16_r14_knn_Q1`, `effective_tau2`,
`ctx_ablation2`, `metrics_beyond_mse`. Temperature fitting:
`projects/recall-gen/scripts/effective_tau.py`. Metric computation:
`projects/recall-gen/scripts/metrics_beyond_mse.py`. Context construction:
`knn_offset` parameter in `projects/recall-gen/lib/train.py` and
`lib/evalsets.py`.
