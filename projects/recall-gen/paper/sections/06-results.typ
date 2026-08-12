#import "/template.typ": *

// OBLIGATIONS
//  - Every metric carries its chance level or baseline IN THE SAME ROW.
//  - Seed variance, or an explicit statement that a number is a single run.
//  - One figure per claim. Report what happened; interpretation is §7.

= Results

Throughout, *nMSE* is the model's mean squared error on hidden pixels divided by
that of predicting the average training image, so 1.0 is "no better than
ignoring the input". A fifth column, *gain*, appears in most tables:

$ "gain" = "nMSE"(D) - "nMSE"(B) $

Conditions B and D both use context images the model has never seen and differ
only in whether the query's true image is among them, so gain is exactly what
having the answer in the context is worth. A model that retrieves scores high; a
model that ignores its context scores zero.

Gain exists because the obvious metric does not work. Identification accuracy —
does the output land on the *correct* one of the $M$ context images, measured on
hidden pixels — is inflated by models that never retrieve at all: a good
completion resembles the true image, so it picks the right neighbour on its own.
The completion-trained model scores 0.951 identification at $M = 4$ while its
gain is $-0.015$. Identification accuracy also has a chance level of $1 slash M$,
which moves by a factor of 64 across the context sizes used here, so it cannot be
compared down a column. Gain has neither problem, and every claim in this section
about *whether a model retrieves* rests on it.

== Reference points

No model is involved in these. They are computed on the same evaluation episodes
as every run.

#align(center, table(
  columns: 6, stroke: none, align: (left, right, right, right, right, right),
  table.hline(stroke: rule),
  [*strategy, on condition D*], [$M=4$], [$M=16$], [$M=64$], [$M=256$], [split],
  table.hline(stroke: rule),
  [mean image (the normaliser)], [1.000], [1.000], [1.000], [1.000], [1.000],
  [ridge, context ignored],      [0.625], [0.645], [0.636], [0.649], [0.851],
  [nearest neighbour in context],[1.770], [1.575], [1.384], [1.249], [1.475],
  [best soft look-up],           [1.237], [1.002], [0.886], [0.786], [0.933],
  table.hline(stroke: rule),
))

Two facts here shape everything that follows. First, at $M = 16$ the best
possible soft look-up scores *1.002* — indistinguishable from ignoring the
context entirely, and the temperature the sweep selects is the one that
flattens the weights into a uniform average. On an absent-target episode with
sixteen random digits there is nothing in the context worth extracting. Second,
hard nearest-neighbour is far *worse* than the trivial prior at every $M$:
copying one neighbour is a bad strategy, not a weak one.

== Retrieval generalises to unseen images, and to unseen classes

Recall-trained, $M = 16$, three seeds (exp1, exp10, exp11):

#align(center, table(
  columns: 5, stroke: none, align: (left, right, right, right, right),
  table.hline(stroke: rule),
  [], [A seen ctx], [B novel ctx], [id. acc. A], [id. acc. B],
  table.hline(stroke: rule),
  [seed 0], [0.015], [0.017], [1.000], [1.000],
  [seed 1], [0.016], [0.018], [1.000], [1.000],
  [seed 2], [0.016], [0.018], [1.000], [1.000],
  [chance / baseline], [1.000], [1.000], [0.063], [0.063],
  table.hline(stroke: rule),
))

Images the model has never seen are retrieved as well as training images, at
identification accuracy 1.000 against a chance level of 0.063.

Under the digit split (exp8, single run), where the training pool contains only
digits 0–4 and the novel pool only 5–9, retrieval survives the class change:

#align(center, table(
  columns: 4, stroke: none, align: (left, right, right, left),
  table.hline(stroke: rule),
  [*condition*], [*nMSE*], [*id. acc.*], [*what is novel*],
  table.hline(stroke: rule),
  [A], [0.014], [1.000], [nothing],
  [E], [0.020], [1.000], [the images],
  [B], [0.043], [1.000], [the images *and* the digit classes],
  table.hline(stroke: rule),
))

Identification is perfect on digits the model has never encountered in any role.

== Retrieval training does not produce completion, and degrades it

The same runs, on the conditions whose target is absent:

#align(center, table(
  columns: 5, stroke: none, align: (left, right, right, right, right),
  table.hline(stroke: rule),
  [], [step 500], [step 2 000], [step 6 000], [step 12 000],
  table.hline(stroke: rule),
  [A/B, target present], [0.380], [0.063], [0.023], [0.015],
  [C/D, target absent],  [0.635], [0.803], [0.848], [0.852],
  table.hline(stroke: rule),
))

The two move in opposite directions for the whole of training. Final C/D of
0.851/0.852 (0.845, 0.827 at seeds 1 and 2) is better than every pure-context
strategy in the reference table but worse than ridge regression at 0.645, which
uses no context at all.

#fig(include "/figures/divergence.typ", caption: [
  Recall-trained model, $M = 16$. The conditions whose answer is in the context
  fall to near zero; the conditions whose answer is absent climb towards the
  mean-image reference line as training proceeds.
])

== The two abilities do not compete

exp3 trains on a 50/50 mixture of the two objectives:

#align(center, table(
  columns: 4, stroke: none, align: (left, right, right, right),
  table.hline(stroke: rule),
  [*trained on*], [*gain*], [*best D*], [*final B*],
  table.hline(stroke: rule),
  [recall only (exp1)],     [0.835], [0.635], [0.017],
  [mixed (exp3)],           [0.134], [0.459], [0.484],
  [completion only (exp2)], [-0.002],[0.458], [0.672],
  table.hline(stroke: rule),
))

The mixture reaches the completion ceiling exactly (0.459 against 0.458) while
retaining a gain an order of magnitude above the completion-trained models.

== Generalisation appears exactly when retrieval fails

Recall training, state fixed at 16 384 numbers, one seed per row (the $M = 16$
row is the mean of three; spread is given in the retrieval table above):

#align(center, table(
  columns: 6, stroke: none, align: (right, right, right, right, right, right),
  table.hline(stroke: rule),
  [$M$], [context content], [*gain*], [final D], [best D], [look-up ceiling],
  table.hline(stroke: rule),
  [4],   [3 136],   [0.840], [0.854], [0.777], [1.237],
  [16],  [12 544],  [0.835], [0.852], [0.635], [1.002],
  [64],  [50 176],  [0.622], [0.658], [0.535], [0.886],
  [256], [200 704], [0.004], [0.561], [0.443], [0.786],
  table.hline(stroke: rule),
))

At $M = 256$ the gain is 0.004. For comparison, the completion-trained runs —
which never retrieve, by construction — score −0.002 (exp2), 0.006 (exp7) and
0.010 (exp12). The recall-trained model at $M = 256$ is indistinguishable from
them. Its best D of 0.443 sits on their ceiling (0.458, 0.450, 0.458), its
condition A equals its condition C (0.128 against 0.134), and its B equals its D
(0.556 against 0.561).

#fig(include "/figures/context_size.typ", caption: [
  Completion quality on novel images with the target absent, against context
  size, for the recall-trained model and for the best soft look-up from the same
  context. The model is below the ceiling at every $M$ and falls further below
  it as $M$ grows.
])

== The control: shrink the memory, hold the context fixed

The $M$-sweep changes two things at once. This sweep changes one: $M$ stays at
16, the model width stays at 256, the parameter count is unchanged, and only the
shape of the memory moves. The evaluation episodes are literally identical
across these four runs.

#align(center, table(
  columns: 6, stroke: none, align: (right, right, right, right, right, right),
  table.hline(stroke: rule),
  [state], [content \/ state], [A], [id. acc. A], [*gain*], [D],
  table.hline(stroke: rule),
  [16 384 (exp1)], [0.8], [0.015], [1.000], [0.835], [0.852],
  [8 192 (exp15)], [1.5], [0.017], [1.000], [0.800], [0.819],
  [4 096 (exp16)], [3.1], [0.022], [1.000], [0.730], [0.755],
  [2 048 (exp17)], [6.1], [0.031], [1.000], [0.646], [0.681],
  table.hline(stroke: rule),
))

Shrinking the memory eightfold, with the context held fixed, moves retrieval and
completion in opposite directions monotonically: gain falls 0.835 #sym.arrow
0.646 and D improves 0.852 #sym.arrow 0.681. The context never changed, so the
improvement cannot be information the context supplied.

Two honest qualifications. The effect is smaller than in the $M$-sweep — at the
smallest state, retrieval is still excellent (identification accuracy 1.000) and
gain is still 0.646, nowhere near the 0.004 collapse at $M = 256$. And matching
on compression ratio alone does not predict the outcome: at a ratio of about 3,
exp16 reaches D = 0.755 while exp4 reaches 0.658, so the *number of items to
tell apart* matters on top of the raw budget. Both are capacity, not
information, but they are not the same knob.

Also note that the smallest state is reached by shrinking the per-head key
dimension to 8, so exp17 confounds memory size with key dimensionality.

#fig(include "/figures/state_size.typ", caption: [
  Retrieval and completion against the size of the recurrent state, with the
  context held at 16 images throughout. The two move in opposite directions on
  episodes that never change.
])

== It is a training-time effect, not an inference-time one

The architecture has no length-dependent parameters, so a trained model runs at
any context size unchanged and the previous result can be asked a sharper
question: is the improvement at $M = 256$ something *large contexts* provide, or
something *training at large context* selects? Evaluation only, no retraining.

#align(center, table(
  columns: 6, stroke: none, align: (left, right, right, right, right, right),
  table.hline(stroke: rule),
  [*evaluated at*], [$M=4$], [$M=16$], [$M=64$], [$M=256$], [],
  table.hline(stroke: rule),
  [trained at $M=16$ — completion error], [0.700], [0.851], [0.996], [0.942], [],
  [trained at $M=16$ — gain],             [0.676], [0.835], [0.715], [0.175], [],
  [trained at $M=256$ — completion error],[0.541], [0.545], [0.558], [0.545], [],
  [trained at $M=256$ — gain],            [-0.011], [-0.002], [-0.002], [-0.014], [],
  table.hline(stroke: rule),
))

The short-trained model does not improve at long context; it *degrades*, to 0.942
against the 0.561 a model trained there reaches. The long-trained model's gain is
zero at every length, including $M = 4$ where sixteen-fold spare capacity is
available, and its completion error is flat to within 0.02 across a 64-fold
change in context size. The completion-trained model behaves identically to it
(gain $-0.015$ to $-0.003$, error 0.440–0.457).

#fig(include "/figures/length_transfer.typ", caption: [
  Gain against test-time context size for three trained models. Only gain is
  plotted: identification accuracy's chance level is $1 slash M$, which falls
  64-fold across this axis, so the two cannot share a panel honestly.
])

== How far the learned mechanism travels

Identification accuracy of the recall-trained model at $M = 16$ — so chance is a
constant 0.063 throughout — on four pools at increasing distance from its
training distribution:

#align(center, table(
  columns: 3, stroke: none, align: (left, left, right),
  table.hline(stroke: rule),
  [*pool*], [*what it is*], [*id. acc.*],
  table.hline(stroke: rule),
  [held-out MNIST], [digits it never saw], [*1.000*],
  [Fashion-MNIST], [same medium, new content], [*0.651*],
  [MNIST, pixels permuted],
  [the same pixels under one fixed permutation], [*0.116*],
  [random fields], [blocky low-frequency noise], [*0.222*],
  table.hline(stroke: rule),
))

The permuted pool is the informative one. Its images have the same pixels, the
same marginal statistics and the same pairwise distances as MNIST's, so a
nearest-neighbour matcher on raw pixels scores identically on both. The model
scores 1.000 and 0.116.

#fig(include "/figures/dataset_transfer.typ", caption: [
  Identification accuracy of one recall-trained model across four image pools,
  with the chance level of 0.063 marked. The pools run left to right from the
  training distribution to unrelated noise.
])

== The recall solution is not a useful starting point

2 000 steps of completion training, from exp1's recall-trained weights (exp13)
against random initialisation (exp14). Same budget, schedule, data and seed:

#align(center, table(
  columns: 3, stroke: none, align: (left, right, right),
  table.hline(stroke: rule),
  [*initialised from*], [*D*], [*B*],
  table.hline(stroke: rule),
  [exp1's recall weights (exp13)], [0.439], [0.435],
  [random noise (exp14)],          [0.454], [0.453],
  table.hline(stroke: rule),
))

The head start is worth 0.015 nMSE, about 3% relative — and the fine-tuned model
has lost its retrieval in the process (gain 0.004, against 0.834 before).

== Nothing generalises across digit classes

Digit split, evaluated on conditions whose target is absent. D is novel classes
(5–9); F is novel images of the training classes (0–4).

#align(center, table(
  columns: 4, stroke: none, align: (left, right, right, right),
  table.hline(stroke: rule),
  [], [C seen], [F novel img], [*D novel class*],
  table.hline(stroke: rule),
  [recall-trained (exp8)], [0.712], [0.705], [*1.006*],
  [completion-trained (exp9)], [0.015], [0.648], [*1.224*],
  [ridge, fitted on 0–4], [0.589], [0.582], [0.851],
  [best soft look-up], [0.941], [0.938], [0.933],
  table.hline(stroke: rule),
))

The recall-trained model completes unseen digit classes at 1.006 — the
mean-image level, to within noise — while retrieving those same classes with
identification accuracy 1.000. The completion-trained model is *worse* than the
mean image on them, at 1.224, despite reaching 0.648 on novel images of the
classes it was trained on. Only ridge regression, at 0.851, does better than
predicting the average digit.
