#import "/template.typ": *

// OBLIGATIONS
//  - Every metric carries its chance level or baseline IN THE SAME ROW.
//  - Seed variance, or an explicit statement that a number is a single run.
//  - One figure per claim. Report what happened; interpretation is §7.

= Results

Throughout, *nMSE* is the model's mean squared error on hidden pixels divided by
that of predicting the average training image, so 1.0 is "no better than
ignoring the input".

Tables in this section report conditions *B and D side by side* rather than any
summary of them. The two use context images the model has never seen and differ
in one respect only — whether the query's true image is among them — so placing
them adjacent asks the reader to make the comparison the paper is about, on
measured numbers, rather than to trust a derived one. When B and D are the same
number, the model gains nothing from the answer being present: it is not reading
its context. When B is near zero and D is near one, it is retrieving and nothing
else.

One measure is reported but never load-bearing. *Identification accuracy* — does
the output land on the correct one of the $M$ context images, on hidden pixels —
is inflated by models that never retrieve, because a good completion resembles
the true image and so picks the right neighbour by itself: the
completion-trained model scores 0.951 at $M = 4$ while its B and D are 0.455 and
0.440, i.e. identical. Its chance level is also $1 slash M$, which moves 64-fold
across the context sizes used here. It is quoted where the answer is present and
the context size fixed, and nowhere else.

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
  [*trained on*], [*B, answer present*], [*D, answer absent*], [*best D*],
  table.hline(stroke: rule),
  [recall only (exp1)],     [0.017], [0.852], [0.635],
  [mixed (exp3)],           [0.484], [0.618], [0.459],
  [completion only (exp2)], [0.672], [0.671], [0.458],
  table.hline(stroke: rule),
))

The mixture reaches the completion ceiling exactly (0.459 against 0.458) and
still answers better when the target is present than when it is absent (0.484
against 0.618), which the completion-trained model does not (0.672 against
0.671).

== Generalisation appears exactly when retrieval fails

Recall training, state fixed at 16 384 numbers, one seed per row (the $M = 16$
row is the mean of three; spread is given in the retrieval table above):

#align(center, table(
  columns: 6, stroke: none, align: (right, right, right, right, right, right),
  table.hline(stroke: rule),
  [$M$], [context content], [*B, present*], [*D, absent*], [best D], [look-up ceiling],
  table.hline(stroke: rule),
  [4],   [3 136],   [0.014], [0.854], [0.777], [1.237],
  [16],  [12 544],  [0.017], [0.852], [0.635], [1.002],
  [64],  [50 176],  [0.036], [0.658], [0.535], [0.886],
  [256], [200 704], [*0.556*], [*0.561*], [0.443], [0.786],
  table.hline(stroke: rule),
))

Read the last row against the first three. Up to $M = 64$, having the answer in
the context is worth almost everything — 0.017 with it against 0.852 without. At
$M = 256$ the two numbers are 0.556 and 0.561: the same number. The
completion-trained runs, which never retrieve by construction, post the same
coincidence (exp2: 0.672 and 0.671; exp7: 0.671 and 0.677; exp12: 0.670 and
0.681). The recall-trained model at $M = 256$ is indistinguishable from them, its
best D of 0.443 sits on their ceiling (0.458, 0.450, 0.458), and its condition A
equals its condition C (0.128 against 0.134).

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
  [state], [content \/ state], [A], [*B, present*], [*D, absent*], [id. acc.],
  table.hline(stroke: rule),
  [16 384 (exp1)], [0.8], [0.015], [0.017], [0.852], [1.000],
  [8 192 (exp15)], [1.5], [0.017], [0.019], [0.819], [1.000],
  [4 096 (exp16)], [3.1], [0.022], [0.024], [0.755], [1.000],
  [2 048 (exp17)], [6.1], [0.031], [0.035], [0.681], [1.000],
  table.hline(stroke: rule),
))

Shrinking the memory eightfold, with the context held fixed, moves the two
conditions towards each other monotonically: the answer-present error doubles
(0.017 #sym.arrow 0.035) while the answer-absent error improves (0.852
#sym.arrow 0.681). The context never changed, so the improvement cannot be
information the context supplied.

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
  [trained at $M=16$ — B, answer present], [0.024], [0.018], [0.281], [0.767], [],
  [trained at $M=16$ — D, answer absent],  [0.700], [0.851], [0.996], [0.942], [],
  [trained at $M=256$ — B, answer present],[0.552], [0.546], [0.560], [0.559], [],
  [trained at $M=256$ — D, answer absent], [0.541], [0.545], [0.558], [0.545], [],
  table.hline(stroke: rule),
))

The short-trained model does not improve at long context; it *degrades*, to 0.942
against the 0.561 a model trained there reaches. Its answer-present error rises
over the same range (0.018 #sym.arrow 0.767) as the memory is asked to hold more
than it ever had to, but the two never converge — it is still retrieving
something at $M = 256$.

The long-trained model's two rows are the same number at every length, including
$M = 4$ where sixteen-fold spare capacity is available, and both are flat to
within 0.02 across a 64-fold change in context size. The completion-trained model
behaves identically to it (0.455/0.440 at $M=4$, 0.463/0.450 at $M=256$).

A caution the same table supplies. At $M = 256$ identification accuracy reads
0.322 for the recall-trained model, 0.462 for the one trained at 256 and 0.454
for the completion-trained one — it ranks the only model that is still
retrieving *last*. The reason is that it rewards a good output and does not ask
how the output was reached: the recall-trained model's memory is being handed
sixteen times what it was tuned for, so its outputs are poor (0.767 even with
the answer present, against 0.463) and land near the right neighbour less often.
On this grid the intuitive metric gives the exact opposite of the right answer,
which is why §5 rests every retrieval claim on the paired errors instead.

#fig(include "/figures/length_transfer.typ", caption: [
  Completion error with the answer absent, against test-time context size, for
  three trained models. The answer-present numbers are in the table above rather
  than on the same axes, since only one of the three models separates them.
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
has lost its retrieval in the process: its answer-present and answer-absent
errors are now 0.435 and 0.439, against 0.017 and 0.852 before.

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
