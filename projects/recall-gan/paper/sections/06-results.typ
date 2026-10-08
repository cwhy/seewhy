#import "/template.typ": *

= Results <sec:results>

All numbers are over 12 runs per configuration: the median, with the smallest
and largest run in brackets. Chance levels are given beside each.

== The discriminator relies on its state when the generator sees the context

@tab:main summarises the end of training for the two generators.

#figure(
  table(
    columns: 4, align: (left, right, right, left), stroke: none,
    table.hline(),
    [*measure (end of training)*], [*context-aware*], [*blind*], [*chance*],
    table.hline(stroke: 0.5pt),
    [AUROC, context vs generated], [0.970 [0.958, 0.981]], [1.000 [1.000, 1.000]], [0.5],
    [same, scored with another episode's context], [0.570 [0.501, 0.657]], [1.000 [1.000, 1.000]], [0.5],
    [generated images rejected (P < 0.5)], [0.821 [0.634, 0.883]], [1.000 [1.000, 1.000]], [0.5],
    [generator variety (pixel std)], [0.180 [0.164, 0.193]], [0.000 [0.000, 0.001]], [—],
    [generator context use], [0.995 [0.924, 1.157]], [—], [0 = none],
    [runs failed], [0 of 12], [12 of 12], [—],
    table.hline(),
  ),
  caption: [End of training, 12 runs each. Real images are the context
  images; "generated" are the generator's. The swap row scores the same
  queries with the context of a different episode.],
) <tab:main>

With the context-aware generator, the discriminator separates context images
from generated ones at AUROC 0.970. Given another episode's context instead,
that falls to 0.570, close to chance. The discriminator's decisions depend on
what it stored.

With the blind generator, the discriminator wins outright: AUROC 1.000 with
either context. Its decisions do not depend on the context at all. Every blind
generator collapsed to a single image, which the discriminator rejects on
sight; @sec:limits discusses what this does and does not show.

@fig:episode shows one episode, scored by the discriminator of the
representative run (the context-aware run with the median classifier
accuracy, seed 59).

#fig(image("/assets/episode.png", width: 100%), caption: [
  One held-out episode (the first draw of a fixed random seed, not chosen).
  Green: context images, which should score high. Red: generated images. Grey:
  real test images that are *not* in the context, which the discriminator never
  saw in training. Number: the discriminator's P(in context).
]) <fig:episode>

== How it develops during training

@fig:curves-class tracks the digit classifier of @sec:probes through training.
It is computed every 500 steps on a fixed set of test images.

#fig(include "/figures/class_probe_curve.typ", caption: [
  Digit-classifier accuracy of the discriminator during training (median of 12
  runs). Chance is 0.1.
]) <fig:curves-class>

#fig(include "/figures/swap_curve.typ", caption: [
  The swap test during training (median of 12 runs). Context images against
  generated images, scored with the episode's own context and with another
  episode's.
]) <fig:curves-swap>

#fig(include "/figures/gen_context_use.typ", caption: [
  The context-aware generator's context use during training, one line per run.
  0 means its output does not depend on the context.
]) <fig:curves-gen>

All 12 context-aware runs follow the same order. First, the discriminator
begins to rely on its state: its swap-test AUROC drops below 0.8 at a median
of step 2750 (range 500–3500), and the digit classifier passes 0.5 at a median
of step 2500. Then the generator follows: its context use passes 0.5 at a
median of step 4000 (range 3000–6500). The generator lagged the discriminator
in all 12 runs, by a median of 1250 steps.

== The generator draws what the context contains

#fig(image("/assets/samples.png", width: 100%), caption: [
  Generated images for four held-out contexts. Middle: the representative
  context-aware generator. Right: the representative blind generator (seed 56).
  Both use the same contexts and the same noise.
]) <fig:samples>

#fig(image("/assets/same_noise.png", width: 100%), caption: [
  The representative context-aware generator with the *same* noise and three
  different contexts, each of 16 images of one digit. Only the context changes
  between rows. Single-digit contexts never occur in training.
]) <fig:same-noise>

The context-aware generator draws digits of the classes present in its context
(@fig:samples). Holding the noise fixed and changing only the context changes
the drawing (@fig:same-noise): a context of zeros gives zeros and a context of
sevens gives sevens. The mapping is not a faithful one, though. A context of
fours gives mostly sevens and nines.

== The memory as a digit classifier <sec:classifier>

#fig(include "/figures/classifier_bars.typ", caption: [
  10-way digit classification from 16 examples per digit (@sec:probes).
  Discriminators: median of 12 runs. Baselines: pixel distance, which needs no
  learning. Chance 0.1.
]) <fig:classifier>

Used as a classifier, the discriminator of the context-aware runs reaches a
median 10-way accuracy of 0.839 (range 0.819–0.849). The pixel baselines see
the same 16 images per digit: nearest neighbour reaches 0.783, and the RBF
density reaches 0.624 with the default bandwidth and 0.788 with the best
bandwidth for this very test. All 12 context-aware runs beat that best
bandwidth. The blind runs sit at chance (median 0.099).

#fig(include "/figures/per_digit_bars.typ", caption: [
  Per digit: AUROC of "this digit vs the rest" under that digit's context.
  Discriminator: median of 12 context-aware runs. Baseline: the RBF density at
  its best bandwidth.
]) <fig:per-digit>

The gain is uneven across digits (@fig:per-digit). The discriminator's
per-digit AUROC ranges over 0.94–0.99, the baseline's over 0.80–1.00. Its
margin comes from the digits pixels confuse: on twos the baseline reaches 0.799
and the discriminator 0.977. On ones it is the other way round, 0.996 for the
baseline against 0.964.

== Density over real images

#fig(include "/figures/density_hist.typ", caption: [
  The discriminator's score on real test images, averaged over the 12
  context-aware runs and 20 random contexts: for the 16 context images, and for
  all other test images, split by whether their digit appears in the context.
]) <fig:density>

The discriminator was never trained to reject an unseen real image, yet it
does (@fig:density). Context images score a median of 0.78, unseen images
0.048, and the AUROC between them is 0.961 (range 0.945–0.984). Unseen images
whose digit appears in the context score higher on average than those whose
digit does not, 0.227 against 0.061. Still, 19% of unseen images score above
0.5.

#fig(image("/assets/density_strips.png", width: 100%), caption: [
  One context (first of the 20 random contexts) and the unseen test images the
  representative discriminator scores highest, at the median, and lowest.
]) <fig:strips>

@fig:strips shows what drives the score. The highest-scored unseen images are
all one digit, seven, at an identical score: the score saturates at a ceiling.
The lowest-scored include threes, although the context holds a three.

== Reliability <sec:stability>

#figure(
  table(
    columns: 4, align: (left, right, right, right), stroke: none,
    table.hline(),
    [*Adam setting*], [*runs failed*], [*collapse events*], [*classifier, median [range]*],
    table.hline(stroke: 0.5pt),
    [$beta_1 = 0, beta_2 = 0.99$ (the method)], [0 of 12], [0], [0.843 [0.823, 0.864]],
    [$beta_1 = 0.5, beta_2 = 0.999$], [1 of 12], [3], [0.811 [0.112, 0.866]],
    table.hline(),
  ),
  caption: [Context-aware generator, 12 runs each. A collapse event is
  generator variety below 0.06 at any evaluation after step 2000, even if it
  recovers. The classifier column is the in-training probe (fixed contexts),
  so it differs slightly from @fig:classifier.],
) <tab:stability>

With the Adam setting of the method, no run failed and no generator collapsed
at any point (@tab:stability). With the more common setting, one run in twelve
failed and three had at least one collapse.
