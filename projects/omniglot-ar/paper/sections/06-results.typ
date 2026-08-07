#import "/template.typ": *

= Results <sec:results>

== Headline

#callout(title: [Chance, everywhere, throughout])[
  No run cleared its own chance level at any point in 12 000 steps. The loss
  reaches $ln N$ — the entropy of a uniform guess over the label slots — within
  the first few hundred steps and stays there. Nothing moved it: not the label
  field, not binarised values, not an ink-biased pool, not coarse-and-complete
  observation, and not making the query a literal copy of its support.
]

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    [*run*], [*what it changes*], [*chance*], [*1-NN*], [*unseen*], [*seen*],
    [exp1], [baseline], [0.200], [0.431], [0.209], [0.203],
    [exp2], [2-way, 392 px (easier)], [0.500], [0.664], [0.531], [0.500],
    [exp3], [\+ label field], [0.200], [0.431], [0.188], [0.200],
    [exp4], [\+ binarised values], [0.200], [0.431], [0.169], [0.250],
    [exp5], [\+ ink-biased pool], [0.200], [0.459], [0.181], [0.253],
    [exp6], [coarse 10×10, fully observed], [0.200], [0.606], [0.203], [0.222],
    [exp7], [*identity query* (positive control)], [0.200], [*1.000*], [0.191], [0.166],
    [exp8], [plateau recipe, 2-way real], [0.500], [0.729], [0.488], [0.510],
    [exp10], [plateau recipe, coarse 10×10], [0.500], [0.805], [0.488], [0.516],
  ),
  caption: [Final accuracies. Each run's 1-NN floor is computed on that run's own
    observed pixels, so floors are comparable down a column only where the pool
    is the same — exp5, exp6 and exp7 change the pool or the pairing and carry
    their own.],
) <tab:results>

Every "unseen" figure is within noise of its own chance level (standard error
0.022 at $N=5$, 0.044 at $N=2$), and every one is far below the
nearest-neighbour floor computed on the very same pixels.

exp8 and exp10 are 25 000-step runs under the batch-64 / lr-$10^(-3)$ recipe
that crosses the plateau on exact matching in 3000 steps
(§#link(<sec:analysis>)[7.5]). Both stay flat at $ln 2$ throughout — exp10 with
a nearest-neighbour floor of 0.805, so the information is abundant. The recipe
that solves exact matching does not solve approximate matching, and coarsening
the images does not close the gap.

#callout(title: [exp7 is the one to read])[
  In exp7 each query drawing #emph[is] its class's support drawing, so the match
  is exact and nearest neighbour scores 1.000. The model scores 0.191 against
  chance 0.200. Whatever is failing is not the difficulty of telling two
  Omniglot characters apart.
]

#fig(
  include "/figures/excess.typ",
  caption: [Every run as accuracy minus its own chance, so runs with different
    $N$ are comparable on one axis. Zero is "learned nothing".],
)

#fig(
  include "/figures/floor_comparison.typ",
  caption: [Each run against its own floors. The model bar sits on the chance
    bar in every case, well below the nearest-neighbour bar.],
)

== Training behaviour

#fig(
  include "/figures/learning_curves.typ",
  caption: [exp1. Train, seen and unseen accuracy all sit on the chance line for
    the entire run.],
)

#fig(
  include "/figures/loss_curve.typ",
  caption: [exp1 cross-entropy on the masked query-label tokens. It reaches
    $ln 5 approx 1.609$ almost immediately and never leaves.],
)

Train accuracy is at chance throughout, alongside test. That is not evidence of
a broken model: because class-to-slot assignment is re-drawn every episode,
there is no memorisable component to fit, so a model that has not learned the
in-context circuit scores chance on training episodes too.

== What the model did learn

Not nothing, and the detail is diagnostic. From the first evaluation after
initialisation — step 250, in every run — the open-vocabulary accuracy equals
the slot-restricted accuracy #emph[exactly], and stays equal for all 47
remaining evaluations. The model always answers a label query with a label slot
and never with a pixel bin: out of a fourteen-value vocabulary it learned,
almost immediately, to emit only the values that are syntactically legal in that
position.

It learned the #emph[form] of the answer and nothing about its content.
§#link(<sec:analysis>)[7] localises which single capability is missing.
