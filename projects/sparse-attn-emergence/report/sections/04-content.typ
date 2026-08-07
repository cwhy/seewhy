#import "/template.typ": *

= On content-keyed routing, the ranking inverts

The same three architectures on associative recall, 30 000 steps, best learning rate per arm:

#fig(
  include "/figures/induction.typ",
  caption: [Median recall accuracy on pairs whose key has already appeared. Chance is
    $1\/32 = 0.031$. The architecture that led on the linear map is the one that cannot do
    this at all.],
)

#table(
  columns: (auto, auto, auto, auto, auto),
  [*architecture*], [*recall accuracy*], [*seeds solving*], [*median $t^*$*], [*recall loss*],
  [*KDA*], [*1.000*], [*14 / 16*], [*15 889*], [*0.0005*],
  [transformer], [0.905], [7 / 16], [26 929], [0.295],
  [static mixer], [0.125], [*0 / 16*], [—], [3.135],
)

#callout(title: [The inversion])[
  #table(
    columns: (auto, auto, auto),
    [], [*linear map (position-keyed)*], [*induction (content-keyed)*],
    [static mixer], [*best* — solves past $s = 4$], [*cannot do it at all*],
    [transformer], [middle], [middle],
    [KDA], [*worst* — fails from $s = 4$], [*best* — 14/16, perfect recall],
  )
  The ordering is exactly reversed. The axis is whether mixing is conditioned on content.
]

The mixer's weights are constants, so it is unbeatable when the correct routing is fixed and
useless when the routing must be computed per sequence. KDA conditions entirely on content
matching, which is this task's native operation — a delta-rule memory is key–value storage
retrieved by key match — and correspondingly poor at finding an arbitrary subset of positions.
Attention does both adequately and neither best: the generalist.

== The mixer's failure is structural, not a capacity limit

A model that scores zero invites the objection that it was simply too small. Three checks say
otherwise — two learning rates, a threefold budget increase, and double the depth:

#table(
  columns: (auto, auto, auto, auto),
  [*mixer configuration*], [*steps*], [*recall accuracy*], [*seeds solving*],
  [2 layers, lr 3e-4], [10 000], [0.032 (chance)], [0 / 16],
  [2 layers, lr 3e-4], [30 000], [0.078], [0 / 16],
  [2 layers, lr 1e-3], [30 000], [0.125], [0 / 16],
  [*4 layers*, lr 3e-4], [30 000], [0.137], [*0 / 16*],
  [*4 layers*, lr 1e-3], [30 000], [0.034], [*0 / 16*],
)

Nothing moves it off zero. For contrast, the transformer went from 1 seed at 10 000 steps to 7
at 30 000 on exactly the same axis — so the budget was the binding constraint for attention and
is not what is binding for the mixer.

What does move is a shortcut worth naming so it is not mistaken for partial induction: recall
accuracy drifts from $0.032$ to $0.125$, and the answer to a recallable pair is always a value
*already present* in the sequence, so spreading probability mass over previously-seen values
beats $1\/32$ without matching anything.

== Emergence itself transfers

The transformer's success here has the same shape the paper describes for the linear map, in a
regime it never tested: emergence is late (median $t^* approx 26 900$ of 30 000 steps), abrupt,
and seed-random — 7 of 16 seeds, the rest still at chance when training stopped. At 10 000
steps exactly one seed had made it, which is why an earlier version of this work wrongly
recorded the task as unlearnable for everything.

#fig(
  include "/figures/emergence_spread.typ",
  caption: [Sorted per-seed emergence times for two independent 16-seed samples of the same
    linear-map configuration (exp1 and exp4, differing only in data order). Medians agree
    within 5%, and both span a factor of 4–5. This spread is the paper's central claim, and it
    is invisible at three seeds.],
)
