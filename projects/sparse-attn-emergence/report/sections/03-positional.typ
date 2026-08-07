#import "/template.typ": *

= On position-keyed routing, the mixer wins

Sweeping sparsity on the linear map, all three architectures, best learning rate per cell:

#fig(
  include "/figures/crossover.typ",
  caption: [Solve rate against row sparsity. The crossover sits at $s = 4$: below it attention
    is better, above it the static mixer is the only architecture still solving anything. KDA
    fails from $s = 4$ — one cell earlier than the transformer.],
)

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  [*s*], [*3*], [*4*], [*5*], [*6*], [*7*], [*8*],
  [transformer], [1.00], [0.50], [0.06], [0.00], [0.00], [0.00],
  [static mixer], [1.00], [*0.69*], [*0.62*], [*0.31*], [*0.31*], [*0.19*],
  [KDA], [1.00], [0.00], [0.00], [0.00], [0.00], [0.00],
)

Below the crossover attention is not merely adequate but *faster*: both it and the mixer solve
$s = 3$ completely, the transformer in a median 732 steps against the mixer's 3 693. So the
paper's claim holds in the regime it was made about, and the unqualified reading — that a mixer
learns this task faster — does not.

== The difficulty is the size of the search

Across the whole $(S, s)$ surface, learnability tracks $C(S, s)$, the number of candidate
supports per row, and not $s$ or the context length separately:

#table(
  columns: (auto, auto, auto, auto),
  [*$C(S,s)$*], [*≲ 500*], [*1 800 – 5 000*], [*≳ 8 000*],
  [outcome], [always solves], [31–50% of seeds], [never],
)

Cells from different context lengths land together when matched on $C$: $S=16, s=3$ ($C = 560$)
and $S=32, s=2$ ($C = 496$) both solve; $S=16, s=4$ ($C = 1 820$) and $S=32, s=3$ ($C = 4 960$)
both land in the middle. So "longer context makes sparse patterns harder to find" resolves into
something sharper — *longer context inflates the number of wrong patterns*, and difficulty
follows that count.

== Why the mixer wins here

The correct routing on this task is content-independent: the same $s$ slots in every sequence.
A constant mixing matrix can express exactly that, and gradient descent tunes it directly with
no softmax competition and no search among discrete alternatives. Attention has to discover
the same fixed pattern through a bilinear form that starts near-uniform.

KDA is the informative failure. It has no softmax either, yet it tracks *attention*, not the
mixer — so removing softmax competition is not what buys the mixer its range. What does is that
its routing weights are free parameters rather than something computed per sequence.

#callout(title: [A caveat on the metric])[
  Read strictly — every row of $A$ learned, not 15 of 16 — both architectures collapse past
  $s = 4$ and the mixer's advantage lives largely in partial solutions. At $s=5$ it clears the
  0.95 accuracy bar in 10 of 16 seeds but learns every row in only 1. At $S=16$ that bar
  tolerates one unlearned row.
]
