#import "/template.typ": *

= An objection that does not survive: memorisation

There is a natural suspicion about the paper's cellular-automata task. It draws $N = 256$ rules
*once per run* and samples one per training example, so the model sees the same 256 lookup
tables for all 10 000 steps — roughly 4 KB of information, trivially storable in an
800 000-parameter model. What must be inferred from context would then be only *which* stored
rule is active: an index, not a function. That would be in-context selection, not in-context
learning.

One parameter settles it. Sweep the pool size, including a *fresh table per sequence* drawn
from $4^64$, where memorisation is impossible.

#fig(
  include "/figures/pool_gain.typ",
  caption: [In-context gain — how far per-state loss falls #emph[within] a single sequence — as
    the pool grows. At $N = 1$ it is exactly zero: nothing is inferred, because nothing needs
    to be. By $N = 256$ the curve has already saturated at the unmemorisable limit.],
)

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*N*], [*1*], [*16*], [*256 (paper)*], [*4096*], [*fresh*],
  [memorisable], [16 B], [256 B], [4 KB], [64 KB], [*impossible*],
  [seeds solving], [8/8], [8/8], [8/8], [7/8], [8/8],
  [median $t^*$], [*448*], [1 064], [4 918], [4 540], [*5 776*],
  [in-context gain], [*0.000*], [0.235], [*1.193*], [1.155], [*1.187*],
)

Removing memorisability costs almost nothing: `fresh` matches the paper's setting on solve rate,
timing and inference. The model has learned to identify an unseen rule from the sequence, not to
index a stored table.

#fig(
  include "/figures/pool_tstar.typ",
  caption: [Time-to-emergence over the same sweep. Bigger pools take longer to learn from — but
    the unmemorisable case is only ~17% slower than the paper's, not qualitatively harder.],
)

The $N = 1$ column is what makes this readable rather than assumed. With a single fixed rule the
model solves the task *ten times faster* and the in-context gain is exactly zero. That is what
memorisation looks like on this instrument, and the paper's setting looks nothing like it.

#callout(title: [Verdict on the objection])[
  The cellular-automata task is genuine in-context learning. Pool size controls #emph[how much]
  inference is required, and the paper's $N = 256$ already sits at the ceiling. The
  memorisation critique — which this project advanced before testing it — is wrong.
]

What remains true is the narrower point: the CA task is in-context about *which rule*, while its
attention pattern is still a fixed local window. In-context learning and content-keyed routing
are independent axes, and the paper varies only the first.
