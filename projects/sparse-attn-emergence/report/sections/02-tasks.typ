#import "/template.typ": *

= Two kinds of pattern

The distinction the report turns on is not in-context versus in-weights. It is *what the head
has to key on*.

#table(
  columns: (auto, auto, auto),
  [*task*], [*what varies per sequence*], [*what the head must key on*],
  [linear map], [nothing — $A$ is fixed for the run], [*position* — a fixed set of slots],
  [cellular automata], [the rule], [*position* — a fixed local window],
  [induction / IOI / copying], [the content], [*content* — match a token, copy its successor],
)

In the first two, the model learns one routing and reuses it in every sequence. In the third
the routing is recomputed from the tokens themselves, and the position holding the answer moves
from sequence to sequence.

== The linear map (position-keyed)

A binary matrix $A$ with exactly $s$ ones per row is drawn once. Each sequence is a random
$x_0$ followed by $x_1 = A x_0 mod 2$, so output bit $i$ is the parity of the $s$ input bits
that row $i$ selects. To predict it, a head must attend to exactly those $s$ positions — known
in advance, identical in every sequence.

The first half of each sequence is uniform noise, so its cross-entropy is pinned at $ln 2$;
all metrics here use the second half only.

== Associative recall (content-keyed)

A sequence is pairs $a, f(a)$ where *$f$ is a fresh random permutation per sequence*. When a
key repeats, its value can only be recovered by finding the earlier occurrence of that key and
copying the token after it — the canonical induction circuit. Nothing about where to look is
fixed: it depends on which key is being asked about.

We score *recall accuracy* on pairs whose key has already appeared, where the answer is
determined rather than a $1\/32$ guess. Two layers minimum for every architecture, since a
one-layer model cannot express the circuit.

#callout(title: [The three architectures, and what separates them])[
  #table(
    columns: (auto, auto, auto),
    [*model*], [*position mixing*], [*conditioned on content?*],
    [transformer], [query–key match through a softmax], [yes],
    [static mixer], [one learned matrix, masked causal], [*no*],
    [KDA linear attention], [delta-rule associative memory, key match], [yes, no softmax],
  )

  The mixer is the only one whose mixing weights are *constants* — they are the same whatever
  the tokens are. KDA is the useful third point: content-conditioned like attention, but with
  no softmax competition, so it separates "softmax is the problem" from "content-dependence is
  the problem".
]

The causal mixer deserves one note. A standard Mixer's token-mixing is a two-layer MLP over
positions, which #emph[cannot] be made causal — its hidden units see every position. The causal
analogue is a single masked matrix, so our mixer carries 1 024 mixing parameters against the
transformer's ~65 000 of QKVO. Where the mixer wins below, it wins with far fewer parameters
devoted to routing.
