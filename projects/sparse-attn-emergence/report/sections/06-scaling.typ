#import "/template.typ": *

= The difficulty law does not transfer either

On the linear map, learnability tracks $C(S, s)$ — how many candidate position-subsets a row
could have. That is the paper's sparsity story made quantitative. Does it describe content-keyed
patterns too?

To ask on one axis, both halves have to be present in one task. In *k-of-m recall* each block is
$m = 8$ attribute tokens plus a value, and a query block's value equals that of the earlier
block agreeing with it on the $k$ *relevant* attributes; the other $m - k$ are re-randomised. The
model must learn which $k$ of $m$ matter — a subset out of $C(m, k)$, fixed for the run, exactly
analogous to the row support — and then match on them, which is content-keyed and per-sequence.

#fig(
  include "/figures/kofm_k.typ",
  caption: [Solve rate against $k$. Difficulty falls monotonically as more attributes become
    relevant. The two variants differ in whether the retrieval target is unique (below); the
    trend is the same in both, so it is not an artifact of the fix.],
)

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*k*], [*1*], [*2*], [*3*], [*4*], [*6*],
  [$C(8,k)$], [8], [*28*], [56], [*70*], [*28*],
  [seeds solving], [3/16], [6/16], [14/16], [14/16], [*16/16*],
  [median $t^*$], [7 753], [8 722], [7 074], [6 934], [*5 392*],
)

*The law does not carry over.* $C(m,k)$ is largest at $k = 4$ and the difficulty peak is not
there; it is at $k = 1$, where $C$ is *smallest*. And the equal-$C$ pair splits as far as it can:
$k = 2$ and $k = 6$ both have $C = 28$, and they land at 6/16 and 16/16.

#fig(
  include "/figures/kofm_candidates.typ",
  caption: [The same numbers ordered by candidate count. If $C$ governed difficulty this would
    descend from left to right; the two $C = 28$ cells sit at opposite ends instead.],
)

== What governs it instead

Match *discriminability*. A query shares its $k$ relevant attributes with its source, and by
chance about $m\/A = 2$ of its irrelevant attributes with any other block. At $k = 1$ the correct
evidence is one shared token against roughly two coincidental ones — the signal is weaker than
the noise. At $k = 6$ it is six against two, and the source stands out.

So the two families are governed by different quantities, and in opposite directions:

#table(
  columns: (auto, auto, auto),
  [], [*position-keyed*], [*content-keyed*],
  [what is being searched], [which $s$ of $S$ positions], [which $k$ of $m$ attributes],
  [difficulty tracks], [$C(S,s)$ — the candidate count], [match discriminability — $k$ against $m\/A$],
  [sparser is], [*easier* ($s = 1$ is trivial)], [*harder* ($k = 1$ is the worst cell)],
)

#callout(title: [A control worth keeping])[
  The first version of this task made low $k$ ambiguous by construction: with alphabet $A = 4$
  and four context blocks, a non-source block also matches on the relevant attributes with
  probability $A^(-k)$ — 0.75 expected spurious matches at $k = 1$ against 0.0007 at $k = 6$.
  Difficulty would then fall with $k$ for a reason that has nothing to do with the search.

  Giving every context block a distinct relevant-attribute tuple removes the ambiguity
  entirely. It lifted the low-$k$ cells off zero (0/16 → 3/16 at $k = 1$, 0/16 → 6/16 at
  $k = 2$) and left the trend unchanged — which is what makes the trend reportable. Both
  variants are plotted above for exactly that reason.
]

== What this costs the paper's story

Nothing about emergence, and quite a lot about extrapolation. The plateau-then-jump behaviour
and its seed-randomness appear in every task here, positional or content-keyed. But the two
laws the paper's synthetic work yields — that difficulty tracks the candidate count, and that a
mixer beats attention — are both properties of position-keyed routing, and neither survives the
move to the kind of pattern its motivating capabilities need. One reverses; the other does not
apply.
