#import "/template.typ": *

= Task formulation <sec:task>

== Episodes

An episode holds $N$ characters, each with $K$ support drawings and $Q$ query
drawings, flattened into a single bag of tokens. Every drawing contributes
$C$ pixel tokens plus one label token at the reserved address $p_"label"$:

- a *support* drawing's label token carries its class's slot as its value;
- a *query* drawing's label token carries `MASK`, and is the prediction target.

Class-to-slot assignment is a fresh random permutation each episode, so a slot
means nothing across episodes.

#fig(
  image("/assets/episode.png", width: 62%),
  caption: [One 5-way 1-shot episode. For each pair of rows, the upper shows the
    full $28 times 28$ drawing and the lower shows what the model actually
    receives: the $C = 196$ pixels of the episode's shared position pool, with
    everything else blank.],
)

== What makes it a genuine in-context test

Three properties are load-bearing, and each closes a shortcut that the prior
work left open.

+ *Slots are re-drawn per episode.* A memorised class-to-slot map is worthless.
+ *A query's label appears nowhere among its own tokens.* The only route to it
  runs through matching the query's pixels against a *different* drawing that
  shares the character.
+ *`ref` cannot shortcut it.* A query's `ref` never co-occurs with a label, so
  tag-matching alone answers nothing. This is precisely the shortcut that made
  two of the prior experiments vacuous — they trained on label retrieval, which
  is solvable by matching a tag and never requires looking at a pixel.

All drawings in an episode observe the *same* random position pool. Without
that, support and query would describe disjoint pixels and cross-drawing
matching would be ill-posed rather than merely hard.

== Vocabulary

Values span pixel bins $0..n_"bins"-1$, then label slots
$n_"bins"..n_"bins"+N-1$, then `MASK`. Positions span the $28^2$ pixels plus
$p_"label"$. Under the unified vocabulary a model may answer a label query with
a pixel bin; §4.3 measures how often it does.

== The label field

Experiments 1 and 2 place the class *only* on the label token. Experiments 3 and
4 additionally place it on every token of a support drawing, as a fourth field
$ell$ alongside $(p, v, r)$, with `MASK` on every token of a query drawing.
§#link(<sec:analysis>)[6] argues from the exp1/exp2 result why this is the
decisive variable; the short version is that it changes the number of hops the
matching circuit needs from three to two, and removes the one hop that softmax
attention cannot perform.

This is a deliberate deviation from the original premise that "the label is just
a token at $p_"label"$". It is the token-bag analogue of sequence models placing
$y$ *adjacent* to its $x$, which is what makes induction heads learnable there.
