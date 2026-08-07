#import "/template.typ": *

= Setup

== The episode

An episode holds #emph[N] characters, each with #emph[K] support drawings and
one query drawing, flattened into a bag of `(pos, value, ref)` tokens. A support
drawing contributes its observed pixels plus a label token carrying the class's
slot; a query drawing contributes its observed pixels plus a label token whose
value is `MASK` — that token is the target.

#fig(
  image("/assets/episode.png", width: 62%),
  caption: [One 5-way 1-shot episode. Top row of each pair: the full drawing.
    Bottom row: what the model actually receives — the 196 pixels of the
    episode's shared position pool, everything else blank.],
)

Three properties make this a real in-context test, and each is load-bearing:

- *Label slots are drawn fresh per episode*, so a memorised class-to-slot map is
  worthless.
- *A query's label appears nowhere among its own tokens.* The only route to it
  runs through matching the query's pixels against a different drawing of the
  same character.
- *`ref` cannot shortcut it*: a query's `ref` never co-occurs with a label. This
  is precisely the shortcut that made universal-ar's exp35 and exp36 vacuous,
  and Omniglot closes it without an anonymisation trick.

All drawings in an episode observe the #emph[same] random position pool. Without
that, support and query would describe disjoint pixels and cross-drawing
matching would be ill-posed rather than merely hard.

== The model

Deliberately identical to `universal-ar/experiments39.py` — same depth, width,
head size, and additive embedding scheme. This project varies the data, not the
model, so a difference in outcome is attributable to the substrate.

#kv(
  ("dataset", "dpdl-benchmark/omniglot, 28x28, inverted"),
  ("episode", "5-way 1-shot, 1 query per class"),
  ("observed pixels", "196 of 784, shared pool per episode"),
  ("tokens per episode", "1970"),
  ("value vocabulary", "8 pixel bins + 5 label slots + MASK"),
  ("layers / width / head", "4 / 256 / 32"),
  ("parameters", "3379725"),
  ("effective batch", "16 episodes (8 micro x 2 accum)"),
  ("optimiser", "adamw 3e-4, warmup-cosine, clip 1.0"),
  ("steps", "12000"),
)

The loss is cross-entropy on the masked query-label tokens only. Pixel
completion is deliberately excluded here and added as an auxiliary term in exp3.

== What the result is measured against

#table(
  columns: (auto, auto, 1fr),
  [*reference*], [*value*], [*why it is here*],
  [chance], [0.200], [floor],
  [pixel 1-NN (cosine)], [0.444],
  [The bar for claiming any learning at all. Computed in-repo on the #emph[same]
   196 observed pixels — a baseline on full images would be measuring a
   different task.],
  [seen-character accuracy], [reported],
  [Episodes built from background characters. The gap against unseen characters
   is the memorisation gap.],
)
