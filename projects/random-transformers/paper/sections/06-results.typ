#import "/template.typ": *

= Results <sec-results>

Every number below is a median over seeds. The min–max range across seeds is
given wherever it matters. Chance sits beside each accuracy, in the same row.

Accuracy is measured per scored token, following the authors' code. For three of
the four tasks each sequence has exactly one scored token, so this is the same as
sequence accuracy. Decimal addition is the exception, and @sec-limitations
explains why we report it their way.

== The main result

#fig(include "/figures/main_table.typ", caption: [
  Accuracy on the four algorithmic tasks. "Random" models have every attention
  and feed-forward weight frozen at initialisation. Chance is drawn as its own
  bar because it differs per task.
])

At width 1024, embedding-only training solves all four tasks.

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt + luma(200), inset: 5pt, align: (left, right, right, right, right),
  table.header([*Task*], [*Random 1024*], [*Paper*], [*Chance*], [*Seed range*]),
  [Modular addition],      [1.000], [1.000], [0.005], [1.00–1.00],
  [Needle in a haystack],  [1.000], [1.000], [0.008], [1.00–1.00],
  [Decimal addition],      [1.000], [1.000], [~0],    [1.00–1.00],
  [Parenthesis balancing], [1.000], [1.000], [0.681], [1.00–1.00],
)

All five seeds reach 1.000 on every task. Not "close to", but the same value
every time. The paper's central claim replicates from an independent
implementation.

Of the twenty cells in the full table, seventeen fall within 0.15 of the paper.
The three that do not are discussed in @sec-limitations. All three are baselines,
not the condition under test.

== Width decides everything

#fig(include "/figures/width_needle.typ", caption: [
  Needle in a haystack. The fully trained curve is flat; the random curve is a
  step. Note the log scale on width.
])

The two curves have completely different shapes.

#table(
  columns: 8,
  stroke: 0.5pt + luma(200), inset: 4pt, align: (left, right, right, right, right, right, right, right),
  table.header([*Width*], [16], [32], [64], [128], [256], [512], [1024]),
  [modular addition, random], [0.000], [0.009], [0.018], [0.001], [0.986], [1.000], [1.000],
  [modular addition, trained], [0.948], [1.000], [1.000], [1.000], [1.000], [0.990], [0.997],
  [needle, random],  [0.076], [0.133], [0.160], [0.167], [0.171], [1.000], [1.000],
  [needle, trained], [0.109], [0.158], [0.163], [0.170], [0.169], [0.971], [0.916],
  [decimal, random],  [0.227], [0.229], [0.249], [0.292], [0.303], [0.697], [1.000],
  [decimal, trained], [0.381], [1.000], [1.000], [1.000], [1.000], [1.000], [1.000],
)

Chance is 0.005, 0.008 and ~0 for the three tasks in that order.

Fully trained models work at every width. Modular addition is at 0.948 by width
16. Decimal addition reaches 1.000 by width 32.

Random models show a threshold instead. Modular addition is stuck at chance
through width 128, then jumps to 0.986 at width 256. Needle in a haystack sits
at 0.17 through width 256, then jumps to 1.000 at width 512.

The jump is not gradual. One doubling of width takes modular addition from 0.001
to 0.986.

Needle in a haystack is the interesting row. Both conditions are near 0.17 up to
width 256. So the *fully trained* model is not solving it either at those widths.
The two conditions diverge only at 512, and then the random one wins.

== Which matrices must be trained

#fig(include "/figures/ablation.typ", caption: [
  Freezing part of the embedding machinery, at width 1024. No subset works
  everywhere.
])

#table(
  columns: 6,
  stroke: 0.5pt + luma(200), inset: 5pt, align: (left, right, right, right, right, right),
  table.header([*Optimised*], [*Mod. add*], [*Needle*], [*Decimal*], [*Parens*], [*Paper's pattern*]),
  [all three],        [1.000], [1.000], [1.000], [1.000], [all four],
  [$E_"tok"$ and $U$],[1.000], [0.968], [0.398], [1.000], [decimal fails],
  [$E_"tok"$, $E_"pos"$], [0.000], [1.000], [0.923], [1.000], [mod. add fails],
  [$U$ only],         [0.000], [0.167], [0.231], [0.825], [all four fail],
  [chance],           [0.005], [0.008], [~0],    [0.681], [],
)

The paper's conclusion holds. Every proper subset breaks at least one task.

Training only the unembedding fails everywhere. That is the informative row. The
unembedding cannot change what the frozen stack computes. It can only change how
the answer is read out. Reading out is not enough.

Freezing the positional embeddings costs decimal addition specifically, dropping
it to 0.398. Decimal addition is the one task where position carries the
arithmetic — digit $i$ of the answer depends on digit $i$ of both inputs.

== Memorization: the frozen interior cannot be used as storage

#fig(include "/figures/memorization_bits.typ", caption: [
  Bits of arbitrary association stored per trainable parameter, width 128.
])

#table(
  columns: 5,
  stroke: 0.5pt + luma(200), inset: 5pt, align: (left, right, right, right, right),
  table.header([], [*Recalled*], [*Bits/param*], [*Paper's bits/param*], [*Trainable params*]),
  [Fully trained], [0.671], [2.40], [2.88], [659,584],
  [Random],        [0.023], [0.20], [0.41], [262,784],
  [Chance],        [0.002], [—],    [—],    [—],
)

A fully trained model stores twelve times as much per parameter. The paper's
ratio is seven times. Both of our absolute numbers are about half the paper's,
which we attribute to our shorter budget — 60,000 steps against their roughly
168,000.

The direction and the size of the gap are the claim, and both reproduce. Nothing
here is structure to be discovered. It is 262,144 arbitrary facts, and they have
nowhere to go but the embeddings.

== Subspace selection, not sparsification

#fig(include "/figures/subspace_L1.typ", caption: [
  Fraction of activation variance explained by the top ten principal components
  against the top ten individual neurons, after layer 1. The gap between the two
  is the claim.
])

#table(
  columns: 5,
  stroke: 0.5pt + luma(200), inset: 5pt, align: (left, left, right, right, right),
  table.header([*Task*], [*Model*], [*Top 10 components*], [*Top 10 neurons*], [*Ratio*]),
  [Modular addition], [random], [0.760], [0.029], [26x],
  [Modular addition], [trained], [0.919], [0.057], [16x],
  [Decimal addition], [random], [0.904], [0.049], [18x],
  [Decimal addition], [trained], [0.846], [0.036], [24x],
  [Needle],           [random], [0.380], [0.017], [22x],
  [Needle],           [trained], [0.742], [0.030], [25x],
  [Parentheses],      [random], [0.937], [0.047], [20x],
  [Parentheses],      [trained], [0.999], [0.071], [14x],
  [Memorization],     [random], [0.887], [0.202], [4x],
  [Memorization],     [trained], [0.692], [0.158], [4x],
)

Ten directions out of 1024 explain most of the variance. Ten neurons out of 1024
explain almost none. The ratio is 14 to 26 across the algorithmic tasks.

So the computation lives in a low-dimensional subspace. That subspace is not
aligned to individual neurons. A pruning or lottery-ticket story would predict
the opposite pattern, and does not fit.

Memorization behaves differently, as the paper reports. The random model is *more*
concentrated than the trained one, 0.887 against 0.692. Concentration is a
liability there, not an asset — @sec-analysis takes this up.

== Circuit imitation: the falsification test

#fig(include "/figures/circuit_imitation.typ", caption: [
  Divergence from a random target network, against how wide that target is.
  Lower is better. The student is width 512 in both conditions.
])

#table(
  columns: 8,
  stroke: 0.5pt + luma(200), inset: 4pt, align: (left, right, right, right, right, right, right, right),
  table.header([*Target width*], [4], [8], [12], [16], [32], [64], [128]),
  [random student], [0.009], [0.020], [0.034], [0.039], [0.178], [0.580], [1.070],
  [trained student],[0.003], [0.002], [0.005], [0.004], [0.007], [0.016], [0.121],
)

This is the prediction subspace selection makes, and it holds.

The random student tracks narrow targets almost perfectly. At target width 16 its
divergence is 0.039. At 32 it is 0.178, four and a half times worse. At 128 it is
1.070, a further sixfold.

The fully trained student shows no such break. It stays under 0.02 out to target
width 64. Its degradation at 128 is real but eight times smaller than the random
student's.

The paper reports a sharp increase between target widths 12, 16 and 32. Ours
falls between 16 and 32. Same location, same shape.

== Language modeling

#fig(include "/figures/lm_scaling.typ", caption: [
  Validation cross-entropy on TinyStories against width, at two depths. Lower is
  better. One run per point.
])

#table(
  columns: 6,
  stroke: 0.5pt + luma(200), inset: 5pt, align: (left, right, right, right, right, right),
  table.header([*Width*], [32], [64], [128], [256], [512]),
  [random, 2 layers],  [3.787], [3.647], [3.523], [3.361], [2.750],
  [trained, 2 layers], [3.087], [2.683], [2.312], [2.038], [1.900],
  [random, 4 layers],  [3.772], [3.639], [3.523], [3.413], [2.886],
  [trained, 4 layers], [3.084], [2.630], [2.211], [1.953], [1.815],
)

Cross-entropy in nats per token. A model that guessed uniformly over the 10,000
token vocabulary would score 9.21.

The random width-512 model reaches 2.750, with 10.5M trainable parameters. The
paper reports 2.64. A fully trained model matches that somewhere between width 32
(3.087) and width 64 (2.683), which needs 1.3M trainable parameters.

So the random model needs roughly eight times the trainable parameters to match a
fully trained one. The paper puts the factor at fifteen. Same order, same
conclusion: random transformers do language modeling, and do it inefficiently.

Depth does not help the random models. Two layers beat four at every width: 2.750
against 2.886 at width 512. The fully trained models show the opposite, 1.900
against 1.815. The paper reports exactly this reversal.

== What the text looks like

The claim that survives is grammaticality, not sense. Both completions continue
the same prompt about a boy named Max and his tower.

#callout(title: [Fully trained, width 512, 2 layers — cross-entropy 1.900])[
  ...Suddenly, his tower fell down to one side and all the shapes vanished! Max
  smiled, feeling proud that he built something special with his arms and bouncy
  blocks. From that day on, Max was known as a superhero.
]

#callout(title: [Random, width 512, 2 layers — cross-entropy 2.750])[
  ...Suddenly, his tower fell down to fall down and hurt. He was being careful so
  careful. He was so strong and careful with lots of things that he would be
  different from all and make sure. One day, Lily went to the tower of different
  shapes and made a beautiful tower together. It was then and was very proud.
  Lily saw that Max had a big red ball so yellow policy.
]

The random model's output is almost entirely well-formed English. Subjects agree
with verbs. Clauses close. It stays on topic: towers, shapes, blocks, a child's
name.

It also says "yellow policy", and elsewhere "a beautiful gap in the tree". The
sentences are grammatical and the meanings are not tracked. This is the paper's
qualitative finding, and it is what our samples show.
