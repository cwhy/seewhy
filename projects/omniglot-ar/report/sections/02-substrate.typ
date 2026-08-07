#import "/template.typ": *

= Why the substrate was the problem

Three defects in MNIST make it unable to answer the question, and they compound.

== Ten classes, six thousand examples each

Memorising a class prototype in the weights is always the fastest way down the
loss surface. Anonymising the labels removes the #emph[payoff] at evaluation
time but not the #emph[pressure] during training: across episodes the label
token is uncorrelated with the image, so the shortest path is to ignore the
label pathway entirely. The gradient never acquires a reason to build the
binding circuit that the claim is about.

== Held-out samples are not held-out concepts

universal-ar split on samples. Its "generalisation" metric therefore measured
within-class interpolation, which a model that memorised ten prototypes scores
perfectly on. Nothing in the setup ever asked whether the mechanism extends to a
class the model has not seen — which is the entire claim.

== Digits have no parts

The founding principle was to bind seen parts into an unseen whole. MNIST digits
offer no part structure to bind.

= Why Omniglot

Omniglot was built as the transfer-learning inverse of MNIST, and it repairs all
three defects at once.

#table(
  columns: (auto, 1fr),
  [*property*], [*what it fixes*],
  [1 623 characters, \ 20 drawings each],
  [Twenty examples is far too few to build a usable prototype in the weights.
   The only way to answer is to read the support set — the pressure that made
   MNIST degenerate is removed by the data itself.],
  [Disjoint background / \ evaluation splits],
  [964 characters (30 alphabets) versus 659 (20 alphabets), sharing no
   characters. Test episodes use characters the model has never seen, so
   memorisation is impossible #emph[by construction] rather than by an
   anonymisation trick. Any above-chance number is in-context learning.],
  [Characters composed \ of strokes],
  [The hold-out principle finally has substrate, and alphabets give a second,
   coarser generalisation axis to test separately.],
  [Sparse, near-binary ink],
  [Strong content-matching signal, and a collapsed pixel-bin vocabulary — which
   removes the loss imbalance that let pixel cross-entropy (~16 nats) swamp
   label cross-entropy (~0.05) in universal-ar's exp38 and exp39.],
)

#callout(title: [The load-bearing difference])[
  On MNIST, anonymised labels made the task unlearnable by memorisation while
  the data still made memorisation the dominant gradient signal. Chance was the
  predictable outcome, and it said nothing about whether token-level in-context
  binding is achievable. On Omniglot the two agree: memorisation is neither
  possible nor useful.
]
