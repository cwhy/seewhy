#import "/template.typ": *

= The question

#link("https://arxiv.org/abs/2606.25010")[Baherwani et al. (2026)] argue that transformer
capabilities appear abruptly, at training steps that vary between otherwise identical runs,
because each capability needs one *sparse, task-relevant attention pattern* and finding it by
gradient descent is a search. Until the search succeeds the model sits at the loss of a
uniform guess; when it succeeds, loss falls within a few hundred steps.

They support this with synthetic tasks where the correct attention pattern is known by
construction, and with a striking architectural result: an MLP-Mixer, whose position-mixing
weights are learned directly rather than computed through a softmax, *learns their linear map
task faster than a transformer does*. If sparse patterns are hard to #emph[find], an
architecture that does not have to search should not suffer the plateau.

This report is about that architectural claim, and about how far it reaches.

#callout(title: [The result])[
  The mixer's advantage is real, but it is a property of the #emph[task family], not of
  mixers. On the paper's positional tasks the ranking is mixer > transformer > KDA. On a
  content-keyed task — the kind the paper's motivating capabilities actually require — the
  ranking *inverts completely*: KDA > transformer > mixer, and the mixer cannot do the task
  at all, at any learning rate, depth, or budget we tried.

  The axis that explains both orderings is whether a model's position-mixing is *conditioned
  on content*.
]

== Why this matters for the paper

The capabilities the paper sets out to explain — indirect object identification, induction,
copying — are all *content-keyed*: which earlier position matters depends on what the tokens
are, and it changes from sequence to sequence. Both of the paper's synthetic tasks are
*position-keyed*: the correct routing is the same set of slots in every sequence.

So any ranking of architectures derived from those tasks describes positional routing only.
Our measurements say that ranking does not survive the move to content-keyed routing — it
reverses. The paper's central claim about emergence #emph[does] survive, which makes the
contrast worth stating precisely rather than as a general warning.

== What was run

Everything here is 16 seeds per configuration (8 on the cellular-automata task), trained
simultaneously under one `jax.vmap` over a leading parameter axis, on a single RTX 4090 pair.
Every architecture gets a learning-rate sweep, because three separate wrong conclusions in this
project came from judging an architecture at one untuned setting.

#kv(
  ("linear map", "S=16, sparsity s in 3..8, 16 seeds, 10k steps, 2 learning rates"),
  ("induction", "32 pairs, vocab 32, 16 seeds, 30k steps, 2 learning rates, 2 and 4 layers"),
  ("cellular automata", "S=16, T=16, C=4, k=1, 8 seeds, 10k steps, pool size 1..fresh"),
  ("architectures", "transformer, causal MLP-Mixer, KDA linear attention"),
)
