#import "/template.typ": *

= Introduction

Suppose a dataset is not a collection of samples but a flat bag of tokens, each
one a triple of #m[position], #m[value], and a #m[ref] tag that says which
sample it belongs to. Then classification, inpainting and denoising stop being
different problems: each is the same operation, completing a masked token's
value given its address. Classification is completing the token whose address is
the label.

That premise is attractive and has an obvious test. Give the model a handful of
labelled examples in its context, re-randomise which label means which class on
every episode so that nothing can be memorised, and ask it to classify a new
example. If the premise holds, the model should read the labelling off its own
context.

Prior work applied this to MNIST and got chance, across six architectural
interventions. This paper asks whether that was a fact about the formulation or
a fact about MNIST, and answers it on Omniglot — a dataset built to be MNIST's
transfer-learning inverse, with 1623 characters, only 20 drawings of each, and a
standard split whose train and test character inventories are *disjoint*. On
Omniglot memorisation is impossible by construction rather than by an
anonymisation trick, and reading the support set is the only available strategy.

== Contributions

+ A token-level in-context classification setup on Omniglot in which every
  known shortcut is closed by construction, with a nearest-neighbour floor
  computed on exactly the pixels the model observes (§#link(<sec:method>)[4]).
+ Evidence that the substrate was not the obstacle: the setup that fails on
  MNIST also fails on Omniglot, and continues to fail when the task is made
  deliberately easy (§#link(<sec:exp>)[5]).
+ A mechanistic account of *why*, derived from the failure's signature rather
  than assumed in advance: the token layout forces a three-hop circuit whose
  middle hop asks softmax attention to preserve information it necessarily
  averages away (§#link(<sec:analysis>)[6]).
+ Interventions that follow from that account, and what they do and do not fix.

== A note on what this paper reports

Most of the numbers here are negative. They are reported as they came out,
against floors fixed in advance, because the informative content of this line of
work is in *which* explanations the failures rule out. Where a result is inside
noise it is labelled as such, with the standard error, rather than described as
a trend.
