#import "/template.typ": *

= Introduction

A neural network that has just been created, before it has seen any data, is
usually thought of as useless. Its weights are random numbers. Training is the
process that turns those random numbers into something that computes.

This paper is a replication of a result that complicates that picture. Zhong and
Andreas @zhong2024random asked what happens if you take a randomly initialised
#gloss[transformer][the neural network architecture behind modern language
models] and refuse to change almost any of it. Specifically: freeze every weight in the network's interior, permanently, at
its random initial value. Training may adjust only the parts that translate
between symbols and vectors, at the very edges of the model.

Concretely, a transformer that reads a sequence of symbols does three things.
First it looks up a vector for each input symbol — the #gloss[embedding][a lookup
table mapping each symbol to a vector of numbers]. Then it passes those vectors
through a stack of layers that mix and transform them. Finally it maps the
result back to a score for each possible output symbol, the *unembedding*. The
lookup tables at either end are a small fraction of the parameters; the stack in
the middle is nearly all of them.

Their experiment trains the lookup tables and freezes the stack. Nothing inside
the model can adapt to the task. Training can only choose *how to write the
problem into* a fixed random function, and *how to read an answer back out*.

The striking claim is that this is often enough. On modular arithmetic, on
retrieving a value from a long list, on multi-digit addition, on checking
whether parentheses are balanced, models trained this way reach perfect
accuracy. If that holds, then the algorithm was in some sense already available
in the random network, and training's job was only to find an encoding that
reaches it.

#callout(title: [Why this would matter])[
  Trained models are found to contain interpretable circuits. One does modular
  arithmetic; another copies a repeated token. The natural reading is that
  gradient descent *built* those circuits from the training signal. If a network with a frozen random interior does the same
  task, that reading is at least incomplete. Some of the structure would be
  attributable to the architecture and its initialisation, and would be present
  before the first gradient step.
]

We reimplemented the paper from its text in JAX, without consulting the
authors' code, and reran its seven experiments at a smaller scale. The point of
reimplementing rather than rerunning is that agreement then says something about
the paper rather than about a shared codebase.

== Contributions

- We confirm the paper's central result. On all four algorithmic tasks, random
  transformers of width 1024 reach the accuracies the paper reports, from an
  independent implementation.

- We confirm that this is a property of *width*, not of the training procedure:
  the width sweep separates random from fully trained models sharply at small
  widths and closes at large ones.

- We confirm the paper's negative results: random transformers store far less
  arbitrary information per trainable parameter than trained ones, and are much
  worse language models.

- We confirm the mechanism the paper proposes — subspace selection — and its
  distinction from sparsification, and we run the paper's own falsification
  test for it.

- We report where our numbers differ from the paper's, and identify an internal
  inconsistency in the paper's parameter-count table for the modular-addition
  task.

== How to read this paper

@sec-background defines the machinery, for a reader who has not trained a
neural network. @sec-task and @sec-methodology give the tasks and the setup in
enough detail to reimplement. @sec-experiments lists what was run and why,
@sec-results reports what happened, and @sec-analysis asks whether the proposed
mechanism survives. @sec-limitations is where the things that did not work
live.
