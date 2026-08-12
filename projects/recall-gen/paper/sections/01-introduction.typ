#import "/template.typ": *

// OBLIGATIONS
//  - The question in plain terms, BEFORE the method.
//  - Explicit contribution list; each item a claim the paper actually supports.

= Introduction

A large language model shown a few worked examples will often solve the next one
correctly. This is called #gloss[in-context learning][solving a task from
examples supplied in the input, with no change to the model's weights], and it
is the property that makes such models useful without retraining.

There is a long-running argument about what is happening. One view is that the
model is *learning* from the examples — extracting a rule and applying it to a
case it has not seen. The other is that it is *retrieving* — finding the example
most like the query and reusing its answer. The two are hard to tell apart in
language, because a model that has read a large fraction of the internet has
usually seen something close to whatever you ask.

This paper takes the question somewhere the ambiguity can be removed. We build a
task where retrieval and generalisation are *mutually exclusive by
construction*, train a model on retrieval alone, and ask what it can do.

== The setup in one paragraph

Each item in the model's input is an entire image — a handwritten digit — rather
than a word. An episode presents $M$ complete digit images, then a query image
with its bottom half cut off, and asks for the missing pixels. During training
the query is *always* a copy of one of the $M$ images already in the context, so
the answer is present and the task is pure look-up. At test time we can withhold
it: put the query's true image nowhere in the context, and look-up becomes
impossible. Whatever the model produces then, it produced without anything to
copy.

The model carries the context in a #gloss[linear recurrent state][a fixed-size
memory updated one item at a time, rather than a growing list of items that can
all be re-read] of 16 384 numbers, and query items are barred from writing to
it. That state is the only route from the context to the answer, and it does not
grow with $M$ — so raising $M$ is a way of *forcing* the model out of retrieval
and into whatever else it can do.

== What the answer turns out to be

Stated once, plainly, because the rest of the paper is the evidence for it:

#callout(title: [The finding])[
  Retrieval training produces a *similarity metric*, not knowledge. The metric is
  general enough to work on images the model has never seen, and even on digit
  classes it has never seen — but it is a metric fitted to the training
  distribution, not a distribution-free one, and it carries nothing usable when
  there is nothing to match. Where a recall-trained model *appears* to start
  generalising, it has stopped retrieving.
]

Two things in that statement need guarding against misreading, and both are
measured rather than asserted.

*"General" is bounded.* It is tempting to say a content-addressed mechanism has
nowhere to put a memorised identity and therefore applies to any input. That is
false here. The same model that identifies unseen digits at 1.000 identifies
Fashion-MNIST items at 0.651, and MNIST images under a fixed pixel permutation —
literally the same pixels, with the same statistics and the same pairwise
distances — at 0.116, against a chance level of 0.063. Nothing the model learned
is free of its training data. What differs between the two things it could learn
is *granularity*: individual images, which transfer nowhere, or a similarity
metric over the distribution, which transfers within it.

*"Stopped retrieving" is not a figure of speech.* At a context of 256 images the
recall-trained model scores 0.556 when the answer is present and 0.561 when it is
absent — the same number. At a context of 16 the same two conditions read 0.017
and 0.852.

== Contributions

+ *A task that separates retrieval from generalisation cleanly*, with four
  model-free reference strategies computed on the same episodes — including the
  best possible soft look-up from the context, which bounds what any mechanism
  can extract from the context alone (§3, §4).

+ *A reporting discipline that survives the obvious confound.* Identification
  accuracy, the natural measure, is inflated by models that never retrieve — a
  good completion picks the right neighbour by itself — and its chance level
  moves with the context size. Rather than replace it with a summary statistic,
  we report the two conditions it conflates side by side as measured, so that
  "this model does not read its context" is something the reader sees in two
  numbers rather than takes on trust (§4, §6).

+ *Retrieval training does not produce completion ability; it consumes it.* A
  recall-trained model ends at 0.852 on queries whose answer is absent, against
  0.017 when it is present — and 0.852 is worse than a linear regression that
  ignores the context entirely (0.645). It gets
  worse through training, from 0.635 at step 500, as its retrieval sharpens.
  Training on a half-and-half mixture reaches the full completion ceiling at no
  cost, so the two are not competing for capacity: the recall objective supplies
  no gradient toward completion (§6).

+ *Generalisation appears exactly when retrieval fails.* Across a context sweep,
  completion improves only as retrieval collapses, and at the largest context the
  model has converged on the same weight-memorised solution a completion
  objective finds. Shrinking the *memory* at fixed context reproduces the trade
  on identical episodes, which separates capacity from information (§6, §7).

+ *It is a training-time effect, not an inference-time one.* Evaluated at a
  context of 256, the model trained at 16 does not improve — it degrades to
  0.942, against 0.561 for a model trained there. The two solutions are separate
  attractors that inference-time context size does not move between (§6).

+ *A measurement of how far the learned mechanism travels*, which is what makes
  the first contribution's word "general" honest (§6).
