#import "/template.typ": *

// OBLIGATIONS
//  - The question in plain terms, BEFORE the method.
//  - Explicit contribution list; each item a claim the paper actually supports.

= Introduction

A large language model shown a few worked examples will often solve the next one
correctly. This is called #gloss[in-context learning][solving a task from
examples supplied in the input, with no change to the model's weights], and it
is the property that makes such models useful without retraining.

There is a long-running argument about what is actually happening. One view is
that the model is *learning* from the examples — extracting a rule and applying
it to a case it has not seen. The other is that it is *retrieving* — finding the
example most like the query and reusing its answer. The two are hard to tell
apart in language, because a model that has read a large fraction of the
internet has usually seen something close to whatever you ask.

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

== Contributions

+ *A task that separates retrieval from generalisation cleanly*, with four
  model-free reference strategies computed on the same episodes — including the
  best possible soft look-up from the context, which bounds what any mechanism
  can extract from the context alone (§3, §4).

+ *Retrieval generalises perfectly and for free.* A model trained only to
  retrieve images from its training pool retrieves images it has never seen
  equally well — identification accuracy 1.000 on both, with no measurable
  penalty. The mechanism is content-addressed, not a memorised table (§6).

+ *Retrieval training does not produce completion ability; it consumes it.*
  #todo[headline numbers once exp10/exp11 land]

+ *Generalisation appears exactly when retrieval fails.* Sweeping the context
  size across the point where the memory can no longer hold the context, the
  recall-trained model's ability to complete unseen digits improves — but only
  as, and because, its retrieval collapses. At the largest context it has
  stopped using the context at all, and has converged on the same
  weight-memorised solution that a model trained on completion finds (§6, §7).

+ *A fine-tuning probe* measuring whether the retrieval solution is worth
  anything as a starting point for a model that must generalise (§6).

The overall answer is negative, and specific about it: within this setting, pure
retrieval training buys no generalisation. What looks like generalisation
emerging at large context is the model abandoning the context.
