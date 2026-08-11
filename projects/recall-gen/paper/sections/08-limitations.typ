#import "/template.typ": *

// OBLIGATIONS
//  - Its own section, not a hedging sentence.
//  - What was tried and FAILED, including things abandoned before they reached
//    results.jsonl — that is the part a reader cannot reconstruct from the repo.
//  - Where the result does not generalise, specifically.

= Limitations and negative results

== The metric rewards hedging

Mean squared error on pixels is minimised by the conditional *mean* of the
plausible completions, not by any particular one. A model that is uncertain
between a 4 and a 9 scores better by drawing a blur of both than by committing
to either. Every number in §6 inherits this: a low nMSE is evidence of a good
*average*, not of a good image.

We mitigate but do not remove it. Identification accuracy is measured against
the specific context image on hidden pixels, so hedging cannot score there — but
it is only defined where the target is present, which is exactly the conditions
that are not in question. On the absent-target conditions the reader has only
nMSE and the completion grids, and the grids show plainly that the
recall-trained model's absent-target output is not a plausible digit. The honest
statement is that its numerical advantage over the mean-image prior is larger
than its visual one.

A likelihood-based objective would avoid this. We did not use one: it changes
the training objective, and the whole design depends on recall and completion
being trained under identical losses.

== Scope

/ One dataset: MNIST only. MNIST digits are unusually predictable from their top
  halves and unusually well aligned, which flatters both retrieval and
  completion. Nothing here has been checked on a dataset where the two are
  harder.
/ One mask: the bottom 14 rows, always. Random-pixel masking was not run. It
  would make retrieval easier (surviving pixels nearly determine identity) and
  completion much easier (neighbouring pixels are visible), and could move the
  balance between the two substantially.
/ One architecture: a single linear recurrent model. Attention was deliberately
  not tested, because a fixed-size state is the mechanism under study — but that
  means the finding is about *compressive* memories, not about in-context
  learning in general. A model that can re-read its context is not covered by
  anything here.
/ Training length: 12 000 steps. Several curves in §6 had not flattened.

== The ceiling is contaminated, and we chose how to quote it

The completion-trained runs do not learn to complete in the way we wanted them
to. They memorise the 60 000 training images into their weights: their score on
seen-context episodes is an order of magnitude better than on novel ones, and
their novel-image score *worsens* monotonically after roughly step 1 000.

We therefore quote the ceiling as the best value over training on the
novel-image condition — an early-stopped number. This is a defensible choice and
also a choice: the final-step number is much worse, and a differently regularised
run might be better than either. Any comparison in this paper against "the
ceiling" should be read with that in mind.

We did not fix the overfitting. Data augmentation would change the data
distribution and break the comparison with the recall arm; a larger pool was not
available within MNIST.

== What the M-sweep cannot separate on its own

Raising $M$ does two things at once: it overruns the memory, and it puts more
digits in the context. The state-size sweep exists precisely because the M-sweep
alone cannot tell those apart, and it is the load-bearing control for the
paper's main claim. Note that it changes memory shape at fixed parameter count
and fixed model width, so it is a clean capacity manipulation — but it is still
one axis, at one context size.

== Things that were tried and abandoned

/ Rendering model output directly: the first completion grids showed the model's
  raw 784-pixel output, which made every completion look broken. The head emits
  all 784 pixels but the loss scores only the hidden ones, so the visible half is
  unconstrained noise. Figures now paste the prediction into the query's visible
  half. No result changed; the figures had been unreadable.
/ Small batches: the token scan is short enough to be dominated by kernel-launch
  overhead, so an early configuration at batch 64 was doing a quarter of the work
  for 80% of the cost. Nothing was wrong with it, but it made the sweep
  unaffordable and was replaced before any reported run.

== Where this does not generalise

The claim is about a model whose memory of the context is a fixed-size state
that cannot be re-read, trained on a task where retrieval and generalisation are
mutually exclusive by construction. Real in-context learning has neither
property: contexts are re-readable under attention, and a query's answer is
rarely either fully present or fully absent. We take the result as evidence
about a mechanism, not as a claim about what language models do.
