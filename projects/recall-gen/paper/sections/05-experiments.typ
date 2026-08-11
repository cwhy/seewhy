#import "/template.typ": *

// OBLIGATIONS
//  - A table of run -> question -> source file.
//  - The controls, especially the POSITIVE control: without one, a negative
//    result cannot be told apart from a broken harness.

= Experiments

Every run below trains the same 4.03 M-parameter model on the same data with the
same optimiser for the same 12 000 steps, unless the table says otherwise. The
arms differ in exactly one of three things: what the training target is
(recall / completion / mixed), how many context images an episode has, and which
digits the training pool contains.

#align(center, table(
  columns: (auto, auto, auto, 1fr),
  stroke: none, align: left, inset: 5pt,
  table.hline(stroke: rule),
  [*run*], [*setting*], [*file*], [*question it answers*],
  table.hline(stroke: rule),

  [exp1], [recall, $M=16$], [`experiments1.py`],
  [Does a model trained only to retrieve generalise — to novel images, and to
   targets it cannot retrieve?],

  [exp2], [completion, $M=16$], [`experiments2.py`],
  [How well can this architecture complete a digit at all? Sets the scale exp1
   is read against.],

  [exp3], [mixed, $M=16$], [`experiments3.py`],
  [Is the recall model's completion deficit simply a matter of never being
   asked? Does half a dose of completion signal buy it back, and at what cost to
   recall?],

  [exp6, exp1, exp4, exp5], [recall, $M = 4, 16, 64, 256$],
  [`experiments{6,1,4,5}.py`],
  [The central sweep. The memory is fixed at 16 384 numbers, so raising $M$
   moves the task from "the context fits" to "it cannot possibly fit". What
   happens to retrieval, and to generalisation, across that boundary?],

  [exp7], [completion, $M=64$], [`experiments7.py`],
  [exp4's ceiling. Without it, exp4's absent-target number cannot be told apart
   from "worse than a ceiling measured at a different $M$".],

  [exp8, exp9], [digit split 0--4 / 5--9], [`experiments{8,9}.py`],
  [Does retrieval transfer to digit classes never seen in training, or only to
   novel images of familiar classes? Six conditions rather than four, to tell
   image-novelty and class-novelty apart.],

  [exp13, exp14], [fine-tune vs. scratch], [`experiments{13,14}.py`],
  [Is the recall solution *worth anything* to a model that must generalise?
   2 000 steps of completion training from exp1's weights, against 2 000 steps
   from noise.],

  [exp10, exp11, exp12], [seed replicates], [`experiments{10,11,12}.py`],
  [Seed variance on the two headline configurations.],
  table.hline(stroke: rule),
))

== Controls

*Positive control.* Conditions A and B are the positive control for the whole
apparatus. If the harness, the masking, the state gating or the metric were
broken, a recall-trained model could not reach near-zero error with
identification accuracy 1.000 on episodes whose answer is sitting in the
context. Every recall-trained run in this paper is required to pass that check
before its absent-target numbers are read, and the runs that do not pass it
(exp5, at $M = 256$) are reported as *failing to retrieve* rather than as
generalising well.

*Model-free ceilings.* The four reference strategies of §4 are computed on the
same evaluation episodes as every model. The soft-look-up ceiling is the one
that matters: it bounds what *any* mechanism can extract from the context alone.
A model scoring above it is not using the context; a model scoring below it is
doing something else as well.

*Separating memorisation from generalisation.* Conditions C and D differ only in
whether the context images come from the training pool. A model that has
memorised its training images scores far better on C than on D. This is not a
hypothetical: it is what both completion-trained runs do, and it is why "the
ceiling" in this paper is quoted from D and not from C.

*Shared evaluation episodes.* All runs are scored on the same 512 episodes per
condition, drawn once from a fixed seed. Differences between runs therefore
cannot come from having drawn an easier evaluation set.
