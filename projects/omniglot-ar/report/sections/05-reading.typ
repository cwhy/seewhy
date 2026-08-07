#import "/template.typ": *

= How to read this

== The bars, in order

+ *Above chance* is necessary and nowhere near sufficient.
+ *Above pixel 1-NN on the same observed pixels* is the real bar. Below it, the
  model has learned less than a cosine distance.
+ *Unseen ≈ seen* is what in-context learning looks like. A large gap in favour
  of seen characters means the weights are carrying class knowledge that the
  episode was supposed to supply.

== What this design deliberately does not test

- *Full images.* Each drawing exposes a random 196-pixel subset of 784. The
  pixel 1-NN baseline is computed on the same subset, so the comparison is fair
  — but the absolute numbers are not comparable to published Omniglot results,
  which use whole images.
- *Convolutional priors.* Positions are learned embeddings with no spatial
  structure, inherited from universal-ar. Two adjacent pixels are as unrelated
  as two distant ones until the model learns otherwise.
- *Stroke structure.* Omniglot's compositional stroke data is not used; only the
  rendered bitmaps are. The "bind parts into a whole" principle is tested only
  implicitly.
- *Alphabet-level generalisation*, which needs the exp4 split.

== A note on the pixel-bin vocabulary

Ink covers about 19% of a drawing, so most tokens carry bin 0. Two unrelated
drawings therefore agree on the vast majority of their observed positions, and
the agreement signal that a matching circuit would need to read is dominated by
uninformative background agreement. This is a plausible obstacle independent of
the substrate argument, and it is what the exp5 ablations should isolate:
binarising the values, or restricting the observed pool to positions with ink
somewhere in the episode, would both sharpen it.

== Next

`proposal.md` planned an N-way × K-shot sweep next. That is now pointless:
sweeping episode shape cannot help when the easiest shape available is already
at chance — which is what exp2 was run to find out, and did.

The ablations move to the front, and the pixel-bin obstacle above is the first
one to test, because it is the only identified mechanism that would break
position-wise matching specifically while leaving every other component working
(which is what the results show):

+ *Binarise the values.* Two bins instead of eight, so a match on ink is not
  diluted across intensity levels.
+ *Draw the observed pool from positions carrying ink somewhere in the
  episode*, rather than uniformly over all 784. This changes what "agreement"
  measures from mostly-background to mostly-stroke.
+ *Then* revisit depth and width, and only then the sweep.

Testing the "one mechanism, many tasks" claim (exp3, masked-pixel completion
alongside classification) should wait until classification works at all.
