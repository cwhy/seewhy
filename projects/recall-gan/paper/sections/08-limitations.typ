#import "/template.typ": *

= Limitations <sec:limits>

*The control is confounded.* Under the method's recipe every blind generator
collapsed to a single image (12 of 12). The comparison with the context-aware
generator therefore mixes two differences: whether the generator sees the
context, and whether it produces varied images at all. The result shows that a
context-aware generator makes the discriminator use its state; it does not
show that a varied but blind generator would fail to.

*One dataset, one context size.* All results are on MNIST with 16-image
contexts and a 16 384-number state. Whether the state still separates context
images from unseen ones when the context approaches or exceeds its capacity,
or on images more varied than digits, is untested.

*Class-level, not item-level, membership.* The discriminator ranks an unseen
image of a digit in the context well above one whose digit is absent, but
still lets 19% of unseen images score above 0.5. It is a coarse estimate of
"belongs with this context", not a test of "is exactly one of these images".

*The scores are not calibrated.* They saturate at a ceiling and sometimes
reject unseen images of a class the context contains. Using the raw score as a
probability would be wrong.

*Reliability is measured, not proven.* No run failed out of 12 with the
method's recipe, against one out of 12 with the common Adam setting. Twelve
runs cannot rule out a small failure rate.

*The generator learns an imperfect mapping.* Given a context of fours, the
representative generator draws sevens and nines. Generation quality is not
what the method optimises, and it has not been measured beyond the probes
here.

*Baselines are in pixel space only.* The digit-classifier comparison is
against nearest-neighbour and RBF density on raw pixels. A learned
representation, such as a trained image encoder, would be a stronger baseline.
