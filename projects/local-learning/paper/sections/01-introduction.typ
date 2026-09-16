#import "/template.typ": *

= Introduction

In September 2026 a short post compared three ways of training a neural network
on handwritten digits @mayfer2026tweet. All three networks had about 56,000
adjustable numbers in them; the difference was in how those numbers were
adjusted. One used backpropagation, the standard method, which computes exactly
how much each number contributed to the error by working backwards through the
network. The other two were *local* methods, in which each layer updates itself
from information available in its own neighbourhood, with no backward sweep —
an idea taken seriously because the brain plainly does something local, and
backpropagation is not a candidate for what @rao1999predictive. One was PC-ALM,
published the same week by Sakana AI @seely2026pcalm; the other was
Forward-Forward, proposed by Hinton in 2022 @hinton2022ff. We add two arms of
our own: plain predictive coding, which is what PC-ALM modifies and therefore
the only control that isolates what PC-ALM contributes, and EGGROLL
@sarkar2026eggroll, a gradient-free evolution strategy that is the other
well-known way to train a network without backpropagation.

The table showed backpropagation at 87.39% accuracy in 0.324 seconds of
training, PC-ALM at 86.28% in 13.485 seconds, and Forward-Forward at 88.83% in
0.549 seconds. Read straight, it says the local methods are competitive, and
that the newer one is the slow one.

We reproduce all three numbers, and then ask what the table measures. The
parameter counts turn out to identify the architectures exactly, and they are
not the same architecture: 56,128 is a residual network 32 layers deep and 32
units wide, and 56,232 is a plain network of two layers of 66 units. Two of the
rows describe a tall thin network; the third describes a short fat one. Matching
the parameter count across them holds the wrong thing fixed — a comparison of
learning rules in which the architecture also changes cannot tell you which one
moved the number.

== Contributions

+ *The three headline numbers replicate.* Over five seeds we measure
  87.04% ± 1.00 for backpropagation, 85.33% ± 0.85 for PC-ALM and 88.69% ± 0.85
  for Forward-Forward, each within about one standard deviation of the posted
  figure, and we reproduce the 40x training-time gap between PC-ALM and
  backpropagation (41x here). The comparison is sound as arithmetic.

+ *Forward-Forward's advantage is its architecture, not its learning rule —
  but the margin is small.* Backpropagation on Forward-Forward's own two-layer
  network reaches 91.16% ± 0.58 against Forward-Forward's 90.39% ± 0.30 on that
  same network: 0.8 points, inside the seed spread. What separates the two is
  which architecture each is run on, not which wins when they share one.

+ *An audit of our own Forward-Forward implementation withdrew two earlier
  claims.* Sampling negative labels uniformly, as we first did, hands every
  layer above the first a task already solved; with the hard negatives Hinton
  uses, the same arm gains 1.7 points on the two-layer network and 28.2 on the
  deep stack — which is where the "2.5 point" margin and the "Forward-Forward
  collapses at depth" claim above came from. @res-audit reports the audit.

+ *The ranking is an artifact of one shared learning rate.* Sweeping the rate
  per method over two orders of magnitude, the best rate differs by a factor of
  thirty between arms. Re-run at each arm's own best rate, the order of the
  three original rows reverses: backpropagation 93.74% ± 0.67, PC-ALM
  92.77% ± 0.23, Forward-Forward 88.69% ± 0.85.

+ *At equal wall-clock, the gap is not close.* Backpropagation passes PC-ALM's
  one-epoch accuracy after 0.4 seconds of training and reaches 94.6% in the time
  PC-ALM needs for one epoch.

+ *A gradient-free method is competitive here, and is the only arm indifferent
  to depth.* EGGROLL @sarkar2026eggroll, which estimates the gradient from
  random weight perturbations rather than computing it, reaches 90.25% ± 0.26 in
  one epoch — above backpropagation at the rate the original comparison used —
  while its update has a cosine of only 0.094 with the true gradient. Its
  accuracy is flat from 2 to 64 layers where every other rule falls away, and
  the alignment of its estimate follows the $sqrt(P slash d)$ law to within 5%
  across a 64-fold range of population sizes.

+ *The source paper's central claim holds where we can test it.* Standard
  predictive coding — PC-ALM with its dual variables switched off, everything
  else identical — collapses from 90.55% at two layers to 61.37% at 32, while
  PC-ALM holds at 85.54%. The gradient it produces stays aligned with
  backpropagation's (cosine 0.89 at depth 32) where plain predictive coding's
  drifts (0.81).

+ *Every arm was audited against its own paper's claims, and one failed.*
  PC-ALM reproduces its central theorem — in a linear network its update
  converges on the backpropagation gradient, cosine 0.999982 — and matches the
  authors' own code to within half a point when we run theirs beside ours.
  EGGROLL's estimator is unbiased and its low-rank claim holds. Feedback
  alignment reproduces its published result where that result was measured.
  Forward-Forward did not, and fixing it cost us two claims.

+ *An independent reimplementation reproduces the authors' own cell.* On their
  Fashion-MNIST configuration we land within one point of their reported
  backpropagation and PC-ALM accuracies and within 0.01 of their gradient
  cosine, from the paper's equations rather than their code.

Section 2 defines the four learning rules for a reader who has not met them,
section 3 the data, section 4 the exact models and hyperparameters, and section
5 the experiments and controls. Section 6 reports the numbers, section 7 asks
what explains them, and section 8 says where this stops.
