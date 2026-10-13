#import "/template.typ": *

= Introduction

Every large language model in use today was trained by backpropagation @rumelhart1986. The method
is so dominant that the architectures, the optimisers and the hardware were all built around it. It
asks a lot of the network: every operation must be differentiable, every intermediate value from the
forward pass must be stored until a second, reversed pass uses it, and the whole computation must be
one connected graph. A family of older methods asks for none of that. They estimate the gradient from
forward passes alone, by perturbing the network at random and watching how the loss responds. They have
not been used to train large language models from scratch, for a reason that is simple arithmetic: each
forward pass returns one number, and one number cannot steer tens of millions of parameters. The
alignment between such an estimate and the true gradient shrinks like $sqrt(P slash d)$, where $P$
is the number of passes and $d$ the number of parameters.

In October 2026 Dahal and colleagues reported a method, DUST, that they say changes this
@dahal2026dust. DUST perturbs a layer's *activations* rather than its weights, independently at every
token, so that one forward pass of a transformer evaluates not one random perturbation but about sixteen
thousand. The paper reports that DUST trains a 38-million-parameter language model to the loss that
backpropagation reaches, that it matches or passes it at a thousand draws and one million training tokens,
and that it is a thousand to ten thousand times more efficient than weight-space evolution strategies
(EGGROLL @sarkar2026eggroll), measured by the number of forward passes.

Those are strong claims, and the comparison that carries them has a feature worth stating plainly: the
baseline is the authors' own implementation of EGGROLL on their transformer, and it is not released. This
repository already contains an independent EGGROLL (`local-learning`), written for another purpose and
audited there. That makes the question testable from the outside: *with EGGROLL implemented by someone
else, in a setting fixed before the comparison, does DUST's advantage reproduce?*

We ported the authors' released DUST code to JAX, checked the port against theirs on identical weights,
implemented EGGROLL on the same network, and ran the paper's protocol at the scales two GPUs allow.
Section 2 explains the algorithms for a reader new to them. Sections 3 and 4 give the task and the
methods. Section 5 lists the experiments, section 6 the results, section 7 what they mean, and
section 8 what could make them wrong.

== Contributions

+ *A port that reproduces the published losses.* A JAX implementation of DUST reaches the paper's backprop and DUST
  numbers at 1M tokens to within 0.03 nats, inside the spread across seeds, and its gradient estimate agrees with the
  authors' PyTorch code layer by layer on identical weights.
+ *DUST's advantage over EGGROLL reproduces on an independent baseline.* At 1M tokens DUST with 64 draws scores 6.315 against
  7.133 for EGGROLL with 1,024 forward passes, at about a fifteenth of the measured cost, and DUST at 1,024 draws, 5.939,
  matches our backprop, 5.937, under SGD.
+ *The cause is measured.* At 16,384 draws the cosine between DUST's estimate and backprop's gradient is 0.99 on the head
  and 0.73 on attention outputs; EGGROLL's is 0.024 and 0.010. EGGROLL's values match the prediction for random-direction
  estimation to within 0.002.
+ *The order of magnitude of the efficiency claim reproduces, with a wide interval.* Extrapolating our EGGROLL's ladder to the
  population at which it would match DUST at 64 draws gives thousands to tens of thousands of times 64 by two line fits, as in the
  paper, and never by a power law whose floor lies above DUST's loss. A three-point version of the same line gave hundreds; the
  fourth point, 4,096 passes, moved it by a factor of six to seventeen. We report that as a warning about the method.
+ *Our EGGROLL is weaker than the paper's, and we could not say why.* It trails by 0.34 nats at 64 passes and 0.08 at 4,096. No
  setting of the Adam rate, noise scale, Adam constants, fitness shaping or precision that we tried closes the gap.
+ *A better optimiser moves DUST more than EGGROLL, and backprop most.* Under Muon DUST at 64 draws falls from 6.315 to 5.864 and at 256 draws from 6.074 to 5.617;
  backprop falls 0.58 nats, to 5.358 under Adam; EGGROLL is not helped by Muon. DUST at 1,024 draws, level with backprop under SGD, is 0.095 nats behind it
  under Muon, and 0.036 behind it at 4,096 draws on one seed. DUST stays ahead of EGGROLL by 1.30 to 1.95 nats within each optimiser.
