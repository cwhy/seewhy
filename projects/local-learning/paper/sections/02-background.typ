#import "/template.typ": *

= Background

A neural network of the kind used here is a stack of #gloss[layers][a matrix
multiplication followed by an elementwise nonlinearity]. Each layer turns a
vector of numbers into another vector; the last one produces ten numbers, the
#gloss[logits][one score per class, the largest of which is the prediction],
one per digit. Training means adjusting the matrices — the #gloss[weights][the
learnable numbers, here about 56,000 of them] — so that the largest logit is
usually the right digit.

To adjust a weight you need to know whether increasing it would help or hurt.
That quantity is the weight's gradient, and the standard way to compute it is
#gloss[backpropagation][the chain rule applied backwards through the stack]: run
the input forwards to get a loss, then propagate the derivative of that loss
back down through every layer. It is exact and cheap. It also has a property
that is awkward if you take the network as a model of the brain: nothing can be
updated until the forward pass has finished and the backward pass has come all
the way back. Every layer must hold still and wait. No mechanism for that kind
of global timekeeping is known in cortex, which is the standing objection to
backpropagation as a biological theory @rao1999predictive.

That objection motivates *local* learning rules, where a layer updates itself
from quantities available in its own neighbourhood. Two such rules are compared
here against backpropagation.

*Predictive coding* (PC) treats the forward pass not as a computation but as a
constraint to be satisfied @rao1999predictive @whittington2017approximation.
Each layer's activation is a free variable; the network settles them by
iteratively reducing the mismatch between what each layer holds and what its
neighbour predicts, and only then updates its weights. The settling is a
dynamical system running on nearest-neighbour messages, with no backward pass,
and at convergence in some regimes it approximates the backpropagation gradient
@whittington2017approximation. Settling the activations before touching the weights has an advantage of its own
— the network reconfigures itself towards the target first, which can make it
more sample-efficient than backpropagation @song2024prospective. Its weakness is
that supervision enters only at the output and has to work its way down through
a chain of local compromises, so in deep, narrow networks the signal is faint by
the time it reaches the early layers.

*PC-ALM* @seely2026pcalm is predictive coding with one addition: every layer
carries a second vector, a #gloss[Lagrange multiplier][an auxiliary variable
that accumulates a constraint violation and pushes back in proportion], which
accumulates that layer's mismatch over the settling steps. LeCun showed in 1988
that the multipliers of a network written as a constrained optimisation problem
*are* its backpropagation credit signals @lecun1988theoretical; PC-ALM is the
observation that letting local dynamics accumulate them recovers those signals
without a backward pass. The paper's headline claim is that this trains residual
networks up to 1000 layers, which standard PC cannot.

*Forward-Forward* (FF) @hinton2022ff discards the global loss entirely. Each
layer has its own objective: be strongly active on real data and weakly active
on fake data. The label is written into the input image, so "real" means the
image carries its own label and "fake" means it carries someone else's, and a
layer that can tell the two apart has learned something about how digits and
labels go together. There is no backward pass and no settling; there is also no
output layer, so classification is done by trying all ten labels and keeping
whichever the network is most enthusiastic about.

All three rules are compared here at a fixed budget of one #gloss[epoch][one
pass through the 60,000 training images], the budget under which the numbers
being replicated were produced.

== Related work

The PC-ALM paper @seely2026pcalm evaluates against standard PC on a width-depth
grid of residual multilayer perceptrons and on ResNet-18, and reports that
PC-ALM tracks backpropagation across the grid while PC falls away in the deep,
narrow corner. It does not compare against Forward-Forward. Hinton's report
@hinton2022ff evaluates Forward-Forward on MNIST and CIFAR-10 with networks of
two to four wide layers, and does not test it at depth.

The comparison replicated here @mayfer2026tweet is, as far as we know, the first
public head-to-head of the two local rules and backpropagation at a matched
parameter count. Matching parameters across methods whose architectures differ
is precisely where it becomes interesting, and is what this paper takes apart.
