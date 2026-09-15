#import "/template.typ": *

= Methodology

Three learning rules, two architectures, one loss. This section gives all of
them in full; the acceptance test is that a reader could rebuild the experiments
without opening the repository.

== Notation

#notation(
  ($L$, [number of layers, counting the output layer]),
  ($N$, [width: units per hidden layer]),
  ($x$, [one standardised input image, $x in RR^784$]),
  ($y$, [one-hot label, $y in RR^10$]),
  ($z_i$, [activation of layer $i$; $z_0 = x$]),
  ($W_i$, [weight matrix of layer $i$, no bias term]),
  ($phi$, [the ReLU nonlinearity, $phi(u) = max(u, 0)$]),
  ($r_i$, [layer $i$'s prediction error (residual)]),
  ($lambda_i$, [layer $i$'s Lagrange multiplier, the "dual" vector]),
  ($T$, [inference budget: settling steps per weight update]),
  ($eta_h$, [step size used during settling]),
  ($alpha$, [dual step size; $alpha = 0$ turns PC-ALM back into PC]),
  ($g$, [Forward-Forward goodness, the mean square of a layer's activations]),
  ($theta$, [Forward-Forward goodness threshold]),
)

== The residual network

Backpropagation, PC and PC-ALM all train the architecture of the PC-ALM paper
@seely2026pcalm: a residual multilayer perceptron with no biases, fixed
per-layer scale factors, and a skip connection around every hidden block.

$ z_1 &= 1/sqrt(784) W_1 x \
  z_i &= z_(i-1) + 1/sqrt(N L) W_i phi(z_(i-1)) quad & i = 2, ..., L-1 \
  hat(y) &= 1/N W_L phi(z_(L-1)) $

Every entry of every $W_i$ is drawn from a unit normal $cal(N)(0, 1)$, and the
scale factors sit outside the weights rather than being folded into the
initialisation. That is the paper's $gamma_0 = 1$ parameterisation, and it is
what makes one optimiser step mean the same thing at depth 8 and at depth 64.

The loss is one-half squared error against the one-hot label, summed over the
ten outputs and averaged over the batch $cal(B)$:

$ cal(L)_"sup" = 1/(2|cal(B)|) sum_(b in cal(B)) sum_(c=1)^(10) (hat(y)_(b c) - y_(b c))^2. $

Squared error rather than cross-entropy, again following the paper: the
predictive-coding formulation needs the output edge to be a Gaussian prediction
error like every other edge.

At $N = 32$, $L = 32$ this network has $784 times 32 + 30 times 32^2 + 10 times 32
= 56128$ weights.

== Backpropagation

The gradient $nabla_(W_i) cal(L)_"sup"$, computed by reverse-mode automatic
differentiation, fed to Adam @kingma2015adam.

== Predictive coding

Write the forward pass as a constraint rather than a computation. The
activations become free variables, and the network must satisfy

$ z_i = "block"_i (z_(i-1)) quad "for every hidden" i $

while minimising $cal(L)_"sup"$. Relaxing each constraint into a quadratic
penalty gives the free energy

$ F(z, W) = cal(L)_"sup" + 1/2 sum_(i=1)^(L-1) norm(r_i)^2, quad
  r_i = z_i - "block"_i (z_(i-1)). $

Training alternates two phases per batch. *Inference*: initialise $z$ at its
forward-pass values, then take $T$ gradient steps $z_i <- z_i - eta_h
nabla_(z_i) F$. Because $F$ couples $z_i$ only to $z_(i-1)$ and $z_(i+1)$, each
step is a message between neighbouring layers — no global pass. *Learning*: one
Adam step on $nabla_(W_i) F$ with $z$ frozen.

== PC-ALM

Identical, with the quadratic penalty replaced by an augmented Lagrangian
@seely2026pcalm:

$ cal(L)(z, W, lambda) = cal(L)_"sup" + sum_(i=1)^(L-1) lambda_i^top r_i
  + rho/2 sum_(i=1)^(L-1) norm(r_i)^2, $

and a dual ascent step interleaved into inference:

$ z_i &<- z_i - eta_h nabla_(z_i) cal(L) quad & "(primal)" \
  lambda_i &<- lambda_i + alpha r_i & "(dual)" $

Each $lambda_i$ accumulates its own layer's prediction error over the settling
steps. In a deep *linear* network the settled $lambda_i$ equals that layer's
backpropagation credit signal exactly @seely2026pcalm @lecun1988theoretical;
whether that carries over to ReLU networks is an empirical question, and one
this paper measures directly by taking the cosine between the resulting weight
update and the true backpropagation gradient.

Setting $alpha = 0$ leaves the duals at zero forever and recovers PC exactly.
PC is therefore the control that isolates what the dual variables contribute,
and it is run at identical $T$, $eta_h$, seed and architecture.

Three implementation details are needed to reproduce the paper's numbers and are
not in its equations; all three follow the authors' released reference
implementation, and we adopted them after implementing from the text alone.

+ The energy above is a batch mean, so a settling step of $eta_h$ on the
  per-sample energy is a step of $eta_h |cal(B)|$ on the batch mean. Without
  that rescale, $T$ steps of inference barely move.
+ The weight update uses the duals from *before* the final dual ascent step.
+ $eta_h = 1 slash lambda_max$, where $lambda_max$ is the largest eigenvalue of
  the Hessian of the inference energy with respect to the activations. The
  authors ship a lookup table of these per architecture; we measure it instead,
  by 60 steps of power iteration on Hessian-vector products at initialisation.
  At the headline cell we measure $lambda_max = 4.33$ against their tabulated
  $4.24$, a $2%$ difference in the resulting step size.

== Forward-Forward

Forward-Forward @hinton2022ff runs on a plain stack: dense layers with biases,
ReLU, no skips, no output layer. Layer $i$ computes

$ h_i = phi(W_i hat(h)_(i-1) + b_i), quad
  hat(h) = h / (sqrt("mean"(h^2)) + epsilon), quad hat(h)_0 = x, $

with $epsilon = 10^(-4)$ and each $hat(h)_(i-1)$ detached from the gradient, which
is what makes training layer-local. The normalisation is the load-bearing part
of the method: it strips the magnitude — which *is* the previous layer's
answer — and leaves only the direction, so a layer cannot win by inheriting its
predecessor's goodness. Hinton divides by the L2 norm; we divide by the root
mean square, which is the same direction rescaled by a constant $sqrt(d)$ that
the weights would otherwise have to learn.

Goodness is the mean square of a layer's activations, $g(h) = "mean"(h^2)$, and
each layer minimises

$ cal(L)_i = "softplus"(-(g(h_i^+) - theta)) + "softplus"(g(h_i^-) - theta) $

where $h^+$ is the layer's response to an image carrying its correct label and
$h^-$ its response to the same image carrying a uniformly sampled wrong label.
All layers train simultaneously on the same batch, so one epoch is one pass over
the data, not one pass per layer.

There is no output layer to read at test time. The image is run through the
network ten times, once per candidate label, and the label maximising
$sum_(i >= 2) g(h_i)$ wins. Layer 1 is excluded from that sum because it sees the
label pixels directly. This is why Forward-Forward's inference is the most
expensive of the three: ten forward passes per image.

A linear softmax probe on the detached activations of layers $2..n$ is trained
alongside, and reported as a secondary readout. Its 670 parameters are counted
separately and are not included in the network's parameter count.

At two hidden layers of 66 units the FF network has $784 times 66 + 66 + 66^2 +
66 = 56232$ weights. Adding a ten-way output layer for the backpropagation
control brings that to 56,902.

== EGGROLL

EGGROLL @sarkar2026eggroll never differentiates anything. It estimates the
gradient of the same loss from paired perturbations of the weights:

$ epsilon_i &= sigma dot A_i B_i^top, quad A_i in RR^(m times r), B_i in RR^(n times r),
  quad (A_i)_(j k), (B_i)_(j k) ~ cal(N)(0, 1)   f_i &= cal(L)_"sup" (theta + epsilon_i) - cal(L)_"sup" (theta - epsilon_i)   hat(g) &= 1/(P sigma^2) sum_(i=1)^P f_i epsilon_i $

with $P$ antithetic pairs per weight update. $hat(g)$ is an unbiased estimate of
the true gradient up to a constant that Adam absorbs, and is then fed to Adam
exactly as a real gradient would be. Bias vectors, having no matrix structure,
get an ordinary Gaussian perturbation.

The low-rank part is the paper's contribution and is what makes this cheap on a
GPU: each $epsilon_i$ needs $(m + n) r$ random numbers rather than $m n$, and a
whole population of them runs as one batched matrix multiply. Each individual
perturbation has rank $r$, but the update is a weighted sum of $P$ of them, so
the aggregate update has rank up to $P r$. We use $r = 1$ throughout.

The cost per weight update is $2 P$ forward passes and no backward pass. Note
what this arm is *not*: it is not layer-local, and it is not biologically
motivated. It is here because it is the other well-known way to train a network
without backpropagation, and because — unlike PC, PC-ALM and Forward-Forward —
it is completely indifferent to the architecture it is applied to, which makes
it the cleanest test of how much the architecture in this comparison matters.

Our implementation is a port of the one in this repository's `es` project, which
had already been validated on MNIST; the estimator is unchanged, and the loss it
perturbs is whichever loss the arm under test uses, so the quantity being
estimated is exactly the gradient backprop computes.

== Training setup

#table(
  columns: (auto, 1fr),
  [optimiser], [Adam @kingma2015adam, $beta_1 = 0.9$, $beta_2 = 0.999$, $epsilon = 10^(-8)$],
  [learning rate], [$10^(-3) sqrt(N slash L)$ for the residual network (the paper's rule; $10^(-3)$ at the headline cell), $10^(-3)$ for the plain network],
  [batch size], [64],
  [epochs], [1 for the headline comparison; 10 for the wall-clock curves],
  [inference budget], [$T = 2L$ (64 at the headline cell), for PC and PC-ALM],
  [$alpha$, $rho$], [1.0, 1.0],
  [$eta_h$], [$1 slash lambda_max$, measured per cell by power iteration],
  [$theta$], [2.0],
  [seeds], [5 for the headline comparison, 3 for the sweeps, 2 for the learning-rate grid],
  [precision], [float32 throughout],
  [hardware], [one NVIDIA RTX 4090],
)

Every number reported below is a mean over seeds, with the spread given
alongside; single runs are labelled as such. A seed sets the weight
initialisation, the batch order and Forward-Forward's negative labels.

Timings need their own definition, because the comparison being replicated is
half a timing comparison. Each epoch is compiled once by XLA into a single
scanned loop, so no method is charged for Python dispatch that another avoids.
We then report three separate quantities: compilation time (a few seconds,
excluded from everything else and reported once), *training time* (the compiled
epochs, blocked on until the GPU is idle), and *inference time* (one pass over
all 10,000 test images, fastest of three, after compilation). All three exclude
data loading, which is identical across methods.
