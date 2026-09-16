#import "/template.typ": *

= Results

Chance is 10% throughout (section 3). Every accuracy is a mean over five seeds
for the headline comparison and three for the sweeps, with the standard
deviation across seeds beside it. Wall-clock figures exclude XLA compilation,
which is 0.7-3.0 s depending on arm and is paid once.

== The table replicates

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right),
  [Arm], [Params], [Accuracy], [Posted], [Train (s)], [Posted], [10k images (s)],
  [backprop, 32x32 residual], [56,128], [0.8704 ± 0.0100], [0.8739], [0.199], [0.324], [0.00025],
  [PC-ALM, 32x32 residual], [56,128], [0.8533 ± 0.0085], [0.8628], [8.123], [13.485], [0.00025],
  [Forward-Forward, 2x66], [56,232], [0.8869 ± 0.0085], [0.8883], [0.042], [0.549], [0.00129],
  [PC (control), 32x32 residual], [56,128], [0.6125 ± 0.0403], [--], [8.133], [--], [0.00023],
  [EGGROLL, 32x32 residual], [56,128], [0.4473 ± 0.0695], [--], [1.981#super[†]], [--], [0.00025#super[‡]],
  [backprop, 2x66 (control)], [56,902], [0.9116 ± 0.0058], [--], [0.037], [--], [0.00016],
  [Forward-Forward, 31x32 (control)], [56,800], [0.5751 ± 0.0744], [--], [0.324], [--], [0.00177],
)

#super[†] EGGROLL's cells were measured while an unrelated job shared the GPU.
§6.8 calibrates that at 2.0x on training, so one EGGROLL epoch is about 0.99 s
on an idle device.

#super[‡] Classification for this arm is one forward pass of the same residual
network backpropagation uses, so its inference cost is backpropagation's by
construction. The quoted 0.00025 s is the measurement from the EGGROLL cells
that happened to run while the co-tenant was quiet (exp18); the cells behind
this row measured 0.00210 s for the identical computation. §6.8 explains why no
inference number from a shared device is worth reading.

EGGROLL is in this table at the shared default rate of 1e-3 and is therefore
barely above the 0.4 mark; §6.3 gives it its own rate, where it reaches 0.9025.
Its population is fixed at 512 antithetic pairs throughout (§6.6).

All three posted accuracies fall inside our seed distribution: 0.8739 is 0.35
standard deviations above our backpropagation mean, 0.8628 is 1.1 above our
PC-ALM mean, and 0.8883 is 0.16 above our Forward-Forward mean. The parameter
counts match exactly, which is what identified the architectures in the first
place.

The cost ratios replicate too. PC-ALM trains 40.8x slower than backpropagation
here against 41.6x in the original, on the same architecture and the same
inference budget of $T = 64$ settling steps per weight update. Forward-Forward's
classification costs 5.2x backpropagation's against 3.4x in the original — same
direction, larger factor, and the mechanism is not in doubt: it is ten forward
passes instead of one.

#fig(include "/figures/accuracy_vs_tweet.typ",
  caption: [One-epoch test accuracy for each arm, our five-seed mean beside the
    posted number where one exists. The three arms with a posted number
    replicate; the rest are arms the original table does not have.])

Absolute times are ours and are not comparable to the original's: everything
here is one RTX 4090 with the epoch compiled into a single scanned loop, and our
absolute numbers are 1.6x to 90x faster across the arms. The ratios between arms
are the comparable quantity, and those hold.

== Backprop on Forward-Forward's architecture

The two rows the original table does not contain are the ones that separate the
learning rule from the shape of the network it was run on.

Backpropagation on Forward-Forward's own 784-66-66 stack reaches 0.9116 ±
0.0058, which is 2.5 points above Forward-Forward on that same stack and 4.1
points above backpropagation on the deep-narrow residual network. Forward-Forward
moved the other way — onto a 31-layer stack of 32 units, matched in parameters —
falls to 0.5751 ± 0.0744, the worst arm in the table and the one with by far the
largest seed variance.

#fig(include "/figures/ff_readouts.typ",
  caption: [Forward-Forward read out two ways. The goodness sweep is the method's
    own classifier; the probe is a linear softmax layer trained on the same
    detached activations. On the deep stack the features survive and the
    readout does not.])

The deep collapse is a failure of the readout, not of the features. A linear
probe on the 31-layer network's own activations reaches 0.8919 — within two
points of the two-layer network's probe accuracy of 0.9102 — while the goodness
sweep on those same activations gets 0.5751. It is not a learning-rate artifact
either: sweeping Forward-Forward's rate on the deep stack over four values, the
best (3e-3) gives 0.5808.

== One shared learning rate is doing a lot of the work

#fig(include "/figures/learning_rate_grid.typ",
  caption: [One-epoch accuracy against Adam's learning rate, two seeds per point.
    The best rate differs by a factor of thirty across arms, and the arm that
    the shared default of 1e-3 suits best is Forward-Forward. EGGROLL is on this
    grid at its settled population of 512 pairs, and is the one arm still
    climbing at the right-hand edge.])

Each arm was swept over five rates spanning two orders of magnitude, with both
edge peaks bracketed by a sixth point at 0.1. The best rate per arm:
backpropagation on the residual network 3e-2 (0.9424), PC-ALM 1e-2 (0.9286), PC
3e-3 (0.6880), backpropagation on the plain network 3e-3 (0.9151), and
Forward-Forward 1e-3 (0.8906). EGGROLL, swept over the same five rates at 512
pairs, runs 0.1691, 0.4796, 0.7312, 0.8705, 0.9036 — monotone across the whole
grid, and the only arm that has not turned over by 3e-2. It turns over outside
it: 0.8928 at 0.1 and 0.1021 at 0.3.

The shared default of 1e-3 — the value the PC-ALM paper's rule produces at this
cell, and the value the original comparison used for everything — is the best
rate for exactly one arm, and it is Forward-Forward's. It costs backpropagation
7.0 points and PC-ALM 7.5. Forward-Forward is also the most fragile arm: at 1e-2
it collapses to chance (0.0964), where backpropagation is still improving.

Re-running the whole comparison at each arm's own best rate, five seeds each:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [Arm], [Rate], [Accuracy], [vs. shared default],
  [backprop, 32x32 residual], [3e-2], [0.9374 ± 0.0067], [+0.0670],
  [PC-ALM, 32x32 residual], [1e-2], [0.9277 ± 0.0023], [+0.0745],
  [backprop, 2x66], [3e-3], [0.9171 ± 0.0025], [+0.0056],
  [Forward-Forward, 2x66], [1e-3], [0.8869 ± 0.0085], [+0.0000],
  [PC, 32x32 residual], [3e-3], [0.6778 ± 0.0258], [+0.0653],
  [Forward-Forward, 31x32], [3e-3], [0.6106 ± 0.0469], [+0.0354],
  [EGGROLL, 32x32 residual], [3e-2], [0.9025 ± 0.0026], [+0.4552],
  [EGGROLL, 2x66], [1e-2], [0.8272 ± 0.0232], [--],
)

The original table's ordering was Forward-Forward, then backpropagation, then
PC-ALM. Tuned, it is backpropagation, then PC-ALM, then Forward-Forward —
reversed among exactly the three arms the original compares. EGGROLL, which has
no entry in the original table, lands third overall and above every arm that is
not backpropagation or PC-ALM; it also has the largest gap between its
shared-rate and own-rate numbers of any arm here, 45.5 points, because a
gradient estimate assembled from random directions is useless at a step size
chosen for an exact one. Forward-Forward's
own number does not move at all, because its best rate is the default; its five
tuned runs reproduce the earlier five to four decimal places, which also fixes
this arm's run-to-run nondeterminism at zero on this hardware.

#fig(include "/figures/tuned_vs_default.typ",
  caption: [Each arm at the shared default rate and at its own best rate. The
    ordering of the original table does not survive the change.])

== At equal wall-clock

#fig(include "/figures/accuracy_vs_walltime.typ",
  caption: [Test accuracy against cumulative training seconds, ten epochs per
    arm, three seeds. The horizontal axis is what the original table's third
    column measures; the vertical axis is its second.])

Ten epochs at the shared default rate, tracking accuracy against cumulative
training time:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [Arm], [After 1 epoch], [After 10 epochs], [Training time, 10 epochs],
  [backprop, 32x32 residual], [0.871 @ 0.20 s], [0.946], [2.05 s],
  [PC-ALM, 32x32 residual], [0.855 @ 7.68 s], [0.937], [76.90 s],
  [PC, 32x32 residual], [0.611 @ 7.68 s], [0.771], [76.91 s],
  [Forward-Forward, 2x66], [0.892 @ 0.04 s], [0.860], [0.55 s],
  [backprop, 2x66], [0.908 @ 0.04 s], [0.946], [0.55 s],
  [EGGROLL, 32x32 residual#super[†]], [0.902 @ 1.99 s], [0.935], [20.12 s],
)

#super[†] shared GPU again; halve the times to compare with the rows above.

Backpropagation on the residual network passes PC-ALM's one-epoch accuracy
within its second epoch — 0.908 after 0.40 s against PC-ALM's 0.855 after
7.68 s — and by the time PC-ALM has finished one epoch, backpropagation has had
thirty-eight epochs' worth of wall clock. Forward-Forward is the only arm whose
accuracy *falls* with more training: it peaks at 0.901 after two epochs and
decays to 0.860 by ten.

== Depth: the source paper's claim

#fig(include "/figures/accuracy_vs_depth.typ",
  caption: [One-epoch accuracy against depth at width 32, three seeds. PC-ALM
    tracks backpropagation; predictive coding falls away past 8 layers;
    Forward-Forward's goodness readout degrades from the start.])

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: (right, right, right, right, right, right, right),
  [Layers], [backprop], [PC], [PC-ALM], [EGGROLL], [FF], [backprop, plain],
  [2], [0.9054], [0.9055], [0.9049], [0.8990], [--], [--],
  [4], [0.9012], [0.8963], [0.8943], [0.9048], [0.7177], [0.8312],
  [8], [0.8900], [0.8856], [0.8839], [0.8982], [0.6533], [0.8760],
  [16], [0.8830], [0.8043], [0.8628], [0.9011], [0.6021], [0.8687],
  [32], [0.8711], [0.6137], [0.8554], [0.9025], [0.5899], [0.8605],
  [64], [0.8518], [0.5547], [0.8207], [0.9017], [0.5896], [0.5550],
)

At two layers the three local rules on the residual network are
indistinguishable — 0.9054, 0.9055, 0.9049 — which is what the theory predicts:
with one hidden layer there is no credit to propagate. The gap opens at 16
layers and is 24 points wide at 32. PC-ALM stays within 3.1 points of
backpropagation everywhere, including at 64 layers.

EGGROLL is flat: 0.8990, 0.9048, 0.8982, 0.9011, 0.9025, 0.9017 across the
whole range, varying by less than its seed spread. It is the best arm in the
sweep at 64 layers and the worst at two. Read that with the caveat in §8: the
three backprop-derived arms follow the source paper's depth-dependent rate rule
$10^(-3) sqrt(N slash L)$, which shrinks their step as the network deepens,
while EGGROLL was run at a fixed 3e-2. The flatness is real; the crossover is
partly a rate schedule.

#fig(include "/figures/grad_cosine_vs_depth.typ",
  caption: [Cosine between each local rule's weight update and the true
    backpropagation gradient at the same weights, one batch, three seeds.])

The gradient cosines say the same thing about the mechanism. At two layers both
rules reproduce the backpropagation gradient to three decimals (0.999). As depth
grows, PC-ALM's update stays aligned — 0.965, 0.928, 0.903, 0.889, 0.888 from 4
to 64 layers — while predictive coding's drops to 0.70-0.81 and its accuracy
falls with it.

== EGGROLL: a gradient estimate made of random directions

EGGROLL never differentiates the network. Its update is an average of $P$ paired
perturbations, and everything about its behaviour follows from how good an
average of $P$ random directions can be in $d = 56128$ dimensions.

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (right, right, right, right, right),
  [Pairs $P$], [Accuracy], [Cosine to backprop], [$sqrt(P slash d)$], [Train (s)#super[†]],
  [32], [0.7939 ± 0.0142], [0.0228], [0.0239], [0.557],
  [128], [0.8668 ± 0.0026], [0.0448], [0.0478], [0.747],
  [512], [0.9028 ± 0.0031], [0.0939], [0.0955], [1.971],
  [2048], [0.9219 ± 0.0057], [0.1883], [0.1910], [7.337],
)

The measured alignment matches $sqrt(P slash d)$ to within 5% at every
population size, which is what random-projection geometry predicts for an
average of $P$ isotropic directions and is a check that the estimator is doing
what it claims rather than working for some other reason.

#fig(include "/figures/es_cosine_law.typ",
  caption: [EGGROLL's measured cosine to the true gradient against population
    size, beside the $sqrt(P slash d)$ prediction. Neither is a fit.])

The striking part is the first column read against the third. At $P = 512$ the
update EGGROLL applies has a cosine of 0.094 with the gradient backpropagation
would have computed — 5 degrees of alignment out of 90, an update that is
essentially orthogonal to the right answer — and it reaches 0.9028, three points
*above* backpropagation at the rate the original comparison used and within
3.5 points of backpropagation at its best. Section 7 takes that apart.

#fig(include "/figures/es_population.typ",
  caption: [EGGROLL accuracy against population size, with tuned backpropagation
    and chance marked. Cost per epoch grows roughly linearly with $P$ past 128.])

Two further observations. EGGROLL is the one arm whose accuracy does not depend
on depth (§6.5). And it is the one arm that prefers the deep-narrow architecture:
tuned separately on each — the plain network needs a ten-fold smaller
perturbation, $sigma = 3 times 10^(-4)$ against $3 times 10^(-3)$ — it reaches
0.9025 ± 0.0026 on the 32-layer residual network and 0.8272 ± 0.0232 on the
two-layer plain one, reversing the ordering every other arm shows.

== Checking the reimplementation against the authors' own run

Our PC and PC-ALM are written from the paper's equations. The authors publish
the numbers their own code produces for one Fashion-MNIST cell at the same width
and depth, seed 0, one epoch:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  [], [ours (3 seeds)], [theirs], [our cosine], [their cosine],
  [backprop], [0.7770 ± 0.0034], [0.7866], [1.000], [1.000],
  [PC], [0.6340 ± 0.0194], [0.6813], [0.786], [0.604],
  [PC-ALM], [0.7687 ± 0.0046], [0.7775], [0.901], [0.909],
)

Backpropagation and PC-ALM land within one point of their single-run numbers and
PC-ALM's gradient cosine within 0.008. PC is 4.7 points low and its cosine is
higher than theirs, which is the one arm where our implementation and theirs
disagree by more than seed noise would comfortably explain. Substituting their
tabulated activity step $eta_h = 0.235885$ for ours — measured per seed at that
seed's initialisation, landing between 0.2298 and 0.2319 here — changes nothing: PC moves from 0.6340 to 0.6330 and PC-ALM from 0.7687 to 0.7682. PC is
also the highest-variance arm in every experiment here (± 4.0 points at the
MNIST headline cell), and their figure is one run.


== A timing calibration, because the GPU stopped being ours alone

Every EGGROLL cell was measured while an unrelated job held about 10 GB and a
share of the same device; every earlier cell had the GPU to itself. Rather than
mix the two silently, three arms with known idle-GPU timings were re-measured
under the shared conditions:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  [Arm], [Accuracy, shared], [Accuracy, idle], [Train, shared], [Train, idle],
  [backprop, 32x32 residual], [0.8713], [0.8704], [0.400 s], [0.199 s],
  [PC-ALM, 32x32 residual], [0.8553], [0.8533], [17.612 s], [8.123 s],
  [Forward-Forward, 2x66], [0.8920], [0.8869], [0.084 s], [0.042 s],
)

Training costs a factor of 2.01, 2.17 and 2.01 — call it 2.0 — and accuracy is
unchanged, as it must be, since scheduling does not alter arithmetic. EGGROLL's
training times in this paper are the shared-GPU measurements; halving them puts
one EGGROLL epoch at 512 pairs at about 0.99 s against backpropagation's 0.199 s
and PC-ALM's 8.12 s.

Inference behaves differently, and not in a way that admits a calibration
factor at all:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [Arm], [10k images, shared], [10k images, idle], [Ratio],
  [backprop, 32x32 residual], [0.00207 s], [0.00025 s], [8.3x],
  [PC-ALM, 32x32 residual], [0.00211 s], [0.00025 s], [8.4x],
  [Forward-Forward, 2x66], [0.00307 s], [0.00129 s], [2.4x],
)

Those ratios are one sample of a quantity that is not stable. The same forward
pass, on the same network, measured 0.00210 s in the EGGROLL cells of exp13 and
0.00025 s in those of exp18 — an eight-fold spread on identical work, depending
on what the co-tenant happened to be doing. A pass over 10,000 test images
through a 56K-parameter network is a handful of short kernels whose cost is
dominated by getting onto the device at all, so it is at the mercy of whatever
else is scheduled; a training epoch is one long scanned loop, where throughput
dominates and the 2.0x factor holds steady across arms and repeats.

The practical consequence is that no inference number measured on the shared
device is worth reading, and none is used: the inference column for EGGROLL in
§6.1 is backpropagation's own figure for the same forward pass, which is what
that arm costs by construction.
