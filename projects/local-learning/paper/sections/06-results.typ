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
  [backprop, 2x66 (control)], [56,902], [0.9116 ± 0.0058], [--], [0.037], [--], [0.00016],
  [Forward-Forward, 31x32 (control)], [56,800], [0.5751 ± 0.0744], [--], [0.324], [--], [0.00177],
)

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
    replicate; the three control arms are what the original table does not have.])

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
    the shared default of 1e-3 suits best is Forward-Forward.])

Each arm was swept over five rates spanning two orders of magnitude, with both
edge peaks bracketed by a sixth point at 0.1. The best rate per arm:
backpropagation on the residual network 3e-2 (0.9424), PC-ALM 1e-2 (0.9286), PC
3e-3 (0.6880), backpropagation on the plain network 3e-3 (0.9151), and
Forward-Forward 1e-3 (0.8906).

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
)

The original table's ordering was Forward-Forward, then backpropagation, then
PC-ALM. Tuned, it is backpropagation, then PC-ALM, then Forward-Forward —
reversed among exactly the three arms the original compares. Forward-Forward's
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
)

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
  columns: (auto, auto, auto, auto, auto, auto),
  align: (right, right, right, right, right, right),
  [Layers], [backprop], [PC], [PC-ALM], [FF], [backprop, plain],
  [2], [0.9054], [0.9055], [0.9049], [--], [--],
  [4], [0.9012], [0.8963], [0.8943], [0.7177], [0.8312],
  [8], [0.8900], [0.8856], [0.8839], [0.6533], [0.8760],
  [16], [0.8830], [0.8043], [0.8628], [0.6021], [0.8687],
  [32], [0.8711], [0.6137], [0.8554], [0.5899], [0.8605],
  [64], [0.8518], [0.5547], [0.8207], [0.5896], [0.5550],
)

At two layers the three rules on the residual network are indistinguishable —
0.9054, 0.9055, 0.9049 — which is what the theory predicts: with one hidden
layer there is no credit to propagate. The gap opens at 16 layers and is 24
points wide at 32. PC-ALM stays within 3.1 points of backpropagation everywhere,
including at 64 layers.

#fig(include "/figures/grad_cosine_vs_depth.typ",
  caption: [Cosine between each local rule's weight update and the true
    backpropagation gradient at the same weights, one batch, three seeds.])

The gradient cosines say the same thing about the mechanism. At two layers both
rules reproduce the backpropagation gradient to three decimals (0.999). As depth
grows, PC-ALM's update stays aligned — 0.965, 0.928, 0.903, 0.889, 0.888 from 4
to 64 layers — while predictive coding's drops to 0.70-0.81 and its accuracy
falls with it.

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
