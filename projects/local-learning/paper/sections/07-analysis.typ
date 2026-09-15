#import "/template.typ": *

= Analysis

== What the original table was measuring

Three numbers, three architectures, one learning rate. Each of those last two
facts explains part of the ranking, and neither is about learning rules.

*Architecture.* Forward-Forward's row used a two-layer network because
Forward-Forward needs one: goodness replaces the output layer, so the parameters
that would have been a classifier head went into the hidden layers instead, and
the natural shape is short and wide. The other two rows used the PC-ALM paper's
benchmark network, which is deliberately tall and thin because that is where
predictive coding is known to struggle. Under a one-epoch budget the short wide
network is simply an easier optimisation problem, and the evidence that this —
not the learning rule — is what the third row measures is that backpropagation
on that same short wide network beats Forward-Forward on it by 2.5 points, and
Forward-Forward on the tall thin network falls 31 points.

*Learning rate.* The rate 1e-3 is what the PC-ALM paper's rule
$eta_0 sqrt(N slash L)$ yields at $N = L = 32$, and it is a rule designed to keep
a fixed step meaningful across a width-depth grid, not to maximise accuracy in
937 updates. Given its own best rate, backpropagation on the residual network
goes from 0.8704 to 0.9424 and PC-ALM from 0.8533 to 0.9286. Forward-Forward
gains nothing, because 1e-3 was already its best rate — it is the only arm the
shared default happened to suit, and it is the arm that won.

Neither observation says the local rules do not work. It says the table does not
rank them, and that a parameter-matched comparison across different
architectures at a shared learning rate mostly measures which architecture and
which rate you picked.

== Why Forward-Forward collapses with depth

The obvious reading of 0.5751 on a 31-layer stack is that the local objective
stops carrying useful information deep in the network. That reading is wrong
here, and the probe is what rules it out: a linear classifier on the same
network's own activations gets 0.8919, within two points of what the same probe
gets on the two-layer network. The features are there. What fails is
Forward-Forward's own way of reading them out.

The mechanism is visible in the readout's arithmetic. Classification sums
goodness over the 30 layers above the first and picks the largest total. Each
layer contributes a quantity trained only to sit above a threshold for correct
labels and below it for wrong ones — a per-layer margin, not a calibrated score.
Summing thirty of them adds thirty layers' worth of variance to a signal that
does not grow with depth, and the argmax over ten candidate labels degrades
accordingly. A two-layer network sums one such term.

This is falsifiable, and one test has been run: if the collapse were an
optimisation failure rather than a readout failure, a different learning rate
should recover it, and it does not — the best of four rates on the deep stack
gives 0.5808. A second test we did not run would settle it: weighting or
normalising each layer's goodness contribution before the sum, or reading out
from the top layers only. If the collapse is the sum, those should recover most
of the probe's 0.8919.

== Why PC-ALM survives depth and PC does not

This part of the source paper's account holds up under the one test that
isolates it. PC and PC-ALM here are the same code path with $alpha$ set to 0 or
1; the same architecture, the same settling budget $T = 2L$, the same activity
step, the same seeds. At two layers they are indistinguishable from
backpropagation (0.9055 and 0.9049 against 0.9054), which is what it means for
there to be no credit to propagate. Each additional layer adds one more local
compromise between what a layer holds and what its neighbours predict, and
predictive coding's supervision signal is attenuated by each one: by 32 layers
it has lost 29 points of accuracy, and the update it produces has drifted to a
cosine of 0.81 with the true gradient. PC-ALM's duals accumulate each layer's
prediction error over the settling steps rather than letting it be averaged
away, and its update stays at 0.89 — consistent with the paper's claim that in
the linear case the duals converge to exactly the backpropagation credit signals
@seely2026pcalm @lecun1988theoretical.

What our evidence does *not* establish is the 1000-layer claim. We tested to 64
layers on one dataset. The trend at 64 — PC-ALM 3.1 points behind
backpropagation and still tracking it, PC 30 points behind and falling — is
consistent with their result, and is as far as a 56K-parameter replication can
speak to it.

== Forward-Forward gets worse with training

One arm in the wall-clock experiment moves backwards: Forward-Forward peaks at
0.901 after two epochs and decays to 0.860 by ten, while every other arm
improves monotonically. This is not overfitting in the usual sense — the
training set has 60,000 images and the network has 56,000 parameters, and the
other arms at the same budget do not decay.

The mechanism is that Forward-Forward optimises a different objective from the
one it is scored on. Its layers are trained to push goodness across a threshold;
its predictions come from comparing goodness across ten label hypotheses.
Driving every positive example far above threshold is progress on the first
objective and can be neutral or harmful for the second, which depends on the
*differences* between goodness under competing labels.

The probe says the damage is in the representation, not only in the readout, and
it says so against our expectation: the linear probe on the same features falls
from 0.9102 after one epoch to 0.7956 after ten, a loss of 11.5 points against
the goodness readout's 2.7. Ten epochs of the local objective leave the features
*less* linearly separable than one epoch did. Whatever the layers are doing with
the extra training, it is not building a better representation of digits. We
have not isolated what they are doing instead; logging the probe and the
per-layer goodness distributions every epoch would be the next measurement, and
it is cheap.
