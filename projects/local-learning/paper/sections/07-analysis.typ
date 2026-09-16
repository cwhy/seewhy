#import "/template.typ": *

= Analysis <sec-analysis>

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

== What the deep Forward-Forward stack was actually telling us

Under our first implementation, Forward-Forward scored 0.5751 on the 31-layer
stack while a linear probe on that same network's activations reached 0.8919,
and we read that as a readout failure: goodness summed over thirty layers adds
thirty layers' worth of variance to a signal that does not grow with depth.

The reading was wrong, and the audit (@res-audit) says why. With hard negatives the
goodness sweep reaches 0.8570 on the same network — close to what the probe was
already extracting — so there was no readout to blame. What the probe had
detected was that the features were fine while the *training signal* was not:
uniformly random wrong labels are separable at the first layer, so the layers
above it had nothing to learn from, and their goodness was noise that the sum
then accumulated.

This is worth stating plainly because the failure mode is general. A local
learning rule can look like it has a depth problem when what it has is a
negative-sampling problem, and the two are distinguishable only by testing the
training signal rather than the architecture. The probe was the right
measurement; we drew the wrong conclusion from it.

What survives: Forward-Forward is still the weakest arm at five epochs, its deep
stack is still 8 points below its shallow one, and it still gains least from
extra training. What does not: any claim that its readout does not scale.

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

== Why a nearly-orthogonal gradient estimate trains anything

EGGROLL's update at 512 pairs has a cosine of 0.094 with the true gradient, and
it trains the network to 0.9025 in one epoch — better than backpropagation
managed at the rate the original comparison used. That looks paradoxical and is
not, for two reasons that are worth separating.

The estimate is *unbiased*, not merely correlated. Averaging antithetic pairs of
isotropic perturbations gives an estimator whose expectation is the gradient; the
0.094 is not the quality of the direction it is aiming at, but the noise in one
sample of it. Over an epoch the weights take 937 such steps, and the noise
averages down across steps at the same rate it would across a larger population:
what matters is roughly $P$ times the number of steps, not $P$ alone. Adam helps
further, because its per-coordinate normalisation is insensitive to the scale of
the estimate — which is also why the perturbation size $sigma$ barely mattered
in exp10, varying accuracy by under a point across a thirty-fold range.

The second reason is that $sqrt(P slash d)$ is a statement about the *whole*
parameter vector. An update can be almost orthogonal to the gradient overall and
still carry most of the signal in the few directions that matter for the loss,
which is the usual situation in an over-parameterised network. This is testable
and we did not test it: projecting the ES estimate onto the top curvature
directions of the loss and measuring the cosine there would say whether the
useful part of the estimate is concentrated where the loss actually moves.

== Why feedback alignment dies where the other rules survive

Feedback alignment and PC-ALM are both answers to the same objection — that
backpropagation needs the forward weights to carry the error backwards — and
they fail and succeed in opposite places. PC-ALM holds within 3.1 points of
backpropagation at 64 layers; feedback alignment is at chance by 16.

The difference is what each does with depth. PC-ALM's duals accumulate a
quantity that converges to the backprop credit signal, so every settling step
brings each layer's estimate closer to the right one. Feedback alignment's error
passes through one random matrix per layer, and the layers do not correct each
other: the estimate at layer $i$ is the product of $L - i$ random projections
applied to a signal that was correct only at the output. The method works by the
forward weights adapting to the feedback, and that adaptation is itself driven by
the feedback — at depth, the signal that would drive alignment has already been
scrambled by the layers above. The alignment column in @res-fa measures exactly
this: 0.303 at two layers, 0.045 at eight, zero at 32.

Two things make it worse here specifically. The network is narrow, so a random
projection into 32 dimensions loses more than one into 800 would. And the units
are ReLU, so a misdirected update can kill a unit permanently — 76% of a layer,
at the rates that train fastest. The same architecture with tanh units, which
cannot die, recovers the literature's result. None of this rescues the method at
depth 32, but it does locate the failure: not in the idea that random feedback
can teach, which works at two and four layers here, but in what happens to a
random signal after it has been re-randomised thirty times.

== Why EGGROLL does not care about depth

EGGROLL's accuracy is flat from 2 to 64 layers where every other rule here
declines. The mechanism is that it has no credit assignment problem to fail at:
it never propagates a signal through the layers, forwards or backwards. Depth
enters its estimator in exactly one place — through $d$, the parameter count,
which grows with depth and therefore lowers the cosine, from 0.136 at two layers
to 0.074 at 64, precisely as $sqrt(P slash d)$ says it should. That the accuracy
does not follow the cosine down is the same point as above: the per-step
alignment is not what determines the epoch's progress.

Two things this does not show. It does not show EGGROLL beating backpropagation:
tuned backpropagation at 32 layers is 0.9374 against EGGROLL's 0.9025, and the
depth sweep's backprop arm follows the source paper's rate rule
$10^(-3) sqrt(N slash L)$, which deliberately shrinks the step with depth, so
part of the crossover at 16 layers is a schedule rather than a method. And it
does not show depth-independence in general — at some depth the *network* stops
being trainable in one epoch regardless of the rule, and 64 layers with skip
connections is evidently not that depth.

The architecture result points the same way. EGGROLL is the only arm that does
better on the deep-narrow network (0.9025) than on the short-wide one (0.8272),
even after tuning each separately, and the likely reason is not depth at all but
uniformity: the residual network's weights are all unit normal by construction,
with the scale factors outside them, so a single perturbation scale $sigma$ fits
every layer. The plain network is He-initialised, so its layers differ in
magnitude and any single $sigma$ is wrong somewhere. That predicts a per-layer
$sigma$ proportional to each layer's weight RMS should close the gap, which is
the obvious next experiment and one we did not run.

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
