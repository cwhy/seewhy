#import "/template.typ": *

= Limitations and negative results

*One dataset, one width, mostly one epoch.* Everything here is MNIST at width
32, plus one Fashion-MNIST cell used as a check against the authors' own numbers.
The claims about ordering, architecture and learning rate are claims about that
regime. In particular, "backpropagation on the short wide network wins" is a
statement about 56K parameters and one epoch on MNIST; nothing here speaks to
what any of these rules do on a task where the architecture has to be deep for
reasons other than making predictive coding hard.

*Our PC is 4.7 points below the authors' PC.* On their Fashion-MNIST cell we get
0.6340 ± 0.0194 against their 0.6813, while backpropagation and PC-ALM land
within a point of theirs. Substituting their tabulated activity step changes
nothing (0.6330), so the deviation is not the one deliberate difference in our
implementation. It could be seed variance — theirs is a single run and PC is the
highest-variance arm we measured — or it could be a difference we have not
found. We did not run their code, which is the test that would settle it. Since
PC appears here only as the control that isolates PC-ALM's duals, and it is
*worse* in our hands than in theirs, the conclusion it supports is if anything
conservative.

*Forward-Forward is our reading of the method, not Hinton's exact recipe.* Four
deviations, each defensible and each a confound: we normalise each layer's input
to unit mean square rather than unit L2 norm (the same direction, a constant
apart); we sample negative labels uniformly rather than mining hard negatives
from the network's own predictions; we train all layers simultaneously rather
than greedily layer by layer; and we fix the goodness threshold at 2.0 without
sweeping it. Hinton's reported MNIST error rates come from a longer schedule,
wider layers and hard negatives, and we do not claim to reproduce those — only
the arm of this particular comparison, which our numbers match to within 0.2
points.

*The learning-rate sweep used two seeds.* The best rate per arm is therefore
picked with about a percentage point of noise on each grid point. The spacing is
a factor of three, and the ranking reversal survives it — backpropagation's best
is 5.2 points above Forward-Forward's best, which is far outside that noise —
but the specific best rates should be read as "within a factor of three", not as
tuned values.

*One hyperparameter of the local rules was never swept.* We use the paper's
$T = 2L$, $alpha = 1$, $rho = 1$ throughout. A larger settling budget is the
obvious way to buy PC accuracy, and it would make PC-ALM's wall-clock
disadvantage worse rather than better, so the trade-off is not neutral; we did
not map it.

*EGGROLL's numbers were measured on a shared GPU.* Every EGGROLL cell ran while
an unrelated job held part of the same device, which costs a factor of 2.0 in
wall-clock (§6.8 measures it on three arms with known idle timings, getting
2.01, 2.17 and 2.01). Accuracy is unaffected. We report EGGROLL's raw times and
the halved equivalents rather than silently mixing the two, but a single
calibration factor is cruder than re-running the arm on an idle device would be.

*EGGROLL's depth sweep is not a like-for-like schedule.* It ran at a fixed rate
of 3e-2 at every depth, while the three backprop-derived arms follow the source
paper's rule $10^(-3) sqrt(N slash L)$, which shrinks with depth by design. The
flatness of EGGROLL's curve is a property of the method; the point where it
crosses backpropagation is partly a property of that rule. A depth-tuned
backpropagation arm would settle it and was not run.

*EGGROLL has knobs we swept coarsely and knobs we did not sweep at all.* Its
population, perturbation scale and outer rate were gridded on the residual
network (exp10, exp11, exp12) and its scale and rate re-gridded on the plain
network (exp15), each on one seed at three-fold spacing. The perturbation rank
was fixed at 1 throughout and never varied, and the same single $sigma$ was
applied to every layer, which §7 argues is the likely reason the plain network
underperforms.

*Timings are one machine, one framework, one execution style.* All arms are
compiled by XLA into a single scanned loop on one RTX 4090, which is why our
absolute numbers are faster than the ones we are replicating by factors between
1.6 and 90. Ratios between arms are what we compare, and the ratio we care about
most — PC-ALM against backpropagation, 40.8x — matches the original's 41.6x. But
a different framework, batch size or device would move these, and the
neuromorphic hardware that motivates local learning in the first place
@seely2026pcalm would change the accounting entirely: a GPU charges PC-ALM for
64 sequential settling steps that dedicated analogue hardware would not.

*The depth sweep's wall-clock numbers are contaminated and are not used.* Two
copies of that sweep ran concurrently on the same GPU for part of its duration,
so its `train_s` values are inflated by contention. The accuracies and gradient
cosines it reports are unaffected — they do not depend on scheduling — and no
timing figure in this paper is drawn from it. The accidental duplication did
yield one useful measurement: 38 cells ran twice at identical seeds, and their
accuracies differ by a median of 0.00005 and a maximum of 0.015, which is the
run-to-run nondeterminism of float32 GPU reductions and the floor under every
difference quoted here.

*What we did not test.* Whether reading Forward-Forward's goodness out
differently (top layers only, per-layer normalisation of the score) recovers the
deep stack's 0.89 probe accuracy — the direct test of the readout explanation in
section 7. Whether Forward-Forward's decay over epochs is specific to random
negatives. Whether PC-ALM's advantage at depth 64 persists at depth 1000, which
is the source paper's actual headline and is out of reach at this scale.
