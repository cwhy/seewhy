#import "/template.typ": *

= Summary <sec-conclusion>

_Answers P14, part H._

*1. What is plasticity?*
The expected information the next batch carries about the model's predictions, $cal(P)^"pred"_t$.
In the Gaussian case it is $1/2 log det(I + Sigma F_"new")$: posterior variance counted only along the directions the new data can see.
A second, separate kind is computational: how fast the learning dynamics can reach what the data imply. Barriers, dead units and mixing belong here.

*2. Why is it lost?*
Statistically, because a static model must lose it: expected information gain cannot rise when the world is assumed fixed (Proposition 1).
Computationally, because networks enter sets that gradients cannot leave, such as dead ReLU units, and because SGD's noise vanishes at the minima networks reach.
Optimiser state, norm growth and representation collapse add further, non-Bayesian causes.

*3. Why can the proposed mechanism restore it?*
A change-point world model sets a floor under statistical plasticity, as process noise does in a Kalman filter (@fig-kalman).
Its resets are the filter's prediction step.
They also leave invariant sets in one jump, which no gradient method can do.
They help through where they land, not through the reset event itself (@fig-mfpt).

*4. What is new relative to CBP?*
Four things, all untested on networks:
the reset probability is a posterior probability, not a rank;
it weighs the value of a fresh unit on recent data against the cost of removing the old one;
it can reset a useful unit whose setting conflicts with new data;
and the fresh unit's output weights are a derived Newton fit.
On dead units with zero output initialisation, CPR reduces to CBP's fixed rate, which explains why that heuristic works there.

*5. What is settled and what is not?*
Settled, as propositions: the monotonicity of static plasticity; the Gaussian formulas and floor; dead-unit invariance; the jump-diffusion forward equation and its stationary laws; the posterior-preservation condition; the first-passage and gap results; the Pareto property of EWC on quadratics.
Not settled: whether the change-point evidence ranks units better than CBP's utility; whether any of this explains plasticity loss in networks, as opposed to optimiser effects; how to learn $rho$, which is unidentifiable from the dead units.
@sec-plan lists the experiments that decide these.

*6. Is there a more promising direction?*
Possibly one.
The deepest limit found here is that a change-point model never values retention (@sec-route2).
Old knowledge is worth keeping only if old tasks can return.
A world model with #emph[recurrence] — where a replaced unit may return to an earlier value, or tasks are drawn from a hierarchy — would derive stability and plasticity from one model, instead of adding a forgetting price by hand.
That model is harder to filter online. It may be the more fundamental object.
