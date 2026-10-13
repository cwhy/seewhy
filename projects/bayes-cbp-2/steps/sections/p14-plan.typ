#import "/template.typ": *

= Final research plan and falsifiable experiments <p14>

#callout(title: [The question])[
  Assemble a research plan: (A) theory; (B) final algorithm; (C) contributions, proved versus to be tested;
  (D) experiments on continual supervised learning, task switches, distribution drift and long online learning, against fair baselines;
  (E) ablations; (F) at least three experiments that could overturn the core assumptions, and when the algorithm should do worse than baselines;
  (G) two alternatives from different starting points; (H) a summary answering six questions.
  Do not hide inconsistent results to make a neat theory.
]

== Core view

The plan tests one claim above all: that a unit's posterior probability of replacement ranks units better than CBP's utility, because it includes the value of a fresh unit.
Nothing here has been run. Statements about outcomes are conditions the theory implies, to be measured — not forecasts.

== Derivation

=== A. Theoretical framework

- *Plasticity* is the expected information the next batch gives about the predictions, $cal(P)^"pred"_t$ (@p1). Its local, equilibrium proxy is linear response. Barrier crossing and mixing are a separate, computational kind.
- *Core assumption:* the world can change, and the learner should say how. Without a change model, exact Bayes loses plasticity by design (Proposition 1).
- *Main derivations:* the Gaussian gain and floor (@p2); the jump-diffusion forward equation and its stationary laws (@p6); the reset first-passage formulas (@p7); the change-point posterior per unit (@p11).
- *Boundaries:* the Gaussian and Laplace results are local. Nothing in the theory covers representation collapse or norm growth, which are optimiser effects.

=== B. Final algorithm

*Change-Point Reset* (CPR), @p12 with the revisions of @p13.

- *Model:* each unit is replaced with probability $rho$ per step by a draw from the initial distribution with Gaussian output weights.
- *Objective:* the posterior probability of replacement, $r_j = sigma("logit" rho_K + Delta_j)$.
- *Key update:* $Delta_j = -c_j^"rem" + F_j - O_j^"stay" - kappa c_j^(A,"rem")$, where $kappa = 0$ is the pure model.
- *Action:* reset with probability $r_j$ (a sample, not an average). New input weights are the candidate used in $F_j$; new output weights are its one-step Newton fit.

=== C. Contributions: proved versus to be tested

*Provable propositions*, each with a proof or proof sketch here:
expected plasticity is non-increasing under a static model;
the Kalman floor $k_oo$;
dead ReLU units are invariant under gradient methods;
the forward equation with source-rate inflow and its stationary laws;
the posterior-preservation condition $p_0 prop lambda pi$;
the soft-to-OU limit;
the coefficient-of-variation criterion;
reversible jumps cannot lower the gap;
the EWC Pareto front;
averaging a symmetric reset mixture sends a unit to the dead boundary.

*Hypotheses for experiment:*
that the change-point evidence ranks units better than CBP's utility;
that the fresh-unit term matters;
that sampling beats soft averaging in networks;
that CPR's advantages, if any, survive matched reset counts.

=== D. Experiments

#table(
  columns: (0.8fr, 1.4fr, 1fr),
  [Setting], [Stream], [Why],
  [Continual supervised], [Online permuted MNIST, hundreds of tasks, small MLP @dohare2024loss], [Standard plasticity-loss benchmark; cheap],
  [Task switching], [Sequences of binary tasks from CIFAR-100 classes, fixed steps per task], [Abrupt changes: the change-point model's home ground],
  [Distribution drift], [Slowly changing regression @dohare2024loss], [Smooth drift: the change-point model is #emph[mis]specified here],
  [Long-run online], [Warm-started class-incremental CIFAR @ash2020warm, $10^7$+ steps], [Long horizons, where dead units accumulate],
)

*Baselines,* each with the same hyperparameter-search budget, architecture and seeds (at least five):
SGD; Adam; Adam with L2; online EWC; CBP; ReDo; shrink-and-perturb; L2-Init; Adam-state reset only; Langevin re-heating at switches;
#emph[random reset at CPR's own rate] (uniformly chosen units); and a #emph[fresh network per task] as the plasticity reference.

*Metrics,* per task: $cal(R)_k$ (@p8); final new-task accuracy; old-task accuracy on held-out data from earlier tasks; fraction of dead units; effective rank of features; resets per step; and #emph[reset survival], the fraction of reset units whose utility exceeds the layer median $k$ steps later.

=== E. Ablations

#table(
  columns: (auto, 1fr),
  [Component], [Variants],
  [Utility], [Full $Delta_j$; removal cost only; fresh term only; CBP contribution utility],
  [Trigger], [Bernoulli sample; threshold $r_j > 1\/2$; top-$k$ by $r_j$ at CPR's count],
  [Target], [Newton $v'$ vs $v' = 0$; $w' tilde p_0$ vs a learned $p_0$],
  [Posterior estimate], [Model-sampled vs empirical Fisher; $N_"mem" = N \/ rho$ vs fixed],
  [Randomness], [$M = 1$ vs $M = 4$ candidates; Thompson sampling vs the most probable decision],
  [Retention], [$kappa in {0, 0.1, 1}$],
)

=== F. Failure modes and falsification tests

Each test can refute a core claim:

+ *Does the evidence rank units?* In small networks, compute an oracle: for each unit, reset it in a copy, train $k$ steps, and record the loss change. If $Delta_j$ correlates with the oracle no better than CBP's utility, or than chance, the model's main claim fails.
+ *Is it the optimiser?* If resetting Adam state alone recovers most of the plasticity that CPR recovers, the cause is optimiser state, not the posterior.
+ *Does it hurt without change?* On a stationary stream the model implies few resets. If CPR still lowers accuracy against Adam there, the reset prior is harmful.
+ *Does the Bayesian story apply to networks at all?* Run an exact Bayesian last layer, static and with the change-point prior, beside the network. Proposition 1 must show in the static Bayesian layer. If the network's plasticity loss tracks dead units and rank but not anything the hazard controls, the Bayesian framing explains nothing about it.
+ *The coefficient-of-variation criterion.* Across tasks, the benefit of resets should grow with the coefficient of variation of adaptation time (@p7). No relation would refute the first-passage picture for networks.

*Where the theory itself implies CPR could do worse than a baseline:*
(i) frequently recurring tasks with $kappa = 0$, since the model does not value retention;
(ii) smooth drift, where the drift models behind shrink-and-perturb and L2-Init fit better;
(iii) very small batches, where the one-candidate evidence is noisy and biased low, so CPR does little;
(iv) wide, redundant layers, where CBP already picks harmless units and the Newton fit can over-fit one batch.

=== G. Alternatives with different starting points

+ *CPR-D* (decision theory, Route 3): an explicit horizon and forgetting price instead of a prior hazard.
+ *MPR* (MCMC, @p6): a posterior-preserving reset; only computational plasticity. A clean test of whether statistical plasticity is needed at all.
+ *Shrink-and-relax* (a drift world model, @p10): the exact Bayesian prediction step for a linear-Gaussian drift. Shrink the weights toward their prior mean and raise the per-weight step size to match the inflated variance, with no reset at all. It derives the S&P family's continuous counterpart instead of CBP's.


=== H. Summary


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
@p14 lists the experiments that decide these.

*6. Is there a more promising direction?*
Possibly one.
The deepest limit found here is that a change-point model never values retention (@p11).
Old knowledge is worth keeping only if old tasks can return.
A world model with #emph[recurrence] — where a replaced unit may return to an earlier value, or tasks are drawn from a hierarchy — would derive stability and plasticity from one model, instead of adding a forgetting price by hand.
That model is harder to filter online. It may be the more fundamental object.


== When it holds

The plan assumes the metric $cal(R)_k$ of Stage P8 captures plasticity, and that small networks behave like large ones for the oracle test.

== Limits

Compute: the oracle test (reset each unit in a copy and train) is feasible only for small networks.
The benchmarks are supervised; reinforcement learning, where plasticity loss is also studied, is left out.

== Alternatives

Part G above lists them: CPR-D, the Metropolis-corrected reset, and shrink-and-relax.

== Next step

Run the falsification tests of part F first. They decide whether the Bayesian framing explains network plasticity at all, before any tuning of CPR.
