#import "/template.typ": *

= Research plan and falsification tests <sec-plan>

_Answers P14, parts A–G._ Nothing here has been run.
Statements about outcomes are conditions the theory implies, to be measured, not forecasts.

== A. Theoretical framework

- *Plasticity* is the expected information the next batch gives about the predictions, $cal(P)^"pred"_t$ (@sec-plasticity). Its local, equilibrium proxy is linear response. Barrier crossing and mixing are a separate, computational kind.
- *Core assumption:* the world can change, and the learner should say how. Without a change model, exact Bayes loses plasticity by design (Proposition 1).
- *Main derivations:* the Gaussian gain and floor (@sec-gaussian); the jump-diffusion forward equation and its stationary laws (@sec-jump); the reset first-passage formulas (@sec-escape); the change-point posterior per unit (@sec-route2).
- *Boundaries:* the Gaussian and Laplace results are local. Nothing in the theory covers representation collapse or norm growth, which are optimiser effects.

== B. Final algorithm

*Change-Point Reset* (CPR), @sec-algorithm with the revisions of @sec-review.

- *Model:* each unit is replaced with probability $rho$ per step by a draw from the initial distribution with Gaussian output weights.
- *Objective:* the posterior probability of replacement, $r_j = sigma("logit" rho_K + Delta_j)$.
- *Key update:* $Delta_j = -c_j^"rem" + F_j - O_j^"stay" - kappa c_j^(A,"rem")$, where $kappa = 0$ is the pure model.
- *Action:* reset with probability $r_j$ (a sample, not an average). New input weights are the candidate used in $F_j$; new output weights are its one-step Newton fit.

== C. Contributions: proved versus to be tested

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

== D. Experiments

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

*Metrics,* per task: $cal(R)_k$ (@sec-switch); final new-task accuracy; old-task accuracy on held-out data from earlier tasks; fraction of dead units; effective rank of features; resets per step; and #emph[reset survival], the fraction of reset units whose utility exceeds the layer median $k$ steps later.

== E. Ablations

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

== F. Failure modes and falsification tests

Each test can refute a core claim:

+ *Does the evidence rank units?* In small networks, compute an oracle: for each unit, reset it in a copy, train $k$ steps, and record the loss change. If $Delta_j$ correlates with the oracle no better than CBP's utility, or than chance, the model's main claim fails.
+ *Is it the optimiser?* If resetting Adam state alone recovers most of the plasticity that CPR recovers, the cause is optimiser state, not the posterior.
+ *Does it hurt without change?* On a stationary stream the model implies few resets. If CPR still lowers accuracy against Adam there, the reset prior is harmful.
+ *Does the Bayesian story apply to networks at all?* Run an exact Bayesian last layer, static and with the change-point prior, beside the network. Proposition 1 must show in the static Bayesian layer. If the network's plasticity loss tracks dead units and rank but not anything the hazard controls, the Bayesian framing explains nothing about it.
+ *The coefficient-of-variation criterion.* Across tasks, the benefit of resets should grow with the coefficient of variation of adaptation time (@sec-escape). No relation would refute the first-passage picture for networks.

*Where the theory itself implies CPR could do worse than a baseline:*
(i) frequently recurring tasks with $kappa = 0$, since the model does not value retention;
(ii) smooth drift, where the drift models behind shrink-and-perturb and L2-Init fit better;
(iii) very small batches, where the one-candidate evidence is noisy and biased low, so CPR does little;
(iv) wide, redundant layers, where CBP already picks harmless units and the Newton fit can over-fit one batch.

== G. Alternatives with different starting points

+ *CPR-D* (decision theory, Route 3): an explicit horizon and forgetting price instead of a prior hazard.
+ *MPR* (MCMC, @sec-jump): a posterior-preserving reset; only computational plasticity. A clean test of whether statistical plasticity is needed at all.
+ *Shrink-and-relax* (a drift world model, @sec-unified): the exact Bayesian prediction step for a linear-Gaussian drift. Shrink the weights toward their prior mean and raise the per-weight step size to match the inflated variance, with no reset at all. It derives the S&P family's continuous counterpart instead of CBP's.
