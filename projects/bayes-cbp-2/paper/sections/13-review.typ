#import "/template.typ": *

= A sceptical review <sec-review>

_Answers P13._ This section reviews the paper as a hostile referee would, check by check, and then revises the algorithm.

== The ten checks

#table(
  columns: (1fr, 2.2fr, 0.9fr),
  [Check], [Finding], [Verdict],
  [1. Posterior variance taken as plasticity?],
  [The paper rejects this (@sec-plasticity): plasticity is $1/2 log det(I + Sigma F_"new")$. CPR uses posterior variance only inside an evidence term, where it belongs.],
  [Claim "plasticity = variance": #wrong. Our use: #rig],

  [2. Prior hyperparameter, posterior variance and predictive uncertainty confused?],
  [$s_v$ is a prior scale, $S$ a posterior variance, $r_j$ a posterior probability. One link is weak: $N_"mem" = N \/ rho$ ties posterior precision to the hazard by assuming the Fisher is constant over a unit's lifetime.],
  [#heur for $N_"mem"$],

  [3. KL, Fisher or Hessian misused?],
  [$hat(G)$ uses model-sampled labels, so it is a Fisher estimate. But it is diagonal, and it is evaluated at the current network, not after removal. The second-order expansion is used for a fresh unit whose effect need not be small.],
  [#cond],

  [4. State-dependent rates treated as constant?],
  [The stationary formulas of @sec-jump state their rate assumptions. None is applied to CPR, whose rate depends on the state. CBP's rank-based rule is not Poisson at all.],
  [#rig as stated],

  [5. Mean reset wait read as recovery time?],
  [@sec-escape and @sec-switch separate the wait for a useful reset from training after it. @fig-mfpt still measures arrival, not adaptation.],
  [#cond],

  [6. Kramers and spectral gaps over-generalised?],
  [Kramers needs $Delta U >> T$; our double well has $Delta U \/ T = 4$, moderate. In high dimension, saddles may matter more than barriers @dauphin2014saddle. A dead unit is an invariant set, not a barrier. The gap result holds only for reversible, posterior-preserving jumps.],
  [Barrier story for networks: #heur],

  [7. Negative rates, probabilities above one, noise counted twice?],
  [$r_j$ is a logistic, in $(0, 1)$. The per-evaluation prior is $1 - (1 - rho)^K$. One double count remains: the current batch enters $hat(G)$ and then the evidence. Minibatch noise is #emph[not] double counted: for a candidate uncorrelated with the residual, $bb(E)[a_k^2] approx B_k$ when $hat(G)$ is a calibrated Fisher, and then $F_j approx 1/2 [x \/ (1 + x) - log(1 + x)] < 0$ with $x = s_v^2 B_k$. The Occam term already rejects pure-noise units.],
  [#cond; fixed below],

  [8. Empirical Bayes unidentifiable or circular?],
  [With zero fresh output weights, dead units have $r_j = rho$ whatever $rho$ is, so they cannot inform $rho$ #rig. Fitting $s_v$ or $rho$ on the window that is then tested is circular. CPR-D's best-of-$M$ choice suffers the winner's curse.],
  [#open for learning $rho$],

  [9. When does CPR destroy important knowledge?],
  [(a) A unit unused on recent batches but needed for a rare old class: its removal cost on the batch is near zero. (b) Route 2's model never values retention (@sec-route2) #rig. (c) A Newton-fitted $v'$ from one batch can change predictions on old inputs. The budget $beta_"max"$ and the norm clip bound but do not prevent this.],
  [Real risk],

  [10. Is there something simpler?],
  [Possibly. The new ingredient — the value of a fresh unit — can be added to CBP as a second utility term without any Bayesian frame. Resetting Adam state alone, L2-Init, ReDo or shrink-and-perturb may match CPR. All are baselines in @sec-plan.],
  [#open],
)

== Other claims in the paper, regraded

#table(
  columns: (1fr, auto),
  [Claim], [Grade],
  [Expected information gain is non-increasing under a static exchangeable model (@sec-plasticity)], [#rig],
  [Gain, information gain and the Kalman floor for linear-Gaussian models (@sec-gaussian)], [#rig],
  ["Fisher = Hessian = posterior precision" for networks], [#cond, three approximations],
  [Dead ReLU units are invariant under GD, SGD, Adam and weight decay (@sec-mechanisms)], [#rig],
  [Forward equation, inflow at the source rate; stationary laws for pure-jump and constant-rate resets (@sec-jump)], [#rig],
  [A reset keeps the posterior iff $p_0 prop lambda pi$; the Metropolis reset does (@sec-jump)], [#rig, checked in @fig-dw-stationary],
  [Soft reset converges to Ornstein–Uhlenbeck as $a -> 0$], [#cond, needs $s^2 prop 1 \/ a$],
  [Small-rate resetting helps iff the coefficient of variation exceeds one (@sec-escape)], [#rig],
  [Reversible posterior-preserving jumps cannot lower the gap], [#rig],
  [EWC with exact Laplace traces the Pareto front of two quadratics (@sec-switch)], [#rig],
  [Second-order removal utility; centring; redundancy via $1 \/ (C^(-1))_(j j)$ (@sec-utility)], [#rig / #cond / #rig],
  [CBP's fixed rate equals the filter on dead units (@sec-unified)], [#cond, zero output initialisation],
  [Averaging the reset mixture sends a ReLU unit to the dead boundary (@sec-route2)], [#rig],
  [CPR improves on CBP], [#open],
)

== Revisions

The review changes the algorithm in three places.

+ *Count the batch once.* Update $hat(G)$ #emph[after] computing the evidence, so the evaluation batch does not enter both.
+ *Know the bias of one candidate.* One draw of $w'$ gives $log Z^"fresh"$ for that draw, not the average. By Jensen's inequality the expected log of one draw underestimates the log of the average #rig. So with $M = 1$, CPR resets less often than the model says. Use the log-mean-exp of $M > 1$ draws when affordable.
  A tempting further fix — subtracting gradient noise from $a_k^2$ — would be wrong: the Occam term already does it (check 7).
+ *Add retention explicitly, and label it.* Since the model does not value retention (check 9b), add Route 3's term: $Delta_j <- Delta_j - kappa thin c_j^(A,"rem")$, using frozen old-task statistics. This departs from the model. It is a decision-theoretic price, and $kappa = 0$ recovers pure CPR. Experiments must report both.
