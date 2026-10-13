#import "/template.typ": *

= Rigorous review and counter-examples <p13>

#callout(title: [The question])[
  As a sceptical reviewer, check every view and algorithm so far:
  posterior variance taken as plasticity; prior hyperparameters, posterior and predictive uncertainty confused;
  misuse of KL, Fisher or Hessian; state-dependent rates treated as constant; reset waiting time read as recovery time;
  Kramers or spectral-gap results over-generalised; negative rates, probabilities above one, noise counted twice;
  empirical Bayes unidentifiable, circular or over-fitting; when the algorithm destroys important knowledge;
  whether something simpler does as well. Grade every conclusion, then revise the theory and algorithm.
]

== Core view

Most earlier results survive as stated, because their assumptions were stated with them.
Three weak points are real: learning the hazard $rho$ is not identifiable from dead units;
the change-point model does not protect old knowledge;
and something simpler might match CPR.
The review changes the algorithm in three places and rejects one tempting fix.

== Derivation


=== The ten checks

#table(
  columns: (1fr, 2.2fr, 0.9fr),
  [Check], [Finding], [Verdict],
  [1. Posterior variance taken as plasticity?],
  [@p1 rejects this: plasticity is $1/2 log det(I + Sigma F_"new")$. CPR uses posterior variance only inside an evidence term, where it belongs.],
  [Claim "plasticity = variance": #wrong. Our use: #rig],

  [2. Prior hyperparameter, posterior variance and predictive uncertainty confused?],
  [$s_v$ is a prior scale, $S$ a posterior variance, $r_j$ a posterior probability. One link is weak: $N_"mem" = N \/ rho$ ties posterior precision to the hazard by assuming the Fisher is constant over a unit's lifetime.],
  [#heur for $N_"mem"$],

  [3. KL, Fisher or Hessian misused?],
  [$hat(G)$ uses model-sampled labels, so it is a Fisher estimate. But it is diagonal, and it is evaluated at the current network, not after removal. The second-order expansion is used for a fresh unit whose effect need not be small.],
  [#cond],

  [4. State-dependent rates treated as constant?],
  [The stationary formulas of @p6 state their rate assumptions. None is applied to CPR, whose rate depends on the state. CBP's rank-based rule is not Poisson at all.],
  [#rig as stated],

  [5. Mean reset wait read as recovery time?],
  [@p7 and @p8 separate the wait for a useful reset from training after it. @fig-mfpt still measures arrival, not adaptation.],
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
  [(a) A unit unused on recent batches but needed for a rare old class: its removal cost on the batch is near zero. (b) Route 2's model never values retention (@p11) #rig. (c) A Newton-fitted $v'$ from one batch can change predictions on old inputs. The budget $beta_"max"$ and the norm clip bound but do not prevent this.],
  [Real risk],

  [10. Is there something simpler?],
  [Possibly. The new ingredient — the value of a fresh unit — can be added to CBP as a second utility term without any Bayesian frame. Resetting Adam state alone, L2-Init, ReDo or shrink-and-perturb may match CPR. All are baselines in @p14.],
  [#open],
)

=== Claims of earlier stages, regraded

#table(
  columns: (1fr, auto),
  [Claim], [Grade],
  [Expected information gain is non-increasing under a static exchangeable model (@p1)], [#rig],
  [Gain, information gain and the Kalman floor for linear-Gaussian models (@p2)], [#rig],
  ["Fisher = Hessian = posterior precision" for networks], [#cond, three approximations],
  [Dead ReLU units are invariant under GD, SGD, Adam and weight decay (@p4)], [#rig],
  [Forward equation, inflow at the source rate; stationary laws for pure-jump and constant-rate resets (@p6)], [#rig],
  [A reset keeps the posterior iff $p_0 prop lambda pi$; the Metropolis reset does (@p6)], [#rig, checked in @fig-dw-stationary],
  [Soft reset converges to Ornstein–Uhlenbeck as $a -> 0$], [#cond, needs $s^2 prop 1 \/ a$],
  [Small-rate resetting helps iff the coefficient of variation exceeds one (@p7)], [#rig],
  [Reversible posterior-preserving jumps cannot lower the gap], [#rig],
  [EWC with exact Laplace traces the Pareto front of two quadratics (@p8)], [#rig],
  [Second-order removal utility; centring; redundancy via $1 \/ (C^(-1))_(j j)$ (@p9)], [#rig / #cond / #rig],
  [CBP's fixed rate equals the filter on dead units (@p10)], [#cond, zero output initialisation],
  [Averaging the reset mixture sends a ReLU unit to the dead boundary (@p11)], [#rig],
  [CPR improves on CBP], [#open],
)

=== Revisions

The review changes the algorithm in three places.

+ *Count the batch once.* Update $hat(G)$ #emph[after] computing the evidence, so the evaluation batch does not enter both.
+ *Know the bias of one candidate.* One draw of $w'$ gives $log Z^"fresh"$ for that draw, not the average. By Jensen's inequality the expected log of one draw underestimates the log of the average #rig. So with $M = 1$, CPR resets less often than the model says. Use the log-mean-exp of $M > 1$ draws when affordable.
  A tempting further fix — subtracting gradient noise from $a_k^2$ — would be wrong: the Occam term already does it (check 7).
+ *Add retention explicitly, and label it.* Since the model does not value retention (check 9b), add Route 3's term: $Delta_j <- Delta_j - kappa thin c_j^(A,"rem")$, using frozen old-task statistics. This departs from the model. It is a decision-theoretic price, and $kappa = 0$ recovers pure CPR. Experiments must report both.


== When it holds

The grades hold relative to the assumptions each stage stated. A grade of #rig means "proved under that model", not "true of networks".

== Limits

A self-review cannot replace an independent one, or an experiment. Grades of #open are the honest ones here.

== Alternatives

Check 10's simpler candidates are the alternatives: CBP with a fresh-unit term added; Adam-state reset alone; L2-Init; ReDo; shrink-and-perturb. All become baselines in Stage P14.

== Next step

Stage P14 assembles a research plan whose experiments can refute the claims graded #heur and #open here.
