#import "/template.typ": *

= CBP and its variants in one frame <p10>

#callout(title: [The question])[
  Place in one frame: utility-based hard reset; resets triggered by activation, gradient or other statistics; continuous soft reset; changing the prior or regulariser; and earlier candidates.
  For each: trigger, strength, target, role in the dynamics, expected effect on old and new tasks.
  Could these heuristics answer a more basic Bayesian inference or decision problem?
  Separate published definitions from our reinterpretations.
]

== Core view

Every method answers two questions without separating them: which parts of the network probably changed (inference), and what to do to a point estimate about it (decision).
They differ in the world model they imply — drift, mean-reversion, replacement — and in the shortcut used for inference.
None computes the posterior over change.

== Derivation

=== As published

#table(
  columns: (0.9fr, 1.4fr, 0.6fr, 1.3fr, 0.9fr),
  [Method], [Trigger], [Strength], [Target], [Role (Stage P5)],
  [CBP @dohare2024loss], [Lowest contribution utility among mature units; fraction $rho$ per step], [Full], [Input weights from init; output weights zero], [State-dependent jump],
  [ReDo @sokar2023dormant], [Normalised activation below a threshold, every $F$ steps], [Full], [As CBP], [Threshold jump],
  [Shrink-and-perturb @ash2020warm], [Schedule], [Partial, all weights], [$c theta + s epsilon$], [Jump, state-dependent kernel],
  [L2-Init @kumar2023maintaining], [Always], [$gamma$], [$theta_0$], [Drift],
  [Weight decay], [Always], [$gamma$], [0], [Drift],
  [Noise injection], [Always], [$T$], [—], [Diffusion],
  [EWC @kirkpatrick2017ewc], [Always], [Fisher-weighted], [$theta_A$], [Drift (stability)],
  [Plasticity injection @nikishin2023deep], [Fixed times], [—], [New subnetwork, output cancelled], [Structural],
)

=== Reinterpreted (ours #heur)

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  [Method], [Implied world model], [What exact Bayes would do], [Old / new task],
  [Shrink-and-perturb], [$theta_(t+1) = c theta_t + s epsilon$: linear-Gaussian drift], [Shrink the mean and inflate the covariance; for a point estimate the inflation acts like a larger step. The added noise is a #emph[sample] of that prediction], [Uniform partial forgetting],
  [L2-Init], [Mean-reversion to $theta_0$ (Stage P6's soft-reset limit)], [The same plus the diffusion term the limit contains], [Slow pull everywhere],
  [CBP / ReDo], [Units can be replaced outright (Stage P5's change-point prior)], [A posterior change probability per unit], [Sparse full forgetting],
  [Noise injection], [None: a sampler's temperature], [Nothing; at $T = 1$ it samples the static posterior], [Forgets in flat directions (Stage P8)],
  [EWC], [Static world], [Exact recursive posterior], [Stability only],
)

=== The five questions, per mechanism

#table(
  columns: (auto, auto, auto, auto, auto),
  [], [Trigger], [Strength], [Target], [Role],
  [Hard reset by utility], [Low utility], [1], [Fixed $p_0$], [Jump],
  [Reset by activity or gradient], [Low activity], [1], [Fixed $p_0$], [Jump],
  [Continuous soft reset], [None], [$gamma$ per time], [Centre $mu$], [Drift + diffusion],
  [Changed prior / regulariser], [None], [Prior precision], [Prior mean], [Drift],
)

=== The underlying problem
+ *Inference:* under a world model that allows change, which parts probably changed? The answer is a posterior, not a utility.
+ *Decision:* given that posterior and a point-estimate learner, do nothing, shrink, or sample a reset?

Shrink-and-perturb and L2-Init make inference trivial (everything changes a little). CBP and ReDo make it a ranking (the least useful changed).

*A conjecture to check in Stage P11* #heur: under the change-point prior, data cannot tell whether a dead unit was replaced, so its posterior change probability may equal the prior hazard — which would make CBP's fixed rate exactly right on dead units.

== When it holds

The published table is factual. The reinterpretations hold only as analogies until a stage derives them from a model.

== Limits

Similar update rules do not imply the same inference (P0). A drift model and a change-point model can produce similar average forgetting with different per-unit behaviour (Stage P6).

== Alternatives

- *Treat resets as exploration* (bandit view): reset to gather information, not because something changed.
- *Treat resets as regularisation:* they limit effective capacity, like dropout over time.

== Next step

Stage P11 takes the inference reading literally and derives reset rules from explicit models.
