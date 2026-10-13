#import "/template.typ": *

= One stochastic process for all mechanisms <p5>

#callout(title: [The question])[
  Can SGD, Langevin dynamics, continuous shrinkage and discrete resets be one stochastic process? Is a jump-SDE suitable?
  Which mechanisms are drift, diffusion or jump terms? How should the parameter state map to event rates?
  Should resets target a fixed initial distribution, a learned one, or a state-dependent one?
  Is there a more natural form than the jump-SDE? Do not equate "same equation" with "same Bayesian meaning".
]

== Core view

All six mechanisms of Stage P4 fit one jump-diffusion: drift, diffusion, and jumps at a state-dependent rate to a chosen distribution.
That is notation, not a theory.
The same jump term can be part of a #emph[sampler] or part of a #emph[model of the world], and these mean different things.
A hidden Markov model with a Bayes filter is the more natural form, because it says which.

== Derivation

=== The process (P5.1, P5.2)
Between jumps, $upright(d) theta = b(theta) thin upright(d) t + sigma(theta) thin upright(d) W_t$.
At rate $lambda(theta)$, the state jumps to $theta' tilde K(theta, dot)$, the #emph[jump kernel].
This is a #gloss[jump-diffusion][continuous motion plus random jumps].

=== Each mechanism's role (P5.3)

#table(
  columns: (auto, auto, auto, 1fr),
  [Mechanism], [Drift $b$], [Diffusion $sigma sigma^top$], [Jump: rate $lambda$, kernel $K$],
  [Gradient descent], [$-nabla L$], [0], [none],
  [SGD (SDE limit, Stage P3)], [$-nabla L$], [$eta Sigma(theta) \/ B$], [none],
  [Langevin], [$-nabla U$], [$2 T I$], [none],
  [Weight decay], [$-gamma theta$ added], [—], [none],
  [L2-Init @kumar2023maintaining], [$-gamma (theta - theta_0)$ added], [—], [none],
  [Shrink-and-perturb @ash2020warm], [—], [—], [rate $1 \/ k$; $K(theta, dot) = cal(N)(c theta, s^2 I)$],
  [Hard reset of unit $j$], [—], [—], [$lambda_j (theta)$; re-draw input weights from $p_0$, zero output weights],
  [Soft reset], [—], [—], [$lambda$; $K$ = law of $(1 - a) theta + a xi$, $xi tilde p_0$],
)

Continual Backprop @dohare2024loss resets the lowest-utility mature unit whenever a counter, incremented by $rho n$ per step, passes one.
That is a rank-based rule, not a Poisson process. A per-unit rate $lambda_j (theta)$ approximates it #cond.

=== From state to rate (P5.4)
Any map $lambda_j = lambda_0 s(u_j)$, with $s >= 0$ decreasing in a utility $u_j$, is admissible.
In discrete time the probability is $1 - e^(-lambda Delta t)$, never $lambda Delta t$, which can exceed one.
Which utility, and which $s$, is open here: Stage P9 studies utility, and Stage P11 derives a rate.

=== Reset target (P5.5)
#table(
  columns: (auto, 1fr, 1fr),
  [Target], [For], [Against],
  [Fixed initial distribution], [Known to train well from; no estimation], [Ignores everything learned about good features],
  [Learned distribution], [Can concentrate on useful features], [Estimated from the same data it serves: circular],
  [State-dependent (around the current point)], [Small, gentle changes], [Cannot leave an invariant set if the perturbation is small (Stage P4)],
)

=== A more natural form (P5.6)
Write the jumps into a model of the world instead.
The #gloss[change-point prior][each step, with probability $rho$, the parameter is re-drawn from $p_0$; otherwise it stays]
$theta_(t+1) | theta_t tilde (1 - rho) delta_(theta_t) + rho thin p_0$
makes $theta$ a hidden Markov chain.
The Bayes filter's prediction step is then #rig

$ tilde(p)_(t+1) = (1 - rho) thin p_t + rho thin p_0 , $

which is exactly one step of a hard reset at constant rate applied to the belief.
In this form every term has a probabilistic meaning: the jumps belong to the world, and the learner infers them.
#gloss[Piecewise-deterministic Markov processes][deterministic motion between random jumps] @davis1984 are another natural form when there is no diffusion.

== When it holds

- The table is exact as a description of each update rule.
- The SGD row inherits Stage P3's small-step approximation.
- The prediction-step identity is exact for a constant hazard $rho$.

== Limits

- Fitting into one equation shows nothing about shared meaning (P0's warning). A jump inside a sampler and a jump in a world model are different objects. Stage P6 makes the difference precise.
- The process describes one parameter vector, not a posterior; Bayesian meaning needs a separate argument.

== Alternatives

- *Lévy-type processes* with heavy-tailed jumps of all sizes, rather than one jump kernel.
- *Pure filtering view:* never write the learner as a stochastic process at all; write the world model and derive the learner.

== Next step

Stage P6 derives the generator and forward equation of this process, and asks the key question: is its stationary law the Bayesian posterior?
