#import "/template.typ": *

= What is a low-utility unit? <p9>

#callout(title: [The question])[
  From Bayesian decision theory and information theory, define a low-utility unit.
  Compare: contribution to the predictive distribution; change in error on removal; posterior information gain; Fisher information; gradient statistics; signal-to-noise; activation or output-weight size; redundancy.
  Are they equivalent? When do they conflict? Can a unit be useless for the old task but useful for the new?
  Propose a principled utility that can be estimated online.
]

== Core view

A unit's utility is the predictive change its removal would cause — a decision-theoretic cost.
To second order this is its centred activation variance, weighted by the output Fisher, and reduced by what other units can take over.
The eight common measures agree only on dead units. Utility measured on one task can be wrong for the next.
Removal cost is only half the reset question; the value of what replaces the unit is the other half.

== Derivation

=== Definition
CBP sets a reset unit's output weights to zero @dohare2024loss, so a reset at first equals removal. Define

$ u_j = bb(E)_(x tilde q) [ D_"KL" (p(y | x, theta) || p(y | x, theta^(-j))) ]. $

=== Second order
Unit $j$ adds $delta_j (x) = h_j (x) v_j$ to the next pre-activation $z$. KL has zero gradient at zero change, so #rig
$u_j approx 1/2 bb(E)_x [delta_j (x)^top G(x) delta_j (x)]$, with $G$ the output Fisher at $z$ (Gauss–Newton for a hidden next layer #cond).

*Centring:* the mean of $h_j$ can move into the next bias, so with $G$ roughly constant #cond, $u_j approx 1/2 op("Var")(h_j) thin v_j^top G v_j$.

*Redundancy:* other units can take over the part of $h_j$ they predict linearly. What remains is $op("Var")(h_j | h_(-j)) = 1 \/ (C^(-1))_(j j)$ for activation covariance $C$ #rig (least squares) — the Optimal Brain Surgeon idea @hassibi1993obs applied to units.

=== The eight measures

#table(
  columns: (1.3fr, 1fr, 1fr),
  [Measure], [Measures], [Fails when],
  [1. Predictive contribution], [The definition], [Costly exactly],
  [2. Removal loss change], [1 on training data], [First order vanishes at an optimum],
  [3. Information gain $D_"KL" (q(theta_j) || p(theta_j))$], [How much data constrained it], [Weights pinned to zero: high gain, no contribution],
  [4. Fisher $F_(j j)$], [Likelihood sensitivity], [Duplicates: each looks vital, either can go],
  [5. Gradient statistics], [Current learning], [All vanish at an optimum],
  [6. Signal-to-noise $mu^2 \/ s^2$ @blundell2015weight], [Significantly non-zero?], [Reset distribution not centred at zero],
  [7. Activation × output weight (CBP's $|h_j| sum_k |v_(j k)|$)], [Uncentred proxy for 1], [Constant units score high; ignores redundancy],
  [8. Redundancy $op("Var")(h_j | h_(-j))$], [Can others replace it?], [Ignores whether the output uses it],
)

They agree on dead units (all zero). They conflict on: constant units (7 high, 1–2 zero); duplicates (4, 7 high, 8 zero); weights pinned at zero (3 high, 1, 6, 7 zero); converged networks (5 zero for all). So they are not equivalent #rig.

=== Old task against new task
A unit can be constant or dead on $A$'s inputs and active on $B$'s (Stage P4: a change of inputs revives dead units).
A utility from $A$'s data then resets something $B$ needs; one from $B$'s data alone resets what $A$ needs.
So utility must be measured on a distribution covering what must be kept and what must be learned.

=== Online estimate
We propose #heur
$ hat(u)_j = 1/2 thin frac(1, (hat(C)^(-1))_(j j)) sum_k hat(G)_(k k) v_(j k)^2 , $
with $hat(C)$ a moving-average activation covariance (inverted every $K$ steps, $O(n^3 \/ K)$), and $hat(G)$ a moving average of squared next-layer gradients for #emph[model-sampled] labels, to avoid Stage P2's empirical-Fisher bias. Dropping redundancy, use $hat(C)_(j j)$: a centred, curvature-weighted CBP utility.

== When it holds

Second-order expansion (small removal effect); $G$ roughly constant over inputs; linear take-over for the redundancy term.

== Limits

- It prices removal, not replacement: a low-utility unit replaced by another useless unit gains nothing.
- The covariance inverse is costly for wide layers.

== Alternatives

- *Shapley-style* unit importance: fair credit with interactions, but exponentially costly.
- *Counterfactual oracle:* reset in a copy and train $k$ steps. Exact but expensive; a good ground truth for experiments.

== Next step

Stage P10 places CBP and its relatives side by side, using this utility language and Stage P5's process.
