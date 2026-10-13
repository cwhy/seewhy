#import "/template.typ": *

= What is a low-utility unit? <sec-utility>

_Answers P9._

== A decision-theoretic definition

Resetting a unit is an action. Its cost is what the network loses by it.
CBP sets a reset unit's output weights to zero, so at the moment of reset the action equals #emph[removal] @dohare2024loss.
Define the utility of unit $j$ as the expected change in predictions caused by removing it:

$ u_j = bb(E)_(x tilde q) [ D_"KL" (p(y | x, theta) || p(y | x, theta^(-j))) ], $

where $theta^(-j)$ is $theta$ with unit $j$ removed and $q$ is the input distribution of interest.
This is a predictive quantity, in line with @sec-plasticity.

*Second-order form.* Let unit $j$ have activation $h_j (x)$ and outgoing weights $v_j$.
It adds $delta_j (x) = h_j (x) v_j$ to the next layer's pre-activation $z$.
KL has zero gradient at zero change, so to second order #rig

$ u_j approx 1/2 bb(E)_x [ delta_j (x)^top G(x) thin delta_j (x) ], $

where $G$ is the Fisher of the output distribution with respect to $z$.
For a hidden next layer, $G$ is the Gauss–Newton matrix $J^top F_y J$ #cond.

*Centring.* The constant part of $h_j$ can be moved into the next layer's bias at removal, at no loss.
So only the varying part counts:
$delta_j (x) = (h_j (x) - overline(h)_j) v_j$.
Assuming $G$ varies little with $x$ #cond,

$ u_j approx 1/2 op("Var")(h_j) thin v_j^top G v_j . $

*Redundancy.* Other units can take over unit $j$'s role if their outgoing weights are re-fitted.
The best linear take-over leaves only the part of $h_j$ not predictable from the others:

$ u_j^"comp" approx 1/2 op("Var")(h_j | h_(-j)) thin v_j^top G v_j, quad op("Var")(h_j | h_(-j)) = frac(1, (C^(-1))_(j j)), $

where $C$ is the covariance of the layer's activations.
The identity on the right is exact for linear least squares #rig.
This is the classical #gloss[Optimal Brain Surgeon][a pruning rule that removes a weight and optimally adjusts the rest using second-order information] idea @hassibi1993obs, applied to units.

== Eight candidate measures

#table(
  columns: (1.3fr, 1fr, 1fr),
  [Measure], [What it measures], [Where it fails],
  [1. Predictive contribution $u_j$], [The definition above], [Costly to compute exactly],
  [2. Loss change on removal], [Same as 1 on the training data, using labels], [First-order term vanishes at an optimum; needs the second-order term],
  [3. Posterior information gain $D_"KL" (q(theta_j) || p(theta_j))$], [How much data constrained the weights], [Weights pinned to zero: high information, zero contribution],
  [4. Fisher $F_(j j)$], [Sensitivity of the likelihood], [Duplicated units: each looks important, either can go],
  [5. Gradient statistics], [Current learning signal], [All gradients vanish at an optimum, important or not],
  [6. Signal-to-noise $mu^2 \/ s^2$], [Whether a weight is significantly non-zero @blundell2015weight], [Meaningless when the reset distribution is not centred at zero],
  [7. Activation × outgoing weight (CBP's contribution utility $|h_j| sum_k |v_(j k)|$)], [An uncentred, curvature-free proxy for 1], [Constant units score high but can be removed at no cost; ignores redundancy],
  [8. Redundancy $op("Var")(h_j | h_(-j))$], [Whether others can replace the unit], [Ignores whether the output uses the unit at all],
)

== Agreement and disagreement

All eight agree on one case: a dead unit. Every measure is zero, or at its minimum.

Outside that case they disagree. Four concrete conflicts:

+ *A constant unit* ($h_j$ the same on every input): measure 7 is high; centred measures 1 and 2 are zero.
+ *Two duplicate units*: measures 4 and 7 are high for both; measure 8 is zero for both, correctly, since either can go — but not both.
+ *A weight pinned to zero*: measure 3 is high; measures 1, 6 and 7 are zero.
+ *At convergence*: measure 5 is zero for every unit, so it cannot rank them.

So the measures are not equivalent #rig (by these examples).
They are approximations of measure 1 under different assumptions.

== Old task against new task

A unit can have low utility for the old task and high utility for the new one.
Two ways this happens:

- The unit is constant, or dead, on task $A$'s inputs, but varies on task $B$'s inputs. @sec-mechanisms showed a change of input distribution can revive a dead unit.
- The unit varies on both, but $A$'s output layer ignores it while $B$ would use it.

Any utility computed on $A$'s data alone will then reset a unit useful to $B$.
The reverse also happens: a utility computed on $B$'s data alone will reset units that $A$ needs, causing forgetting.

So utility must be measured on a distribution $q$ that represents what the network must keep #emph[and] what it must learn.
The Bayesian route in @sec-routes does this automatically.
The "stay" evidence carries the old task's precision; the new-task likelihood carries the new task's demands.

== A utility that can be estimated online

Combining the above, we propose #heur

$ hat(u)_j = 1/2 thin frac(1, (hat(C)^(-1))_(j j)) thin sum_k hat(G)_(k k) v_(j k)^2 , $

with three running estimates:

- $hat(C)$: an exponential moving average of the layer's activation covariance. It is inverted every $K$ steps, which costs $O(n^3 \/ K)$ per step for a layer of $n$ units.
- $hat(G)$: a moving average of squared gradients with respect to the next layer's pre-activations, a diagonal stand-in for $G$. To avoid the empirical-Fisher bias of @sec-gaussian, the backward pass for $hat(G)$ uses labels sampled from the model. That costs one extra backward pass through one layer.
- $v_j$: the current outgoing weights.

Without the redundancy term, replace $1 \/ (hat(C)^(-1))_(j j)$ by the diagonal entry $hat(C)_(j j)$. This is the centred, curvature-weighted version of CBP's contribution utility.

This estimates the cost of removing a unit. It does not estimate the value of what replaces it.
@sec-routes adds that second half.
