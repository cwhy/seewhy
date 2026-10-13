#import "/template.typ": *

= Why plain gradient updates may not restore plasticity <p4>

#callout(title: [The question])[
  A network trained long on old tasks learns a new task slowly.
  Analyse the causes: posterior concentration; curvature and gradient dynamics; local optima and barriers; dead units and rank collapse; parametrisation, optimiser state and history.
  If local gradient updates are not enough, propose at least three different new mechanisms, with motivation, advantages, risks and conditions. Do not rank them yet.
]

== Core view

Networks lose plasticity in both of Stage P3's senses, but the computational kind is the one gradients cannot repair.
One cause is exact: a dead ReLU unit sits in a set that no gradient method can leave.
So mechanisms beyond the local gradient are justified, at least for such invariant sets and for high barriers.

== Derivation

=== Five causes

*Posterior concentration* (statistical). Stage P1: under a static model, expected information gain must fall #rig.
This binds a Bayesian learner; SGD's analogue is accumulated curvature and shrinking effective steps.

*Curvature and gradient dynamics.* Gradient descent moves Hessian direction $i$ at rate $eta h_i$; flat directions barely move.
Plasticity loss has been observed alongside changes in curvature @lyle2023understanding #heur — a correlation, not a mechanism.

*Barriers.* Stage P3: local noise escapes a barrier in time $e^(Delta U \/ T)$, which can be arbitrarily long #cond.

*Dead units.*
#proposition("Dead ReLU units stay dead", rig)[
  If a ReLU unit's pre-activation $z = w^top x + b$ is negative for every input $x$ in the data's support,
  its loss gradient with respect to $(w, b)$ is zero for any loss.
  Gradient descent, SGD and Adam leave $(w, b)$ unchanged.
  L2 weight decay shrinks $(w, b)$ by $e^(-gamma t)$, which keeps $max_x z$ negative, so the unit stays dead.
]
_Proof._ The output $max(0, z)$ is zero with zero derivative on the support, so all gradients through the unit vanish; Adam's moments decay to zero; decay only rescales. $square$

Three things can revive such a unit: noise (slowly: free diffusion covers distance $d$ in time $d^2 \/ (2T)$), a direct reset, or a #emph[change of input distribution] that makes $z > 0$ somewhere.

*Rank collapse.* The #gloss[effective rank][how many independent directions the features span] of hidden features falls in long training @kumar2021implicit #heur.
The output layer can then fit only functions in that span, until lower layers move.

*Parametrisation, optimiser state, history.*
- For a layer whose output is normalised, the gradient is orthogonal to $w$, so $||w_(k+1)||^2 = ||w_k||^2 + eta^2 ||g_k||^2$ #rig: norms only grow and the effective step falls as $eta \/ ||w||^2$ @arora2019theoretical.
- Adam's second moment forgets over about $1 \/ (1 - beta_2)$ steps #heur. A weight with near-zero second moment takes steps of size about $eta$ whatever its gradient #heur.
- Early data can shape features later data cannot undo: the primacy bias @nikishin2022primacy #heur.

=== Is the local gradient enough?
No, in general: the dead set is invariant under every gradient method #rig.

=== Six candidate mechanisms (not ranked)

#table(
  columns: (1.3fr, 1fr, 1fr, 1fr, 0.8fr),
  [Mechanism], [Motivation], [Possible gain], [Risk], [Suits],
  [*M1. Non-local jumps:* re-draw a unit or weight], [Leave invariant sets; cross barriers in one step], [Escape time independent of depth], [Destroys stored knowledge; a fresh draw may be useless], [Dead units],
  [*M2. Dynamic prior:* assume the world drifts or jumps], [Stage P2's gain floor], [Principled forgetting at a set rate], [Forgets uniformly; shrinking to a symmetric mean drives units to zero], [Drifting tasks],
  [*M3. Structural change:* add, or prune and regrow, units], [New directions have no old precision: plastic, no forgetting (Stage P2)], [Plasticity without forgetting], [Unbounded growth; compute], [Capacity limits],
  [*M4. Temperature schedule:* re-heat noise at switches], [MSD and escape grow with $T$], [Cheap, local], [Noise in flat directions erases what is stored there], [Shallow barriers],
  [*M5. Reparametrisation:* normalisation, norm control], [Keep the effective step constant], [No information lost], [Does not revive dead units], [Norm growth],
  [*M6. Optimiser-state reset*], [Stale moments], [Trivial], [Large first steps], [Adam stalls],
)

== When it holds

- The dead-unit result is exact for ReLU; for tanh or sigmoid, units saturate instead, and gradients are tiny, not zero #cond.
- The norm-growth identity needs exact scale invariance and no weight decay.

== Limits

- The causes are not exclusive; which dominates in a given network is an empirical question #open.
- Nothing here measures how much each cause contributes.

== Alternatives

- *Plasticity as a property of the optimiser, not the parameters:* fix it by changing the optimiser alone (M5, M6).
- *Plasticity as a capacity question:* fix it by adding capacity (M3) and never touching old weights.

== Next step

M1, M2 and M4 differ in kind: a jump, a drift and a diffusion.
To compare them, Stage P5 writes them in one stochastic process.
