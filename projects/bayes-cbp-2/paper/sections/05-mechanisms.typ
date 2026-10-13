#import "/template.typ": *

= Why gradient updates may not restore plasticity <sec-mechanisms>

_Answers P4._ First the causes, then mechanisms that go beyond the local gradient. No mechanism is ranked here; that comes in @sec-routes.

== Five causes

=== Posterior concentration (statistical)
Covered in @sec-plasticity. Under a static model the posterior must narrow, and expected information gain must fall #rig.
This affects a Bayesian learner. SGD is not one, so its relevance to SGD is indirect.
SGD has a loose analogue: accumulated curvature and shrinking effective steps.

=== Curvature and gradient dynamics
Gradient descent moves direction $i$ of the Hessian at rate $eta h_i$.
Directions with tiny $h_i$ barely move, and a step size large enough for the sharpest direction is too small for the flat ones.
Plasticity loss has been observed to coincide with changes in curvature @lyle2023understanding #heur.
That is a correlation, not a mechanism.

=== Dead units: an invariant set gradients cannot leave
#proposition("Dead ReLU units stay dead", rig)[
  Let a ReLU unit have pre-activation $z = w^top x + b < 0$ for every input $x$ in the data's support.
  Then the loss gradient with respect to $(w, b)$ is zero for any loss.
  Plain gradient descent, SGD and Adam leave $(w, b)$ unchanged.
  With L2 weight decay, $(w, b)$ shrinks as $e^(-gamma t)(w, b)$. Then $max_x z$ shrinks by the same factor but keeps its sign, so the unit stays dead.
]

_Proof._ The unit's output is $max(0, z) = 0$ with zero derivative on the whole support, so every gradient through it vanishes.
Adam's first moment then decays to zero, and so does its update.
Weight decay only rescales $(w, b)$. $square$

Only three things can revive such a unit:
injected noise (slowly — free diffusion needs time $d^2 \/ (2 T)$ to cover distance $d$),
a direct reset, or a #emph[change of input distribution] that makes $z > 0$ somewhere.
The third matters for @sec-utility: a unit dead on task $A$ may be alive on task $B$.
Smooth activations such as tanh saturate instead; their gradients are exponentially small, not zero #cond.

=== Representation collapse
The #gloss[effective rank][roughly, how many independent directions the hidden features span on the data] of hidden features falls during long training @kumar2021implicit #heur.
The last layer can then fit only functions in that span.
Fitting anything else needs the lower layers to move first, and their gradients are scaled by the output weights.
This makes new tasks slow but not impossible: a computational, not statistical, loss.

=== Parametrisation, optimiser state and history
- *Growing norms.* For a layer whose output is normalised, the loss is invariant to the weight scale. Then the gradient is orthogonal to $w$, and $||w_(k+1)||^2 = ||w_k||^2 + eta^2 ||g_k||^2$ #rig. The norm only grows, and the effective step shrinks as $eta \/ ||w||^2$ @arora2019theoretical.
- *Optimiser state.* Adam's second moment tracks squared gradients over about $1 \/ (1 - beta_2)$ steps, so stale state from an old task fades quickly #heur. After a reset, a unit's second moment is near zero, so its first steps have size about $eta$ in each coordinate regardless of gradient size. This can destabilise newly reset units #heur.
- *History.* Early data can shape features that later data cannot undo, the #gloss[primacy bias][over-fitting to early experience that persists through later training] @nikishin2022primacy #heur.

== Is the local gradient enough?

Not always.
The dead-unit set is invariant under every gradient method #rig.
No local rule escapes it without noise, a reset, or a change of inputs.
For barriers, local noise escapes in time $e^(Delta U \/ T)$ (@sec-dynamics), which can be arbitrarily long.
So new dynamics are justified at least for invariant sets and high barriers.

== Candidate mechanisms

Six candidates. None is preferred yet.

#table(
  columns: (1.3fr, 1fr, 1fr, 1fr, 0.8fr),
  [Mechanism], [Motivation], [Possible gain], [Risk], [Suits],
  [*M1. Non-local jumps* (reset a unit or weight to a fresh draw)],
  [Leave invariant sets and cross barriers in one step],
  [Time to leave a dead set no longer depends on its depth],
  [Destroys stored knowledge; a fresh draw may be useless in high dimension],
  [Dead or saturated units],
  [*M2. Dynamic prior* (assume the world drifts or jumps)],
  [Statistical plasticity floor (@fig-kalman)],
  [Principled forgetting at a known rate],
  [Forgets uniformly; shrinking toward a symmetric prior mean drives units toward zero],
  [Drifting tasks],
  [*M3. Structural change* (add, or prune and regrow, units)],
  [New directions with no old precision ($a_i = 0$ in @sec-gaussian) are plastic at no forgetting cost],
  [Plasticity without forgetting],
  [Growth without bound; compute],
  [Capacity-limited regimes],
  [*M4. Temperature schedule* (re-heat Langevin noise at switches)],
  [MSD and escape grow with $T$],
  [Cheap and local],
  [Diffusion in flat directions erases what is stored there (@sec-switch)],
  [Shallow barriers],
  [*M5. Reparametrisation* (normalisation, norm control)],
  [Keeps the effective step constant],
  [No information lost],
  [Does not revive dead units],
  [Norm growth],
  [*M6. Optimiser-state reset*],
  [Stale moment estimates],
  [Trivial to implement],
  [Large first steps for zero second moments],
  [Adam-specific stalls],
)

M1, M2 and M4 differ in kind: a jump, a drift, and a diffusion.
@sec-jump writes all of them in one equation, and @sec-escape compares their speed.
