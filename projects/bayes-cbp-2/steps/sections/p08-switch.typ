#import "/template.typ": *

= Plasticity after a task switch <p8>

#callout(title: [The question])[
  A model fully trained on task $A$ must adapt quickly to task $B$ while keeping $A$.
  In a local model, compare SGD, noisy (Langevin) optimisation, jump or reset methods, and the other candidates.
  Derive response time, adaptation speed and forgetting cost.
  Discuss the new-task Hessian against the old precision, non-convexity and flat directions, local response against non-local exploration, and jump waiting time against training time.
  Propose an operational, measurable adaptation-speed indicator.
]

== Core view

In a local quadratic model, EWC with exact old precision traces the best possible forgetting–adaptation trade-off; plain gradient descent does not.
Noise erases old knowledge exactly where the new task does not constrain.
Resets have no special power in a convex model: their value is entirely non-convex.
Adaptation should be measured against a freshly initialised network.

== Derivation

=== Local model
$L_A = 1/2 (theta - theta_A)^top H_A (theta - theta_A)$, $L_B = 1/2 (theta - theta_B)^top H_B (theta - theta_B)$; training on $B$ starts at $theta_A$.
All results here are exact for this model #rig.

=== Four dynamics
*Gradient flow on $L_B$:* $theta(t) = theta_B + e^(-H_B t)(theta_A - theta_B)$. Direction $i$ adapts in time $1 \/ h_i$; directions with $h_i = 0$ never move, so $A$'s knowledge there is kept.

*With an EWC penalty* $beta \/ 2 (theta - theta_A)^top Lambda_A (theta - theta_A)$: rates of $H_B + beta Lambda_A$, limit
$theta^*(beta) = (H_B + beta Lambda_A)^(-1) (H_B theta_B + beta Lambda_A theta_A)$.
With $Lambda_A = H_A$, sweeping $beta$ traces the #gloss[Pareto front][points where neither loss can fall without the other rising] of $(L_B, L_A)$, since for convex quadratics that front is the set of weighted-sum minimisers.
At equal new-task loss 0.5, gradient flow forgets 0.40 and the front 0.10 (@fig-pareto).

*Langevin at temperature $T$:* the covariance along direction $i$ of $H_B$ is $T(1 - e^(-2 h_i t)) \/ h_i$, which is $2 T t$ when $h_i = 0$. Forgetting gains $1/2 tr(H_A C(t))$: noise erases what gradient flow kept.

*Reset of a block $S$ (one unit):* immediate forgetting $1/2 bb(E)[(xi - theta_A)_S^top (H_A)_(S S) (xi - theta_A)_S]$ plus cross terms. In a convex quadratic a reset only moves the start; it helps only if $p_0$ lands nearer $theta_B$ along slow directions.

#fig(include "/figures/pareto.typ", caption: [
  Forgetting against adaptation. $H_A$: eigenvalues 1 and 0.05, rotated 30°; $H_B = "diag"(1, 0.02)$; $theta_A = (0, 0)$, $theta_B = (2, 1)$. Pareto front from EWC, $beta in [10^(-3), 10^3]$; gradient flow and Langevin ($T = 0.05$) from $theta_A$ to $t = 400$ (top-left is late). Closed form.
]) <fig-pareto>

=== The four discussion points
- *Hessian against precision:* rates are the eigenvalues of $H_B$ (or $H_B + beta Lambda_A$); the moving fraction per direction is $f_i \/ (a_i + f_i)$ (Stage P2).
- *Flat directions:* they keep old knowledge under gradient flow and lose it under noise. The real obstacle is directions flat for $B$ #emph[locally] where $B$'s loss could fall non-locally.
- *Local against non-local:* a dead unit has zero gradient and zero Hessian. The local model gives it zero value, yet reviving it may help a lot. So a reset's value must be judged counterfactually.
- *Waiting against training:* adaptation time = wait for a useful reset ($approx 1 \/ (r h)$, Stage P7) + training from the landing point. Comparing only the first term with SGD's convergence time is a category error #wrong.

=== Indicator
Let $ell_k (s)$ be task $k$'s training loss $s$ steps after the switch, and $ell^"fresh" (s)$ the same for a fresh network. Define

$ cal(R)_k = frac(sum_(s <= S) ell_k (s), sum_(s <= S) ell^"fresh" (s)) . $

$cal(R)_k > 1$ is lost plasticity; $cal(R)_k < 1$ is transfer. A secondary indicator: steps to reach the loss a fresh network reaches at $S \/ 2$.

== When it holds

Quadratic losses over the distance travelled; exact old precision for the Pareto claim.

== Limits

Dead units and barriers lie outside the quadratic picture — exactly where resets matter. The local model cannot rank resets against anything.

== Alternatives

- *Linearised (neural tangent) models* instead of quadratics: also convex, same limitation.
- *Indicators based on initial gradient power* $g_B^top P g_B$: local and cheap, but blind to dead units.

== Next step

A reset's value is counterfactual and per unit. Stage P9 asks how to define which units are worth resetting.
