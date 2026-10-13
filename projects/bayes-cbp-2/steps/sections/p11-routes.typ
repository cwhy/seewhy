#import "/template.typ": *

= Deriving reset rules from first principles <p11>

#callout(title: [The question])[
  Decide when to reset, which parameters, to where, and how hard — improving adaptation while limiting damage to old tasks.
  Explore at least three routes (hierarchical/empirical Bayes; dynamic Bayes with change points; Bayesian decision with prediction risk and reset cost; others).
  For each give: model or objective; latents and hyperparameters; utility; trigger; reset distribution; online feasibility; essential difference from CBP.
  Do not presuppose a variance ratio as the strength or a posterior mean as the target. Derive the algorithm from the model.
]

== Core view

Three routes give three different answers.
Empirical Bayes derives #emph[pruning], not resetting.
A change-point world model derives the whole rule — trigger, strength and target — from one model.
It weighs the value of a fresh unit against the cost of removing the old one, and says to #emph[sample] the reset, not average it.
Its limit: it never values retention for its own sake.
A decision formulation can add retention, at the price of free parameters.

== Derivation

Notation: unit $j$ has input weights $w_j$, activations $h_j (x)$, outgoing weights $v_j in bb(R)^m$ into pre-activations $z$.
On a window of examples $b = 1..N$: $g_b = partial ell_b \/ partial z$, and $G_b$ is the (diagonal) output Fisher at $z$.

=== Route 1: hierarchical and empirical Bayes
- *Model:* $w_(j i) tilde cal(N)(0, sigma_j^2)$ per unit; $sigma_j^2$ by #gloss[type-II maximum likelihood][maximising the data's probability with parameters integrated out].
- *Latents / hyperparameters:* weights / $sigma_j^2$.
- *Utility:* fitted $sigma_j^2$; EM fixed point $sigma_j^2 = (||m_j||^2 + tr S_j) \/ d_j$ @mackay1992practical — #gloss[ARD][automatic relevance determination] @tipping2001sparse.
- *Trigger:* $sigma_j^2$ below a threshold.
- *Target:* none from the model — it says prune. Regrowth from init is an extra assumption.
- *Online:* cheap with a diagonal Laplace posterior.
- *Versus CBP:* evidence-based relevance, threshold not rank.
- *Problem:* for a dead unit the evidence does not depend on $sigma_j^2$, so it is unidentifiable #rig; fitting on the same data is circular.

=== Route 2: a change-point world model
- *Model:* each step, each unit is replaced with probability $rho$: $w_j tilde p_0^"in"$, $v_j tilde cal(N)(0, s_v^2 I)$; otherwise unchanged.
- *Latents / hyperparameters:* weights and change indicators / $rho$, $p_0^"in"$, $s_v$.
- *Posterior change probability:* $r_j = sigma("logit" rho + Delta_j)$, $Delta_j = log Z_j^"fresh" - log Z_j^"stay"$, a ratio of window marginal likelihoods.
- *Computing $Delta_j$* (second order in $z$ #cond): draw a candidate $w' tilde p_0^"in"$ with activations $h'_b$; integrate its outgoing weights over their prior. With $a_k = sum_b h'_b g_(b k)$ and $B_k = sum_b h'^2_b G_(b k k)$:

$ Delta_j approx underbrace(-c_j^"rem", "removal cost") + underbrace(sum_k [frac(a_k^2, 2 (B_k + s_v^(-2))) - 1/2 log(1 + s_v^2 B_k)], "fresh unit's fit minus Occam penalty") - O_j^"stay" . $

  $c^"rem"$ is Stage P9's utility; $O^"stay"$ is the same bracket for re-fitting $v_j$ from its old posterior, near zero when the old task pinned it.
- *Utility:* both halves at once — cost of removal and value of replacement. CBP uses only the first.
- *Dead units:* $c^"rem" = O^"stay" = 0$, so $Delta_j$ is the fresh term: data decide. If fresh output weights are fixed at zero ($s_v -> 0$, CBP's choice), $Delta_j = 0$ and $r_j = rho$ exactly #rig. This confirms Stage P10's conjecture, under that condition only.
- *Strength — sample, do not average:* the posterior for a unit is a two-component mixture. Its mean $(1 - r_j) w_j + r_j bb(E)_(p_0)[w]$ shrinks toward $w = 0, b = 0$ when $p_0$ is symmetric; a ReLU unit there outputs zero everywhere — the edge of Stage P4's dead set #rig. So reset with probability $r_j$: Thompson sampling of the change indicator #cond.
- *Target:* input weights $w'$ (the candidate used for $Delta_j$); output weights the posterior mean $v'_k = -a_k \/ (B_k + s_v^(-2))$, a one-step Newton fit. Both derived.
- *Online:* one candidate forward pass and two matrix products per layer: $O(N n m)$.
- *Versus CBP:* posterior probability not rank; value of replacement included; can reset useful units that conflict with new data; derived output initialisation.
- *Limit:* a change affects only future data; past data stay explained by the old unit. So the filter never values retention #rig. Retention matters only if old tasks can recur.

=== Route 3: Bayesian decision with explicit cost
- *Objective:* minimise $bb(E)[sum_(s <= H) L_B (theta_s) | R] + kappa thin bb(E)[Delta L_A (R)]$ over reset sets $R$.
- *Latents / hyperparameters:* future trajectory / horizon $H$, forgetting price $kappa$.
- *Utility:* $V_j = G_j (H) - c_j^(B,"rem") - kappa thin c_j^(A,"rem")$, with $G_j (H) approx min(H eta ||a||^2, sum_k a_k^2 \/ (2 B_k))$ #heur — Route 2's fresh term in the long-horizon limit, without Occam.
- *Trigger:* $V_j > 0$, within a budget.
- *Target:* the best of $M$ candidates from $p_0$.
- *Online:* Route 2's cost times $M$, plus stored old-task statistics.
- *Versus CBP:* horizon and forgetting price replace the replacement rate.
- *Problems:* $H$, $kappa$ free; best-of-$M$ suffers the winner's curse #cond.

== When it holds

Route 2 needs the second-order (Laplace) expansion in output space, a diagonal Fisher, and one candidate as a stand-in for the average over $p_0$.

== Limits

- Route 1 gives pruning, not plasticity.
- Route 2 ignores retention by construction.
- Route 3's parameters have no first-principles value.

== Alternatives

A fourth route from Stage P6: the Metropolis-corrected reset. It keeps the static posterior and adds only computational plasticity. It suits a stuck optimiser with a correct posterior, not a world that changed.

== Next step

Route 2 alone derives trigger, strength and target from one model. Stage P12 turns it into an algorithm, keeping Route 3 and the Metropolis reset as alternatives.
