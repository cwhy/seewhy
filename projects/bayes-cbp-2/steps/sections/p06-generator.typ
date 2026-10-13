#import "/template.typ": *

= Generator, density evolution and theoretical properties <p6>

#callout(title: [The question])[
  For Stage P5's process: derive the generator and the forward (Kolmogorov) equation.
  Handle mass flowing in and out correctly when the jump rate depends on the state.
  Is there a stationary law, and is it the posterior? How do non-local jumps differ from diffusion?
  When do soft and hard resets approximate each other? Check on a one-dimensional example.
  Separate general results from those that need a constant rate or a fixed reset distribution.
]

== Core view

The forward equation has an outflow at the rate of the current point and an inflow at the rate of the source point.
With resets, the stationary law is generally #emph[not] the posterior.
Resets keep the posterior only if the reset distribution is proportional to $lambda pi$; a Metropolis test achieves this.
Soft resets become an Ornstein–Uhlenbeck pull in the small-step limit; hard resets are the opposite pole, with the same mean relaxation but a different distribution.

== Derivation

=== Generator (P6.1)
The generator is #rig

$ (cal(L) f)(theta) = b dot nabla f + 1/2 tr(sigma sigma^top nabla^2 f) + lambda(theta) integral (f(theta') - f(theta)) K(theta, upright(d) theta'). $

=== Forward equation (P6.2, P6.3)
Its adjoint gives the density's evolution #rig:

$ partial_t p(theta) = -nabla dot (b p) + 1/2 sum_(i j) partial_i partial_j [(sigma sigma^top)_(i j) p] - lambda(theta) p(theta) + integral lambda(theta') K(theta', theta) p(theta') thin upright(d) theta'. $

Outflow uses $lambda$ at the current point; inflow uses $lambda$ at the source. Both integrate to $minus.plus bb(E)_p [lambda]$, so mass is conserved.
For a hard reset, $K(theta', theta) = p_0(theta)$, and the inflow is $p_0(theta) thin bb(E)_p [lambda]$.
Writing $lambda(theta) p_0(theta)$ instead is a common error #wrong; it does not conserve mass.

=== Stationary laws (P6.4)
These are the general/special results P6 asks to separate.

#proposition("Pure jump, state-dependent rate", rig)[
  With $b = 0$, $sigma = 0$ and hard reset to $p_0$: $p_oo prop p_0 \/ lambda$, if $integral p_0 \/ lambda < oo$. Only a #emph[constant] rate gives $p_0$.
]
_Proof._ $lambda p_oo = p_0 bb(E)_(p_oo)[lambda]$. $square$

#proposition("Constant rate with diffusion", rig)[
  With constant $lambda$ and hard reset to $p_0$, $p_oo = integral_0^oo lambda e^(-lambda s) P_s^* p_0 thin upright(d) s$: the diffusion run from $p_0$ for an exponential time.
]
_Proof._ After the last reset the process is a fresh diffusion; at stationarity the time since that reset is exponential. $square$

#proposition("When resets keep the posterior", rig)[
  Let the diffusion leave $pi$ invariant. Adding jumps keeps $pi$ invariant iff
  $integral lambda(theta') K(theta', theta) pi(theta') thin upright(d) theta' = lambda(theta) pi(theta)$.
  For hard resets: iff $p_0 prop lambda pi$.
  Proposing $xi tilde p_0$ and accepting with probability $min{1, pi(xi) p_0(theta) \/ (pi(theta) p_0(xi))}$ satisfies it @tierney1994.
]
_Proof._ The diffusion terms vanish at $pi$; jump outflow must equal inflow. The Metropolis move is in detailed balance. $square$

=== One-dimensional check (P6, last item)
Double well $U = 4 (theta^2 - 1)^2$, Langevin at $T = 1$, resets proposed from $cal(N)(0.5, 1)$ at rate 2.
The posterior puts 50% in the right well.
Naive resets give 68%; Metropolis-corrected resets give 50% (@fig-dw-stationary, @fig-dw-occupancy).

#fig(include "/figures/dw_stationary.typ", caption: [
  Density at $t = 30$ against the posterior. Naive resets shift mass right and onto the barrier; Metropolis-corrected resets match. Monte Carlo, 20 000 particles, Euler step $10^(-3)$, all started at $theta = -1$.
]) <fig-dw-stationary>

#fig(include "/figures/dw_occupancy.typ", caption: [
  Fraction in the right well over time, same runs. Langevin alone is slow (Kramers time about 15); naive resets are fast but converge to 0.68; Metropolis-corrected resets are fast and converge to 0.5.
]) <fig-dw-occupancy>

=== Two meanings of the jump term
Stage P5 left this open. Now it can be stated.

- *Sampler move:* jumps must satisfy the balance condition. They speed mixing — computational plasticity — but cannot add statistical plasticity, since the target is fixed. For a concentrated posterior the Metropolis test rejects nearly every fresh draw, correctly.
- *World model:* jumps are Stage P5's change-point prior. The filter's target itself changes; old data are forgotten at rate $rho$; plasticity has a floor like Stage P2's drift.

Same equation, different object #rig (as a statement of what each preserves).

=== Jumps versus diffusion (P6.5)
Diffusion moves probability locally; its forward operator is differential.
Jumps move it non-locally; their operator is an integral. Diffusion paths are continuous, so to get from one basin to another a path must cross every point between. Jump paths need not #rig. That is why only jumps can skip a barrier or leave an invariant set in one move.

=== Soft versus hard reset (P6.6)
A soft reset $theta <- (1 - a) theta + a xi$, $xi tilde cal(N)(mu, s^2)$, at rate $lambda$ has drift $lambda a (mu - theta)$, second jump moment $lambda a^2 [(mu - theta)^2 + s^2]$, and higher moments of order $lambda a^3$ #rig.
With $a -> 0$, $lambda a = gamma$ and $lambda a^2 s^2 = 2 D$ fixed, it converges to $upright(d) theta = gamma (mu - theta) upright(d) t + sqrt(2D) upright(d) W_t$ #cond (needs $s^2 prop 1 \/ a$).
A hard reset ($a = 1$) leaves the mixture $e^(-lambda t) delta_(theta_0) + (1 - e^(-lambda t)) p_0$.
With $gamma = lambda$ and $s^2 = D \/ gamma$ both have the same mean relaxation and stationary law, but different marginals (@fig-soft-hard):
all-or-nothing per unit versus a little for every unit.

#fig(include "/figures/soft_vs_hard.typ", caption: [
  Distribution of $theta$ at $t = 1$ from $theta_0 = 3$, reset centre 0. Hard reset: exact and Monte Carlo agree, with an atom of mass $e^(-1)$ at 3. Soft resets with $lambda a^2 s^2 = 2$ approach the OU limit as $a$ falls from 0.5 to 0.1. 200 000 samples each.
]) <fig-soft-hard>

== When it holds

- The generator and forward equation need regular coefficients and a bounded rate (standard Feller conditions).
- Proposition "Pure jump" is general in $lambda$; Proposition "Constant rate" needs a constant rate; the posterior-preservation condition is general.
- The OU limit needs the perturbation scale to grow as $a$ shrinks.

== Limits

- A Metropolis test needs the full-data likelihood; on minibatches it is biased.
- In high dimension a fresh draw rarely passes the test; the mixing gain then vanishes.

== Alternatives

- *Non-reversible jumps* (lifted chains) that keep $pi$ without detailed balance; possibly faster.
- *Accept that resets change the target*, and choose the target deliberately: the world-model reading.

== Next step

Stage P6 says what resets converge to. Stage P7 asks how fast they get somewhere useful compared with diffusion.
