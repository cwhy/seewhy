#import "/template.typ": *

= One stochastic process for all mechanisms <sec-jump>

_Answers P5 and P6._

== The jump-diffusion

All mechanisms of @sec-mechanisms fit one #gloss[jump-diffusion][a process that moves continuously by drift and noise, and also jumps at random times to new places]:
between jumps,

$ upright(d) theta = b(theta) thin upright(d) t + sigma(theta) thin upright(d) W_t , $

and at rate $lambda(theta)$ the state jumps to a new point $theta' tilde K(theta, dot)$.
$b$ is the #emph[drift], $sigma$ the #emph[diffusion], $lambda$ the #emph[jump rate], and $K$ the #emph[jump kernel]: the distribution of where a jump lands.

#table(
  columns: (auto, auto, auto, auto),
  [Mechanism], [Drift $b$], [Diffusion $sigma sigma^top$], [Jump: rate $lambda$, kernel $K$],
  [Gradient descent], [$-nabla L$], [0], [none],
  [SGD (SDE limit)], [$-nabla L$], [$eta Sigma(theta) \/ B$], [none],
  [Langevin], [$-nabla U$], [$2 T I$], [none],
  [Weight decay], [$+ (-gamma theta)$], [—], [none],
  [L2-Init @kumar2023maintaining], [$+ (-gamma (theta - theta_0))$], [—], [none],
  [Shrink-and-perturb @ash2020warm], [—], [—], [rate $1 \/ k$; $K(theta, dot) = cal(N)(c theta, s^2 I)$],
  [Hard reset of unit $j$], [—], [—], [$lambda_j (theta)$; re-draw $w_j tilde p_0$, set output weights $v_j = 0$],
  [Soft reset], [—], [—], [$lambda$; $K$ = law of $(1 - a) theta + a xi$, $xi tilde p_0$],
)

Writing these in one equation is notation, not theory.
The table says where each mechanism sits.
It does not say that they share a Bayesian meaning (@sec-two-meanings).

*Caveat on CBP.* CBP does not reset at Poisson times.
Each step it adds $rho n$ to a counter and resets the lowest-utility eligible unit when the counter passes one @dohare2024loss.
So the rate for unit $j$ depends on the ranks of all units' utilities.
A Poisson model with rate $lambda_j (theta)$ is an approximation of this rule #cond.

*From state to rate (P5.4).* Any map $lambda_j = lambda_0 s(u_j)$ with $s >= 0$ decreasing in a utility $u_j$ is admissible.
In discrete time the reset probability must be $1 - e^(-lambda Delta t)$, not $lambda Delta t$, which can exceed one.
@sec-routes derives a specific map.

*Reset target (P5.5).* Three choices: a fixed initial distribution (CBP), a learned distribution (empirical Bayes), or a state-dependent kernel (shrink-and-perturb draws around $c theta$).
@sec-routes compares them.

== Generator and forward equation

The #gloss[generator][the operator giving the expected instantaneous rate of change of any function of the state] is #rig

$ (cal(L) f)(theta) = b dot nabla f + 1/2 tr(sigma sigma^top nabla^2 f) + lambda(theta) integral (f(theta') - f(theta)) K(theta, upright(d) theta'). $

Its adjoint gives the density's evolution, the #gloss[Kolmogorov forward equation][the generalisation of Fokker–Planck to processes with jumps] #rig:

$ partial_t p(theta) = underbrace(-nabla dot (b p) + 1/2 sum_(i j) partial_i partial_j [(sigma sigma^top)_(i j) p], "drift and diffusion") underbrace(- lambda(theta) p(theta), "outflow") + underbrace(integral lambda(theta') K(theta', theta) p(theta') thin upright(d) theta', "inflow"). $

Outflow uses the rate at the #emph[current] point. Inflow uses the rate at the #emph[source] point.
Both jump terms integrate to $minus.plus bb(E)_p [lambda]$, so mass is conserved.

For a hard reset, $K(theta', theta) = p_0(theta)$ for every source, and the inflow is $p_0(theta) thin bb(E)_p [lambda]$.
Its size depends on the whole current distribution.
Writing it as $lambda(theta) p_0(theta)$ — rate at the destination — is a common error #wrong; it does not conserve mass.

== Stationary laws

#proposition("Pure jump, hard reset", rig)[
  With $b = 0$, $sigma = 0$ and hard reset to $p_0$, the stationary density is
  $p_oo prop p_0 \/ lambda$, provided $integral p_0 \/ lambda < oo$.
  Only for constant $lambda$ is it $p_0$.
]
_Proof._ Set outflow equal to inflow: $lambda p_oo = p_0 thin bb(E)_(p_oo) [lambda]$. $square$
Units that reset slowly accumulate.

#proposition("Constant rate with diffusion", rig)[
  With constant $lambda$ and hard reset to $p_0$, the stationary law is that of the diffusion started from $p_0$ and run for an exponential time with mean $1 \/ lambda$:
  $p_oo = integral_0^oo lambda e^(-lambda s) thin P_s^* p_0 thin upright(d) s$.
]
_Proof._ After the last reset the process is a fresh diffusion from $p_0$. The time since that reset is exponential at stationarity. $square$
So if the diffusion alone targets the posterior $pi$, adding resets changes the target unless $p_0 = pi$.

#proposition("Which resets keep the posterior", rig)[
  Let the diffusion leave $pi$ invariant. Add jumps with rate $lambda(theta)$ and kernel $K$.
  Then $pi$ stays invariant if and only if $integral lambda(theta') K(theta', theta) pi(theta') thin upright(d) theta' = lambda(theta) pi(theta)$.
  For a hard reset this means $p_0 prop lambda pi$.
  A sufficient construction: propose $xi tilde p_0$ at rate $lambda_0$ and accept with probability
  $min{1, (pi(xi) p_0(theta)) \/ (pi(theta) p_0(xi))}$ — an independence Metropolis–Hastings move @tierney1994.
]
_Proof._ The diffusion terms vanish at $pi$; what remains is the balance of jump outflow and inflow. The Metropolis move satisfies detailed balance, which implies it. $square$

@fig-dw-stationary and @fig-dw-occupancy test this on a double well.
The naive reset puts 68% of the mass in the right well; the posterior has 50%.
The Metropolis-corrected reset gives 50% and reaches it by $t approx 5$.
Langevin alone is still at 46% at $t = 20$.

#fig(include "/figures/dw_stationary.typ", caption: [
  Density at $t = 30$ under three dynamics, against the posterior $pi prop exp(-U)$, $U = 4(theta^2 - 1)^2$.
  Langevin at $T = 1$ plus resets proposed from $cal(N)(0.5, 1)$ at rate 2.
  The naive reset accepts every proposal and shifts mass to the right well and the barrier.
  The Metropolis-corrected reset matches $pi$.
  Monte Carlo, 20 000 particles, Euler step $10^(-3)$, all started at $theta = -1$.
]) <fig-dw-stationary>

#fig(include "/figures/dw_occupancy.typ", caption: [
  Same runs: fraction of particles in the right well over time.
  Langevin alone crosses the barrier slowly (Kramers time about 15).
  The naive reset is fast but converges to the wrong value, 0.68.
  The Metropolis-corrected reset is fast and converges to the posterior value 0.5.
]) <fig-dw-occupancy>

== Two meanings of one jump term <sec-two-meanings>

The same jump term can mean two different things.

*A sampler move.* The jumps are part of an algorithm that targets a fixed posterior $pi$.
They must satisfy the balance condition above.
They speed up mixing, which is #emph[computational] plasticity.
But they cannot add #emph[statistical] plasticity: the target is unchanged.
For a posterior concentrated on an old task, the Metropolis test rejects almost every fresh draw. That is correct behaviour for a sampler.

*A world model.* The jumps describe the world: the true parameter may be replaced at any step.
Take the #gloss[change-point prior][each step, with probability $rho$, the parameter is re-drawn from $p_0$; otherwise it stays]
$theta_(t+1) | theta_t tilde (1 - rho) delta_(theta_t) + rho thin p_0$.
The Bayes filter's prediction step is then #rig

$ tilde(p)_(t+1) = (1 - rho) thin p_t + rho thin p_0 . $

This is exactly one step of the hard-reset forward equation at constant rate.
Now the target itself changes: the filter forgets old data at rate $rho$, and plasticity has a floor, like $q$ in @fig-kalman.

CBP as practised is neither exactly.
It has no Metropolis test, so it is not a sampler move.
Its rate depends on data through the utility, so it is not a prior hazard either.
A data-dependent rate has the shape of a #emph[posterior] change probability.
@sec-routes takes this literally: keep a constant prior hazard $rho$, and let the data set the posterior probability that a unit changed.

== More natural forms than the jump-SDE (P5.6)

The jump-SDE is a useful common notation. Two other forms carry more meaning:

+ *A hidden Markov model with a filter.* The jumps belong to the world, and the learner runs a Bayes filter. This gives every term a probabilistic meaning. It is the form used in @sec-routes.
+ *Piecewise-deterministic Markov processes* @davis1984. With no diffusion, motion is deterministic between random jumps. This fits gradient flow plus resets exactly and has a developed theory of generators and invariant laws.

== Soft versus hard reset (P6.6)

A soft reset $theta <- (1 - a) theta + a xi$, with $xi tilde cal(N)(mu, s^2)$, at rate $lambda$ has #rig

- mean drift $lambda a (mu - theta)$,
- second jump moment $lambda a^2 [(mu - theta)^2 + s^2]$,
- higher jump moments of order $lambda a^3$.

Let $a -> 0$ with $lambda a = gamma$ and $lambda a^2 s^2 = 2 D$ fixed.
Then the process converges to an Ornstein–Uhlenbeck process #cond (diffusion approximation; it requires $s^2$ to grow like $1 \/ a$):

$ upright(d) theta = gamma (mu - theta) thin upright(d) t + sqrt(2 D) thin upright(d) W_t . $

So a soft reset in the small-step limit is a pull toward the reset centre plus noise. That is L2-Init plus Langevin noise.
The hard reset ($a = 1$) is the opposite pole.
Its marginal at time $t$, from $theta_0$, is the mixture $e^(-lambda t) delta_(theta_0) + (1 - e^(-lambda t)) p_0$.

With $gamma = lambda$, both poles have the same mean relaxation $e^(-lambda t)$, and with $s^2 = D \/ gamma$ the same stationary law.
Yet their marginals differ (@fig-soft-hard).
A hard reset leaves each particle either untouched or fully renewed.
A soft reset moves every particle a little.
Averaged over units, forgetting proceeds at the same rate.
Per unit it is all-or-nothing for hard resets and gradual for soft ones.
For barrier crossing this matters: the hard reset jumps over; the soft limit must climb.

#fig(include "/figures/soft_vs_hard.typ", caption: [
  Distribution function of $theta$ at $t = 1$, started at $theta_0 = 3$, reset centre 0.
  Hard reset at rate 1 to $cal(N)(0, 1)$: exact mixture and Monte Carlo agree, with a step (an atom) at 3 of height $e^(-1)$.
  Soft resets at rate $1 \/ a$ with $lambda a^2 s^2 = 2$: as $a$ falls from 0.5 to 0.1 they approach the Ornstein–Uhlenbeck limit ($gamma = D = 1$, closed form).
  Monte Carlo with 200 000 samples each.
]) <fig-soft-hard>
