#import "/template.typ": *

= From Bayesian inference to optimisation dynamics <sec-dynamics>

_Answers P3._

== SGD as a stochastic differential equation

SGD updates $theta_(k+1) = theta_k - eta hat(g)_k$, where the minibatch gradient $hat(g)_k$ has mean $nabla L$ and covariance $Sigma(theta) \/ B$ for batch size $B$.
Set continuous time $t = k eta$. To first order in $eta$, SGD tracks @li2017sde

$ upright(d) theta = -nabla L(theta) thin upright(d) t + sqrt(eta Sigma(theta) \/ B) thin upright(d) W_t , $

where $W_t$ is #gloss[Brownian motion][the continuous-time limit of a random walk with independent Gaussian steps].
This is a weak approximation #cond: it matches expectations of smooth test functions to first order in $eta$, for smooth losses with bounded noise.
It is not a statement about individual trajectories.

== Langevin dynamics and its stationary law

#gloss[Overdamped Langevin dynamics][gradient descent plus Gaussian noise of fixed size, as for a particle in a viscous fluid] at temperature $T$ is

$ upright(d) theta = -nabla U(theta) thin upright(d) t + sqrt(2 T) thin upright(d) W_t , quad U = -log p(D | theta) - log p(theta). $

Its density obeys the #gloss[Fokker–Planck equation][the equation that moves the probability density along with the dynamics]

$ partial_t p = nabla dot (p nabla U) + T Delta p . $

The stationary density is $p_oo prop e^(-U \/ T)$ whenever this is normalisable #rig.
At $T = 1$ this is the posterior.
#gloss[SGLD][stochastic-gradient Langevin dynamics: the Euler step of this equation with a minibatch gradient] @welling2011sgld
discretises it as $theta_(k+1) = theta_k - eta nabla tilde(U) + sqrt(2 eta) xi_k$.
It samples the posterior only as $eta -> 0$ #cond. At a fixed step there is an $O(eta)$ bias, and minibatch noise adds to the injected noise.

== Curvature, noise and temperature

Take one direction with curvature $kappa$, $U = 1/2 kappa theta^2$.
This is the #gloss[Ornstein–Uhlenbeck process][a random walk pulled back toward a centre; the simplest stationary Gaussian process]. Exactly #rig:

- the mean relaxes as $e^(-kappa t)$, so the response time is $1 \/ kappa$;
- the stationary variance is $T \/ kappa$;
- in a flat direction ($kappa = 0$) the variance grows without bound, as $2 T t$.

High curvature means a fast, narrow response. Low curvature means a slow, wide one. Temperature scales the width only.

== Candidate dynamical measures of plasticity

*Mean-squared displacement* (MSD). At equilibrium in the quadratic well, $"MSD"(t) = bb(E)[(theta_t - theta_0)^2] = 2 (T \/ kappa)(1 - e^(-kappa t))$ #rig.

*Linear response.* Add a small constant force $epsilon g$, standing for a new task's gradient.
The mean shifts by $chi(t) epsilon g$ with $chi(t) = (1 - e^(-kappa t)) \/ kappa$ #rig.

*Fluctuation–dissipation.* For reversible dynamics at equilibrium,
$chi(t) = "MSD"(t) \/ (2 T)$ #rig.
So at equilibrium these two measures are one measure.
Out of equilibrium, as under SGD, they separate.

*Escape over a barrier.* For a barrier of height $Delta U$ between a minimum $a$ and a saddle $s$, Kramers' law @kramers1940 gives

$ tau approx frac(2 pi, sqrt(U''(a) |U''(s)|)) thin e^(Delta U \/ T). $

It needs $Delta U >> T$, overdamped motion and a non-degenerate saddle #cond.

*Mixing time.* For two wells, the slowest relaxation is the hopping between them.
The #gloss[spectral gap][the smallest non-zero decay rate of the dynamics; its inverse is the slowest relaxation time] is then about $2 \/ tau$ #cond.

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  [Measure], [What it sees], [Measurable online?], [Blind to],
  [Linear response $chi$], [Local motion under a small new gradient], [Yes: Hessian-vector products], [Barriers, non-local optima],
  [MSD], [Local exploration by noise], [Yes: track parameter drift], [Direction of the new task],
  [Escape time $tau$], [Crossing one known barrier], [Only in toy models], [Everything outside that barrier],
  [Spectral gap], [Slowest relaxation, global], [No, in practice], [Speed of the useful directions],
)

== The bridge to Bayesian plasticity

The key question of P3 is whether these measures relate to the Bayesian $cal(P)$ of @sec-plasticity.

*Locally, yes.* At $T = 1$ for a Gaussian posterior, $chi(oo) = H^(-1) = Sigma$, the posterior covariance.
A small new likelihood term with gradient $g$ moves the Bayesian posterior mean by $Sigma g$: one Newton step.
So static linear response equals Bayesian sensitivity #rig (Gaussian), #cond (locally otherwise).
Information gain to first order is $1/2 tr(Sigma F_"new") = 1/2 tr(chi(oo) F_"new")$.

*Globally, no.* Exact Bayes has no time.
The posterior after new data is defined instantly, whatever the barriers between its modes.
Escape time and mixing time measure the cost of an #emph[algorithm] that computes the posterior.
They do not measure how far the posterior moves.
So escape and mixing are #emph[computational] plasticity; $cal(P)$ is #emph[statistical] plasticity.
They answer different questions, and neither bounds the other in general #rig.
A learner can be statistically plastic and computationally stuck (a far-away new mode), or the reverse (fast mixing over a posterior that barely moves).

#callout(title: [Answer to P3])[
  Locally and at equilibrium, response, MSD and posterior covariance coincide (fluctuation–dissipation plus Laplace).
  Barrier crossing and mixing are a separate, computational kind of plasticity.
  Loss of plasticity in practice is mostly of this second kind (@sec-mechanisms).
]

== SGD noise is not Langevin noise

Five differences, each a reason not to read SGD as a posterior sampler:

+ *Shape.* SGD noise covariance is the per-example gradient covariance over $B$. Near an optimum this is close to $F \/ B$ #cond: anisotropic and aligned with curvature. Langevin noise is isotropic.
+ *It vanishes where networks end up.* If every per-example loss is minimised at $theta^*$, every per-example gradient is zero there, and so is $Sigma(theta^*)$ #rig. Over-parameterised networks reach such #gloss[interpolating minima][points where every training example is fit exactly]. SGD then freezes, while Langevin keeps moving.
+ *Wrong stationary law.* For a quadratic loss with constant $Sigma$, SGD's stationary covariance $C$ solves $H C + C H = eta Sigma \/ B$. This equals $T H^(-1)$ only if $Sigma prop H$ #rig. So SGD's stationary law is generally not a posterior at any temperature.
+ *No detailed balance.* State-dependent, anisotropic noise generally drives probability currents around loops. Then fluctuation–dissipation fails #cond.
+ *Scale.* SGD's per-step noise is $O(eta)$; Langevin's is $O(sqrt(eta))$. For small steps, injected noise dominates by far.
