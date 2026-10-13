#import "/template.typ": *

= From Bayesian inference to optimisation dynamics <p3>

#callout(title: [The question])[
  Relate SGD to a stochastic differential equation; relate SGLD to posterior sampling; derive Langevin dynamics and its stationary law.
  How do curvature, noise and temperature shape parameter motion?
  Can mean-squared displacement, response, barrier escape time and mixing time measure plasticity?
  How does SGD noise differ from injected Langevin noise?
  Above all: can response speed, exploration or barrier crossing be tied quantitatively to Bayesian plasticity?
]

== Core view

Locally and at equilibrium, three things coincide: the dynamics' response to a small new gradient, its mean-squared displacement, and the Bayesian posterior covariance.
Barrier crossing and mixing are different: they measure the cost of #emph[computing] a posterior, not how far the posterior moves.
So there are two kinds of plasticity, statistical (Stage P1) and computational. Neither bounds the other.
SGD noise is not Langevin noise; in particular it vanishes at the minima networks reach.

== Derivation

=== SGD as an SDE (P3.1)
SGD steps $theta_(k+1) = theta_k - eta hat(g)_k$, with $hat(g)_k$ of mean $nabla L$ and covariance $Sigma(theta) \/ B$.
In time $t = k eta$, to first order in $eta$ @li2017sde:

$ upright(d) theta = -nabla L thin upright(d) t + sqrt(eta Sigma(theta) \/ B) thin upright(d) W_t . $

This is a weak approximation: it matches expectations of smooth functions, not paths #cond.

=== Langevin and SGLD (P3.2, P3.3)
#gloss[Overdamped Langevin dynamics][gradient descent plus Gaussian noise of fixed size] at temperature $T$:
$upright(d) theta = -nabla U thin upright(d) t + sqrt(2 T) thin upright(d) W_t$, with $U = -log p(D | theta) - log p(theta)$.
Its density obeys $partial_t p = nabla dot (p nabla U) + T Delta p$, whose stationary solution is $p_oo prop e^(-U \/ T)$ when normalisable #rig.
At $T = 1$ this is the posterior.
SGLD @welling2011sgld is its Euler step with a minibatch gradient. It samples the posterior only as the step size goes to zero #cond.

=== Curvature, noise, temperature (P3.4)
In one direction with curvature $kappa$ (the #gloss[Ornstein–Uhlenbeck process][a random walk pulled toward a centre]) #rig:
the mean relaxes as $e^(-kappa t)$; the stationary variance is $T \/ kappa$; with $kappa = 0$ the variance grows as $2 T t$.

=== Candidate measures (P3.5)
- *Mean-squared displacement* at equilibrium: $"MSD"(t) = 2 (T \/ kappa)(1 - e^(-kappa t))$ #rig.
- *Response* to a small constant force $epsilon g$: the mean shifts by $chi(t) epsilon g$, $chi(t) = (1 - e^(-kappa t)) \/ kappa$ #rig.
- *Fluctuation–dissipation:* for reversible dynamics at equilibrium, $chi(t) = "MSD"(t) \/ (2 T)$ #rig.
- *Barrier escape* (Kramers @kramers1940): $tau approx 2 pi \/ sqrt(U''(a) |U''(s)|) thin e^(Delta U \/ T)$ for $Delta U >> T$ #cond.
- *Mixing time:* for two wells, the #gloss[spectral gap][slowest non-zero decay rate] is about $2 \/ tau$ #cond.

=== The bridge to Bayesian plasticity
At $T = 1$ with a Gaussian posterior, $chi(oo) = H^(-1) = Sigma$.
A small new likelihood term with gradient $g$ moves the posterior mean by $Sigma g$.
So static response = Bayesian sensitivity #rig (Gaussian), #cond (locally).
Stage P2's information gain is, to first order, $1/2 tr(Sigma F_"new") = 1/2 tr(chi(oo) F_"new")$.

Escape and mixing have no such counterpart. Exact Bayes has no time: the new posterior exists at once, whatever barriers separate its modes.
So escape and mixing measure the cost of the algorithm. We call this #emph[computational] plasticity #rig (as a statement of definitions).

=== SGD noise is not Langevin noise (P3.6)
+ *Shape:* near an optimum SGD noise covariance is close to $F \/ B$ #cond — anisotropic, aligned with curvature.
+ *It vanishes at interpolation:* if every per-example loss is minimised at $theta^*$, all per-example gradients vanish there, and so does $Sigma(theta^*)$ #rig. SGD then freezes; Langevin does not.
+ *Wrong stationary law:* for a quadratic with constant $Sigma$, SGD's stationary covariance $C$ solves $H C + C H = eta Sigma \/ B$, which equals $T H^(-1)$ only if $Sigma prop H$ #rig.
+ *No detailed balance* in general, so fluctuation–dissipation fails #cond.
+ *Scale:* SGD's per-step noise is $O(eta)$, Langevin's $O(sqrt(eta))$.

== When it holds

- The bridge needs equilibrium, reversibility, $T = 1$ and a locally Gaussian posterior.
- Kramers needs a high barrier, overdamped motion and a non-degenerate saddle.
- The SDE view of SGD needs small steps and smooth losses.

== Limits

#table(
  columns: (auto, 1fr, 1fr),
  [Measure], [Sees], [Blind to],
  [Response $chi$], [Local motion under a new gradient], [Barriers; distant optima],
  [MSD], [Local exploration], [Direction of the new task],
  [Escape time], [One known barrier], [Everything else],
  [Spectral gap], [Slowest global relaxation], [Speed in useful directions; not measurable in practice],
)

== Alternatives

- *Underdamped (kinetic) Langevin* or Hamiltonian dynamics: faster exploration, same stationary law.
- *Preconditioned Langevin*, with noise shaped like the Fisher: closer to SGD's noise, and still targets the posterior if the correction term is included.
- *Information-geometric speed* (natural-gradient flow): measures response in a reparametrisation-invariant metric.

== Next step

Stage P3 separates statistical from computational plasticity.
Stage P4 asks which kind networks actually lose, and why plain gradient updates may fail to recover it.
