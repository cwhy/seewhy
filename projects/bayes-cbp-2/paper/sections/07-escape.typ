#import "/template.typ": *

= Escape time, spectral gap and recovery speed <sec-escape>

_Answers P7._

== Diffusion alone

In one dimension, at unit temperature, the #gloss[mean first-passage time][the expected time until the process first reaches a target set] $T(x)$ from $x$ to a target at $b$ solves $T'' - U' T' = -1$ with $T(b) = 0$.
With a reflecting wall at $ell$ the solution is #rig

$ T(x) = integral_x^b e^(U(y)) integral_ell^y e^(-U(z)) thin upright(d) z thin upright(d) y . $

For a high barrier, Laplace's method on this integral gives Kramers' law (@sec-dynamics) #cond.
The exponential dependence on $Delta U \/ T$ is the reason local noise is a poor plasticity mechanism for high barriers.

== Diffusion with resetting

Add resets at rate $r$ to a draw from $p_0$.
Let $tau$ be the first-passage time #emph[without] resets, started from $p_0$, and
$hat(q)(r) = bb(E)[e^(-r tau)]$ its Laplace transform: the probability of reaching the target before the next reset.
Each attempt is a fresh start, so the mean time from a fresh reset is #rig @evans2011diffusion

$ T_r = frac(1 - hat(q)(r), r thin hat(q)(r)) . $

Starting instead from the stuck point $x_0$, with $hat(q)_0$ the same transform from $x_0$:

$ T = frac(1 - hat(q)_0 (r), r) + (1 - hat(q)_0 (r)) thin T_r . $

== A reset is not an arrival (P7.4)

Take the pure-jump limit: no motion between resets, and a reset lands directly in the good region $A$ with probability $h = p_0(A)$.
Then $hat(q) = h$ and #rig

$ T_r = frac(1 - h, r h) approx frac(1, r h) . $

The waiting time for #emph[a] reset is $1 \/ r$. The waiting time for a #emph[useful] reset is $1 \/ (r h)$.
And arrival in $A$ is not adaptation either. After landing, training must still converge, which takes its own time (@sec-switch).
The total is (time to a useful reset) + (training time from the landing point). The second term is not zero.

== When does resetting help?

Expand $T_r$ in small $r$, resetting to the start point, with $m_1$ and $sigma_tau$ the mean and standard deviation of the reset-free $tau$. Then #rig @reuveni2016optimal

$ frac(upright(d) T_r, upright(d) r) |_(r = 0) = frac(m_1^2 - sigma_tau^2, 2) . $

A small reset rate helps if and only if the #gloss[coefficient of variation][standard deviation over mean] of $tau$ exceeds one.
Escape from the bottom of a well is nearly #emph[exponential], so its coefficient of variation is about one. Resetting to the stuck point then does nothing to first order and hurts at larger $r$ #cond.

What helps is the reset #emph[target].
@fig-mfpt compares four cases. In the double well, starting at the left minimum $theta = -1$:

- resetting to the start raises the mean time from 17.3 (no reset) to 22.3 at $r = 1$;
- resetting to a fresh draw from $cal(N)(0, 1)$, which often lands near the barrier top, cuts it to 1.96 at $r = 1$ and 0.31 at $r = 10$.

With a constant downhill force and no barrier, resetting to the start only hurts: 1.0 rises to 10.2 at $r = 10$.
Resetting to the fresh draw still helps there (0.34 at $r = 10$), because the draw lands closer to the target on average.

#fig(include "/figures/mfpt_reset.typ", caption: [
  Mean time to reach $theta >= 1$ from $theta = -1$, against reset rate $r$ (both axes logarithmic).
  The left end, $r = 10^(-3)$, is close to no resetting.
  Double well $U = 4(theta^2 - 1)^2$, or a constant force $+2$; unit temperature; reflecting wall at $-3$.
  Lines: finite-difference solution of $T'' + F T' - r T + r R[T] = -1$ on 801 grid points, where $R[T]$ is $T$ at the reset point or its average over $cal(N)(0, 1)$.
  Points: Monte Carlo for the double well with fresh resets (4 000 particles, Euler step $10^(-3)$); within 2.4% of the lines, with Euler error largest at small $r$.
]) <fig-mfpt>

== When a reset is worse than diffusion

Three cases, each a falsifiable prediction for networks:

+ *The target is far from anything useful.* Resetting to the start is the extreme case (@fig-mfpt). For networks: re-drawing a unit from a distribution whose typical features are useless for the new task.
+ *The hit probability is tiny.* For an isotropic Gaussian $p_0$ in $d$ dimensions and a useful region of fixed angular width, $h$ falls exponentially in $d$ #cond. Then $1 \/ (r h)$ is astronomical. A network escapes this only because a fresh unit need not be "right"; it only needs a usable gradient, which is a much larger set.
+ *Steady progress is interrupted.* When the reset-free time has coefficient of variation below one, as in downhill descent, any reset to the start adds time.

== Do jumps raise the spectral gap? (P7.6)

#proposition("Reversible jumps cannot slow mixing", rig)[
  Let the diffusion be reversible with respect to $pi$. Add jumps that satisfy detailed balance with respect to $pi$.
  Then the spectral gap of the combined process is at least that of the diffusion alone.
  If the jumps re-draw from $pi$ itself at rate $lambda$, the gap is at least $lambda$.
]
_Proof._ The #gloss[Dirichlet forms][the quadratic forms $⟨ f, -cal(L) f ⟩_pi$ whose minimum over unit-variance $f$ is the gap] of the two parts add, and the variance under $pi$ is unchanged. Hence the minimum can only grow. Re-drawing from $pi$ has Dirichlet form $lambda op("Var")_pi (f)$, so its gap is $lambda$. $square$

Outside this case there is no general result #open.
A jump that does not preserve $pi$ changes the stationary law, so "faster mixing" then means faster convergence to a different target.
@fig-dw-occupancy shows exactly this: the naive reset converges fast, to the wrong value.
And for the Metropolis-corrected reset, the gain in gap is limited by the acceptance rate. In high dimension, with $pi$ concentrated far from $p_0$, the acceptance rate and the gain both go to zero #cond.

== From these results to network predictions (P7.7)

These hypotheses #heur are testable in @sec-plan:

+ *The landing distribution decides.* A reset helps only if a fresh unit has a usable gradient on the new task with non-negligible probability. Measure the fraction of reset units that become useful within $k$ steps.
+ *Resetting to near the current state does not help dead units.* A small perturbation of a dead unit stays inside the dead set with high probability.
+ *The coefficient of variation predicts the benefit.* Where adaptation times across seeds or units have coefficient of variation below one, more resets should hurt.
