#import "/template.typ": *

= Escape time, spectral gap and recovery speed <p7>

#callout(title: [The question])[
  Compare local diffusion with non-local resets as exploration.
  State Kramers' law and its conditions; the mean first-passage time over a high barrier; the same with resets or jumps.
  Is a reset event the same as reaching a good region? How do rate, target, hit probability and state dependence combine?
  Do jumps always raise the spectral gap or shorten mixing? What testable predictions follow for networks?
  Give a solvable example, and cases where resets are worse than diffusion.
]

== Core view

A reset helps through where it lands, not through the event itself.
Resetting to the stuck point never helps a memoryless escape; resetting near the barrier top cuts escape time from 17 to 0.3 in our example.
Posterior-preserving reversible jumps can only raise the spectral gap; other jumps change the target, so "faster" means faster to something else.

== Derivation

=== Diffusion alone (P7.1, P7.2)
At unit temperature, the #gloss[mean first-passage time][expected time to first reach a target] $T(x)$ to a target at $b$, with a reflecting wall at $ell$, is #rig

$ T(x) = integral_x^b e^(U(y)) integral_ell^y e^(-U(z)) thin upright(d) z thin upright(d) y . $

Laplace's method on it gives Kramers' law (Stage P3) for $Delta U >> T$ #cond.

=== With resetting (P7.3)
Resets at rate $r$ to a draw from $p_0$. Let $tau$ be the reset-free first-passage time from $p_0$ and $hat(q)(r) = bb(E)[e^(-r tau)]$, the chance of arriving before the next reset.
By renewal @evans2011diffusion #rig:

$ T_r = frac(1 - hat(q)(r), r thin hat(q)(r)), quad T_"from stuck point" = frac(1 - hat(q)_0 (r), r) + (1 - hat(q)_0 (r)) T_r . $

=== A reset is not an arrival (P7.4)
In the pure-jump limit a reset lands in the good region $A$ with probability $h = p_0(A)$, so $hat(q) = h$ and $T_r = (1 - h) \/ (r h)$ #rig.
The wait for #emph[a] reset is $1 \/ r$; for a #emph[useful] one, $1 \/ (r h)$. After landing, training must still converge.

=== What decides the benefit (P7.5)
Resetting to the start, for small $r$ @reuveni2016optimal #rig:
$ frac(upright(d) T_r, upright(d) r) |_(r = 0) = frac(m_1^2 - sigma_tau^2, 2) . $
It helps iff the #gloss[coefficient of variation][standard deviation over mean] of $tau$ exceeds one.
Escape from a well bottom is nearly exponential (coefficient one), so resetting to the stuck point is neutral at first and harmful later #cond.
The target is what matters (@fig-mfpt). From the left minimum of $U = 4(theta^2 - 1)^2$:
reset to the start raises the time from 17.3 to 22.3 at $r = 1$; reset to $cal(N)(0, 1)$ cuts it to 1.96 at $r = 1$ and 0.31 at $r = 10$.

#fig(include "/figures/mfpt_reset.typ", caption: [
  Mean time to reach $theta >= 1$ from $theta = -1$ against reset rate (log–log; $r = 10^(-3)$ is near no reset). Double well or constant force $+2$; reflecting wall at $-3$.
  Lines: finite-difference solution of $T'' + F T' - r T + r R[T] = -1$ (801 points). The Monte Carlo check (4 000 particles) agrees within 2.4%.
]) <fig-mfpt>

=== When resets are worse than diffusion
+ The target is far from anything useful: with a constant downhill force, resetting to the start raises 1.0 to 10.2 at $r = 10$.
+ The hit probability is tiny: for isotropic $p_0$ in $d$ dimensions and a region of fixed angular width, $h$ falls exponentially in $d$ #cond.
+ Steady progress is interrupted: whenever the coefficient of variation is below one.

=== Spectral gap (P7.6)
#proposition("Reversible jumps cannot slow mixing", rig)[
  If the diffusion is reversible with respect to $pi$ and the jumps satisfy detailed balance with respect to $pi$, the combined spectral gap is at least the diffusion's. Re-drawing from $pi$ itself at rate $lambda$ gives a gap of at least $lambda$.
]
_Proof._ The #gloss[Dirichlet forms][$⟨ f, -cal(L) f ⟩_pi$, minimised over unit-variance $f$ to give the gap] add, and the variance is unchanged. $square$

For jumps that do not keep $pi$ (Stage P6), there is no general result #open; @fig-dw-occupancy shows fast convergence to a wrong value.

== When it holds

The renewal formulas need Poisson resets independent of the path. Kramers needs a high barrier. The gap result needs reversibility and posterior preservation.

== Limits

- One-dimensional examples; networks have saddles more than barriers @dauphin2014saddle.
- First passage measures arrival, not adaptation.

== Alternatives

- *Optimal resetting rates* tuned to the first-passage distribution, rather than fixed rates.
- *State-dependent resetting* (reset more where progress stalls), which these formulas do not cover.

== Next step: predictions for networks (P7.7)

Hypotheses #heur, testable later:
the landing distribution decides (a reset helps only if a fresh unit has a usable gradient with non-negligible probability);
small perturbations do not revive dead units;
where adaptation times have coefficient of variation below one, more resets hurt.
Stage P8 moves from one barrier to a full task switch.
