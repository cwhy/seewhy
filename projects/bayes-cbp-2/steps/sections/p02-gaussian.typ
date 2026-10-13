#import "/template.typ": *

= Gaussian posteriors, precision and continual learning <p2>

#callout(title: [The question])[
  In the simplest Gaussian model: how do prior variance, posterior precision and data noise set plasticity?
  Derive the update, the mean shift, the effect of prior precision, and the multivariate case.
  How do Hessian, Fisher and posterior precision relate? Does this explain the stability–plasticity trade-off?
  Are variational continual learning and EWC two ways to keep the old posterior? Which relations are exact?
]

== Core view

In a Gaussian model, plasticity is set per direction by one ratio: prior variance along that direction over the new data's noise along it.
The stability–plasticity trade-off is real only in directions both tasks constrain.
In a static model the gain decays to zero; a model in which the parameter drifts has a non-zero floor.
EWC and variational continual learning are two Gaussian compressions of the old posterior.

== Derivation

=== One dimension (P2.1–P2.3)
Prior $theta tilde cal(N)(mu_0, sigma_0^2)$; observation $y | theta tilde cal(N)(theta, sigma_n^2)$. Then #rig

$ sigma_1^(-2) = sigma_0^(-2) + sigma_n^(-2), quad mu_1 - mu_0 = k (y - mu_0), quad k = frac(sigma_0^2, sigma_0^2 + sigma_n^2). $

The #gloss[gain][$k$: the fraction of the surprise $y - mu_0$ that moves the mean] is the response to new data.
The information gained, Stage P1's $cal(P)$, is $1/2 log(1 + sigma_0^2 \/ sigma_n^2)$ #rig.
Both depend only on the ratio of prior variance to noise.
High prior precision means low gain: the model barely responds.

=== Many dimensions (P2.4)
$y = J theta + epsilon$, $epsilon tilde cal(N)(0, R)$, prior $cal(N)(mu, Sigma)$ with precision $Lambda = Sigma^(-1)$. Then #rig

$ Lambda' = Lambda + F, quad F = J^top R^(-1) J, quad mu' = mu + Sigma J^top (J Sigma J^top + R)^(-1) (y - J mu), $
$ cal(P) = 1/2 log det (I + Sigma F). $

With $kappa_i$ the eigenvalues of $Sigma^(1\/2) F Sigma^(1\/2)$, direction $i$ responds with gain $kappa_i \/ (1 + kappa_i)$ and contributes $1/2 log(1 + kappa_i)$.
This makes Stage P1's point exact: variance counts only where $F$ looks.

=== Hessian, Fisher and posterior precision (P2.5)
The #gloss[Laplace approximation][a Gaussian at the posterior mode, with precision equal to the curvature of the negative log-posterior there] uses $H = -nabla^2 log p(theta | D)$ at the mode.

#table(
  columns: (1.3fr, 1.7fr, auto),
  [Matrix], [Relation to the data Hessian], [Status],
  [Fisher $F = bb(E)_(y tilde "model") [nabla log p thin nabla log p^top]$], [Equals $bb(E)_(y tilde "model") [-nabla^2 log p]$; equals the data Hessian only at the true $theta$ of a correct model, as $n -> oo$], [#rig / #cond],
  [Gauss–Newton $J^top H_"out" J$], [Drops second derivatives of the network; equals $F$ for exponential-family outputs with canonical link @martens2020], [#rig],
  [Empirical Fisher (squared observed gradients)], [Not $F$ in general; can point wrong away from an optimum @kunstner2019], [#wrong as a stand-in],
)

=== The trade-off, exactly (P2.6)
Task $A$ leaves precision $Lambda_A$; task $B$ brings Fisher $F_B$.
If they commute, with eigenvalues $a_i$ and $f_i$, direction $i$ moves by the fraction $f_i \/ (a_i + f_i)$ #rig. Three regimes:

- $a_i >> f_i$: stable, not plastic.
- $a_i << f_i$: plastic, no forgetting.
- $a_i approx f_i$: the true trade-off.

=== A floor needs drift
In the static model precision accumulates and the gain falls like $1 \/ t$.
Let the parameter drift: $theta_(t+1) = theta_t + w_t$, $w_t tilde cal(N)(0, q)$, the #gloss[Kalman filter][the exact Bayesian filter for a drifting linear-Gaussian state] model.
The variance before an update solves $(P^-)^2 - q P^- - q r = 0$ with $r = sigma_n^2$, so #rig

$ P^-_oo = frac(q + sqrt(q^2 + 4 q r), 2), quad k_oo = frac(P^-_oo, P^-_oo + r) approx sqrt(q \/ r). $

#fig(include "/figures/kalman_gain.typ", caption: [
  Gain $k_t$ for a scalar Gaussian model, $sigma_0^2 = r = 1$. Static: $1 \/ (t + 1)$. With drift $q$, the gain levels off at $k_oo$ (dashed). Exact recursion.
]) <fig-kalman>

=== EWC and variational continual learning
*Variational continual learning* @nguyen2018vcl fits a Gaussian $q_t$ minimising $D_"KL" (q_t || q_(t-1) p(D_t | theta) \/ Z)$.
*EWC* @kirkpatrick2017ewc keeps a mode and diagonal Fisher: a Laplace approximation.
The exact recursive Laplace form is one penalty with precision $Lambda_t = Lambda_(t-1) + F_t$ @huszar2018; one penalty per task double-counts #rig.
So yes: both are approximate ways to keep the old posterior. The first fits a mean and variance by an evidence bound; the second reads them off the mode and the curvature.

== When it holds

- Every formula above is exact for linear-Gaussian models with full covariance #rig.
- For networks they hold through Laplace or variational approximations: a single mode, local curvature #cond.
- With diagonal covariance, correlations are lost and errors compound over tasks #cond.

== Limits

- A network posterior has many symmetric modes; one Gaussian covers one basin at best.
- A tempered KL weight $beta != 1$ in a variational objective is not Bayesian updating; it is a knob with no first-principles value #cond.
- The drift model is one of many possible change models; the Gaussian case cannot say which is right.

== Alternatives

- *Full-covariance or Kronecker-factored* posteriors instead of diagonal: better retention, higher cost.
- *Mixture posteriors*: cover several modes, at a cost in tractability.
- *Change models other than drift*: abrupt replacement instead of gradual drift. Stage P5 introduces one.

== Next step

Stage P2 is about the posterior itself. A real network does not hold a posterior; it moves parameters by gradient steps.
Stage P3 asks how this motion relates to Bayesian plasticity.
