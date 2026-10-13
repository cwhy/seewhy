#import "/template.typ": *

= Gaussian posteriors, precision and continual learning <sec-gaussian>

_Answers P2._ Everything in this section is exact for linear-Gaussian models.
For neural networks it holds only through a Laplace or variational Gaussian approximation; the last subsection says where.

== One dimension

Prior $theta tilde cal(N)(mu_0, sigma_0^2)$; one observation $y | theta tilde cal(N)(theta, sigma_n^2)$.
Precisions add #rig:

$ sigma_1^(-2) = sigma_0^(-2) + sigma_n^(-2), quad mu_1 = mu_0 + k (y - mu_0), quad k = frac(sigma_0^2, sigma_0^2 + sigma_n^2). $

The #gloss[gain][$k$: the fraction of the surprise $y - mu_0$ that moves the mean] is the model's response to new data.
It is large when the prior is vague ($sigma_0^2$ large) or the data are clean ($sigma_n^2$ small).
The information gain of this observation is $1/2 log (1 + sigma_0^2 \/ sigma_n^2)$ #rig.
Both quantities depend on the #emph[ratio] of prior variance to noise, not on either alone.

== Many dimensions

Let $y = J theta + epsilon$ with $epsilon tilde cal(N)(0, R)$ and prior $cal(N)(mu, Sigma)$, precision $Lambda = Sigma^(-1)$. Then #rig

$ Lambda' = Lambda + F, quad F = J^top R^(-1) J, $
$ mu' = mu + K (y - J mu), quad K = Sigma J^top (J Sigma J^top + R)^(-1), $
$ cal(P) = I(theta ; y) = 1/2 log det (I + Sigma F). $

Diagonalise $Sigma^(1\/2) F Sigma^(1\/2)$ with eigenvalues $kappa_i$.
Each eigen-direction responds with gain $kappa_i \/ (1 + kappa_i)$ and contributes $1/2 log(1 + kappa_i)$ to $cal(P)$.
$kappa_i$ is a signal-to-noise ratio: prior variance along direction $i$, measured in units of the new data's noise along it.
This is the precise sense in which uncertainty and plasticity relate (@sec-plasticity).

== Hessian, Fisher and posterior precision

For a non-Gaussian likelihood, the #gloss[Laplace approximation][replace the posterior by a Gaussian at its mode, with precision equal to the curvature of the negative log-posterior there] gives
$p(theta | D) approx cal(N)(theta^*, H^(-1))$ with $H = -nabla^2 log p(theta | D) |_(theta^*)$.
Three matrices are often used for $H$. Their relations:

#table(
  columns: (1.3fr, 1.7fr, auto),
  [Matrix], [Relation to the Hessian of the negative log-likelihood], [Status],
  [Fisher $F = bb(E)_(y tilde p(y | x, theta)) [nabla log p thin nabla log p^top]$],
  [Equals $bb(E)_(y tilde "model") [-nabla^2 log p]$. Equals the data Hessian only at the true $theta$ with a correct model, as $n -> oo$.],
  [#rig / #cond],
  [Generalised Gauss–Newton $J^top H_"out" J$],
  [Drops the term with second derivatives of the network. For exponential-family outputs with the canonical link it equals $F$ exactly @martens2020.],
  [#rig],
  [Empirical Fisher (squared observed gradients)],
  [Uses data labels instead of model samples. Not $F$ in general, and can point in wrong directions away from an optimum @kunstner2019.],
  [#wrong as a stand-in for $F$],
)

So "Fisher = Hessian = posterior precision" is a chain of approximations, not an identity.
The first link needs a well-specified model near its optimum; the second needs a single dominant mode.

== Stability–plasticity, exactly

Let task $A$ leave precision $Lambda_A$ and task $B$ bring Fisher $F_B$.
The gain that maps a new-task surprise into a parameter change is

$ K_B = (Lambda_A + F_B)^(-1) F_B. $

When $Lambda_A$ and $F_B$ commute, with eigenvalues $a_i$ and $f_i$, direction $i$ moves by the fraction $f_i \/ (a_i + f_i)$ #rig.
Three regimes follow:

- $a_i >> f_i$: the old task pins this direction. Stable, not plastic.
- $a_i << f_i$: the old task did not care. Plastic, and no forgetting.
- $a_i approx f_i$: the true trade-off. Both tasks want this direction.

The trade-off exists only in the third regime.
A network with spare directions ($a_i approx 0$) can be both stable and plastic.

== A plasticity floor needs a change model

In the static model, precision only accumulates, and the gain decays like $1 \/ t$ (@fig-kalman).
Now let the parameter drift: $theta_(t+1) = theta_t + w_t$ with $w_t tilde cal(N)(0, q)$.
This is a #gloss[Kalman filter][the exact Bayesian filter for linear-Gaussian models with drifting state].
The prediction step adds $q$ to the variance before each update.
In one dimension the variance before an update, $P^-$, converges to the root of $(P^-)^2 - q P^- - q r = 0$, where $r = sigma_n^2$. So #rig

$ P^-_oo = frac(q + sqrt(q^2 + 4 q r), 2), quad k_oo = frac(P^-_oo, P^-_oo + r) approx sqrt(q \/ r) quad (q << r). $

The gain no longer goes to zero.
It settles at a floor, and $1 \/ k_oo approx sqrt(r \/ q)$ is the model's #emph[memory length] in observations.

#fig(include "/figures/kalman_gain.typ", caption: [
  Gain $k_t$ (weight on the newest observation) for a scalar Gaussian model, $sigma_0^2 = r = 1$.
  The static model's gain falls as $1 \/ (t + 1)$ without bound.
  With process noise $q$ it levels off at the floor $k_oo$ (dashed lines, closed form).
  Exact recursion, not simulation.
]) <fig-kalman>

#callout(title: [Answer to P2.6])[
  The Gaussian model explains the stability–plasticity trade-off completely, and also shows its limits.
  The trade-off is per direction, set by the ratio of old precision to new information.
  Long-run plasticity is set by the change model ($q$), not by the data seen.
]

== EWC and variational continual learning as compressed posteriors

Exact recursive Bayes needs the whole old posterior. Two methods compress it to a Gaussian.

*Variational continual learning* (VCL) @nguyen2018vcl picks $q_t$ in a Gaussian family to minimise $D_"KL" (q_t || q_(t-1) p(D_t | theta) \/ Z)$.
It keeps a mean and a variance, both fitted by the evidence lower bound.

*Elastic weight consolidation* (EWC) @kirkpatrick2017ewc keeps a mode and a diagonal Fisher, and adds a quadratic penalty.
It is a Laplace approximation.
The original adds one penalty per old task, each centred at that task's mode.
Huszár @huszar2018 showed the recursive Laplace posterior is instead one penalty: precision $Lambda_t = Lambda_(t-1) + F_t$, centred at the latest mode.
The per-task form double-counts old tasks #rig.

Both are approximate ways to keep the old posterior. Where each is exact or not:

- For a linear-Gaussian model with full covariance, both are exact #rig.
- With a diagonal covariance, both discard correlations between parameters. Errors then compound over tasks #cond.
- Both assume one mode. A network posterior has many symmetric modes; a single Gaussian describes one basin at best #cond.
- A weighted KL term $beta D_"KL" (q_t || q_(t-1))$ with $beta != 1$ is not Bayesian updating. It is a tempered posterior. It is a legitimate knob, but it has no first-principles value #cond.
