#import "/template.typ": *

= Plasticity from Bayesian first principles <sec-plasticity>

_Answers P1._

== Setting

Parameters $theta$ have a posterior $p(theta | D_(1:t))$ after $t$ batches.
The old posterior is the prior for the next batch:

$ p(theta | D_(1:t+1)) = frac(p(D_(t+1) | theta) thin p(theta | D_(1:t)), p(D_(t+1) | D_(1:t))). $

This recursion is exact when batches are #gloss[conditionally independent given $theta$][once $theta$ is known, one batch says nothing about another].
Call this the #emph[static model]: one $theta$ generates all data, forever.

== Definition: plasticity as expected information gain

How much will the next batch move the posterior?
The move from one posterior to the next is measured by #gloss[KL divergence][$D_"KL" (p || q) = bb(E)_p [log p\/q]$, the expected extra code length for using $q$ when $p$ is true; zero only when $p = q$].
Before seeing the batch, average over what it could be:

$ cal(P)_t = bb(E)_(D_(t+1) | D_(1:t)) [ D_"KL" (p(theta | D_(1:t+1)) || p(theta | D_(1:t))) ] = I(theta ; D_(t+1) | D_(1:t)). $

The second equality is a standard identity #rig: expected posterior-to-prior KL is the #gloss[mutual information][how many nats, on average, one variable tells about another] between parameters and data.
We take $cal(P)_t$ as the definition of #emph[Bayesian plasticity].
It is the information the learner expects to absorb from the next batch.

Two readings must be kept apart.
$cal(P)_t$ is an #emph[expected] quantity, fixed before the batch arrives.
The #emph[realised] KL for one surprising batch can be far larger.
A model with low $cal(P)_t$ can still be moved a lot by data it considers unlikely.

== Plasticity can only fall in a static world

#proposition("Monotone plasticity", rig)[
  Let observations $X_1, X_2, dots$ be conditionally i.i.d. given $theta$, with finite entropies.
  Then $cal(P)_n = I(theta ; X_(n+1) | X_(1:n))$ is non-increasing in $n$.
]

_Proof._ Write $I(theta ; X_(n+1) | X_(1:n)) = H(X_(n+1) | X_(1:n)) - H(X_(n+1) | theta, X_(1:n))$.
The second term equals $H(X | theta)$ for every $n$, by conditional independence.
For the first, conditioning cannot raise entropy, and the sequence is #gloss[exchangeable][any reordering has the same joint law]:

$ H(X_(n+2) | X_(1:n+1)) <= H(X_(n+2) | X_(2:n+1)) = H(X_(n+1) | X_(1:n)). $

So the first term, and with it $cal(P)_n$, cannot increase. $square$

The total is bounded too.
Summing gives $sum_(n < N) cal(P)_n = I(theta ; X_(1:N))$.
For a regular $d$-parameter model this grows like $d/2 log N$ @clarke1990, so per-step plasticity decays like $d \/ (2 n)$ #cond.

#callout(title: [Consequence])[
  For an exact Bayesian learner, losing plasticity is not a malfunction.
  It is the right response to a world it assumes never changes.
  The fix is not to "restore" plasticity inside the static model.
  It is to change the model so the world #emph[can] change.
  Every route in @sec-routes follows from this.
]

Neural networks lose plasticity for other reasons too (@sec-mechanisms).
Those are not Bayesian effects; they come from the optimiser and the architecture.
The static-model effect is only one of several, but it is the one that a Bayesian redesign can address directly.

== Parameter plasticity versus predictive plasticity

Neural networks have many parameter settings that compute the same function.
Swapping two hidden units is one example; rescaling across a ReLU is another.
So the posterior can move in $theta$ without changing any prediction.
$cal(P)_t$ counts such moves.

A function-space version avoids this.
Let $f$ be the vector of predictions on a set of query inputs. Define

$ cal(P)_t^"pred" = I(f ; D_(t+1) | D_(1:t)) <= cal(P)_t. $

The inequality is the #gloss[data-processing inequality][a function of $theta$ cannot carry more information than $theta$ itself] #rig.
We treat $cal(P)^"pred"$ as the more fundamental quantity (P1.6).
It is invariant to reparametrisation and it measures what users care about.
Its cost: it depends on a choice of query inputs, and it is harder to estimate.

== Uncertainty is not plasticity

Posterior variance is necessary for plasticity but not sufficient.
@sec-gaussian shows that in the Gaussian case #rig

$ cal(P) = 1/2 log det (I + Sigma F_"new"), $

where $Sigma$ is the posterior covariance and $F_"new"$ is the #gloss[Fisher information][how sharply the new data's log-likelihood curves around the parameters; large means the data pin the parameters down] of the next batch.
Large variance in a direction the new data cannot see ($F_"new" v = 0$) adds nothing.
Small variance in the one direction the new data probes can make $cal(P)$ near zero even when total variance is large.
So "plasticity = posterior variance" is #wrong as a general statement.
It is right only along the directions that $F_"new"$ weights.

== Why a model fed much old data learns slowly

Take the static Gaussian model with equal curvature per datum, $n_A$ points from task $A$ and $n_B$ from task $B$.
The posterior mean moves from the $A$ optimum toward the $B$ optimum by the fraction

$ frac(n_B, n_A + n_B). $

With $n_A = 10^6$ and $n_B = 10^3$, the mean moves 0.1% of the way #cond.
This is correct inference if $A$ and $B$ come from the same $theta$.
If the task changed, the model is #emph[misspecified]: it reads evidence about a new $theta$ as noise around the old one.
The cure is again a model that admits change.

== Stability, plasticity and forgetting

Let $q_A$ be the old task's input distribution and $q_B$ the new one's. Define:

- *Plasticity:* $cal(P)^"pred"$ with queries drawn from $q_B$.
- *Stability:* the expected change of old-task predictions,
  $cal(S) = bb(E)_(x tilde q_A) [ D_"KL" (p(y | x, D_(1:t)) || p(y | x, D_(1:t+1))) ]$. Small $cal(S)$ is stable.
- *Forgetting:* the rise in old-task log loss,
  $cal(F) = bb(E)_((x, y) tilde A) [ log p(y | x, D_(1:t)) - log p(y | x, D_(1:t+1)) ]$.

Stability and forgetting differ.
A change of old-task predictions can lower the old-task loss: that is learning, not forgetting.

In exact static Bayes nothing is discarded #rig.
The old likelihood stays in the product forever.
Any change of old-task predictions is a revision the model judges warranted.
Harmful forgetting therefore has only two sources #cond:
an approximate posterior that compressed away old information, or a model that assumes the world changes.
The second is deliberate: it trades old information for plasticity, at a rate set by the change model.

The trade-off between stability and plasticity is not universal.
It is forced only where old and new data constrain the same directions.
@sec-gaussian makes this exact for Gaussians, and @sec-switch for a task switch.

== Is there a better definition than variance?

Yes, in two steps.

+ Replace parameter variance by predictive information gain $cal(P)^"pred"$. This fixes the symmetry problem and the direction problem at once.
+ In a model that admits change, plasticity no longer decays to zero. It settles at a floor set by the change rate (@fig-kalman). The #emph[floor] is then the design quantity. It is a property of the world model, not of the current posterior.

A Bayesian theory of plasticity is therefore mainly a theory of #emph[which change model to assume].
