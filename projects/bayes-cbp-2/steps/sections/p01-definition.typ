#import "/template.typ": *

= Defining plasticity from Bayesian first principles <p1>

#callout(title: [The question])[
  If the network's parameters have a posterior $p(theta | D)$, and each old posterior becomes the next prior:
  how should plasticity be defined? Can it be measured by KL divergence, information gain or change of predictions?
  Is it the same as posterior uncertainty? Why does much old data make new tasks hard?
  How do stability, plasticity and forgetting differ? Is there a better definition than variance?
]

== Core view

Plasticity is the information the learner #emph[expects] to absorb from the next batch:
the mutual information between parameters and the next batch, given the past.
Its function-space version — information about predictions rather than parameters — is more fundamental.
Under a model that assumes a fixed world, it can only fall.
So for an exact Bayesian learner, losing plasticity is correct inference, not a malfunction.

== Derivation

=== Setting
After $t$ batches the posterior is $p(theta | D_(1:t))$, and

$ p(theta | D_(1:t+1)) = frac(p(D_(t+1) | theta) thin p(theta | D_(1:t)), p(D_(t+1) | D_(1:t))). $

This is exact when batches are #gloss[conditionally independent given $theta$][once $theta$ is known, one batch says nothing about another].
Call this the #emph[static model]: one $theta$ generates all data.

=== Definition (P1.1, P1.2)
#gloss[KL divergence][$D_"KL" (p || q) = bb(E)_p [log p\/q]$; zero only when $p = q$] measures how far the posterior moves.
Averaged over the batch before it arrives #rig:

$ cal(P)_t = bb(E)_(D_(t+1) | D_(1:t)) [ D_"KL" (p(theta | D_(1:t+1)) || p(theta | D_(1:t))) ] = I(theta ; D_(t+1) | D_(1:t)). $

We define $cal(P)_t$ as #emph[Bayesian plasticity].
The realised KL for one batch is a different quantity: a surprising batch can move a low-plasticity model a lot.

=== Plasticity can only fall
#proposition("Monotone plasticity", rig)[
  Let $X_1, X_2, dots$ be conditionally i.i.d. given $theta$, with finite entropies.
  Then $cal(P)_n = I(theta ; X_(n+1) | X_(1:n))$ is non-increasing in $n$.
]
_Proof._ $I(theta ; X_(n+1) | X_(1:n)) = H(X_(n+1) | X_(1:n)) - H(X | theta)$.
The second term does not depend on $n$.
Conditioning cannot raise entropy, and the sequence is #gloss[exchangeable][any reordering has the same joint law], so
$H(X_(n+2) | X_(1:n+1)) <= H(X_(n+2) | X_(2:n+1)) = H(X_(n+1) | X_(1:n))$. $square$

The total is also bounded: $sum_(n < N) cal(P)_n = I(theta ; X_(1:N))$, which grows like $d/2 log N$ for a regular $d$-parameter model @clarke1990.
So per-step plasticity decays like $d \/ (2n)$ #cond.

=== Uncertainty is not plasticity (P1.3)
Take a parameter that the likelihood does not depend on.
Its posterior keeps its full prior variance, yet it contributes zero information to $cal(P)_t$ #rig.
So large posterior variance does not imply plasticity.
The converse holds in one direction only: in a discrete parameter space $cal(P)_t <= H(theta | D_(1:t))$, so plasticity needs uncertainty #rig.
Uncertainty is necessary, not sufficient. What matters is uncertainty #emph[in directions the new data can inform].
Stage P2 makes this quantitative.

=== Why much old data blocks new learning (P1.4)
Take scalar data with equal noise: $n_A$ points about task $A$, $n_B$ about task $B$, both read as evidence about one $theta$.
The posterior mean is the pooled mean. It moves from task $A$'s value toward task $B$'s by the fraction #rig

$ frac(n_B, n_A + n_B). $

With $n_A = 10^6$ and $n_B = 10^3$, it moves 0.1% of the way.
This is right if $A$ and $B$ share one $theta$. If the task changed, the model is #emph[misspecified]: it reads evidence about a new $theta$ as noise.

=== Stability, plasticity and forgetting (P1.5)
With $q_A$, $q_B$ the input distributions of old and new tasks:

- *Plasticity:* $cal(P)_t$, or its predictive version below, with queries from $q_B$.
- *Stability:* $cal(S) = bb(E)_(x tilde q_A) [ D_"KL" (p(y | x, D_(1:t)) || p(y | x, D_(1:t+1))) ]$, small when old predictions stay put.
- *Forgetting:* $cal(F) = bb(E)_((x, y) tilde A) [ log p(y | x, D_(1:t)) - log p(y | x, D_(1:t+1)) ]$, the rise in old-task log loss.

Stability and forgetting differ: a change of old predictions can lower old-task loss.
In exact static Bayes nothing is discarded #rig: the old likelihood stays in the product.
So harmful forgetting has two possible sources #cond: an approximate posterior that compressed old information away, or a model that assumes the world changes.

=== A better definition (P1.6)
Parameter information counts moves that change no prediction.
Neural networks have many: swapping two hidden units, or rescaling across a ReLU.
Let $f$ be the predictions on a set of query inputs. Define

$ cal(P)_t^"pred" = I(f ; D_(t+1) | D_(1:t)) <= cal(P)_t , $

where the inequality is the #gloss[data-processing inequality][a function of $theta$ carries no more information than $theta$] #rig.
$cal(P)^"pred"$ is invariant to reparametrisation and measures what users see. We take it as the more fundamental definition.

== When it holds

- The identity $cal(P)_t = I(theta ; D_(t+1) | D_(1:t))$ holds for any Bayesian model.
- Monotonicity needs conditional independence given a fixed $theta$, and finite entropies.
- The $d \/ (2n)$ rate needs a regular parametric model (a smooth, identifiable likelihood).

== Limits

- Neural networks trained by SGD are not Bayesian learners. Stage P1 explains what an ideal learner would do; how SGD departs from it is Stage P3's and P4's topic.
- $cal(P)^"pred"$ depends on the choice of query inputs, and is hard to estimate.
- An expected quantity says nothing about one surprising batch.

== Alternatives

#table(
  columns: (auto, 1fr, 1fr),
  [Definition], [Strength], [Weakness],
  [Posterior variance], [Cheap; intuitive], [Counts directions no data can reach (shown above)],
  [Realised KL per batch], [Observable after the fact], [Depends on the batch; not a property of the learner],
  [Expected information gain $cal(P)_t$], [Principled; provably monotone in a static world], [Counts symmetric moves],
  [Predictive information gain $cal(P)^"pred"_t$], [Invariant; measures what matters], [Needs a query distribution],
  [Value of information (expected drop in Bayes risk)], [Ties plasticity to a task loss], [Depends on the loss chosen],
)

The last two agree on log loss: then the expected drop in risk is a predictive information gain #cond.

== Next step

Proposition 1 says plasticity loss is built into the static model.
Two questions follow.
How large is plasticity, quantitatively, and how does it depend on precision and noise? That is Stage P2.
And what must change in the model for plasticity to stop decaying? Stage P2 gives the first answer in the Gaussian case.
