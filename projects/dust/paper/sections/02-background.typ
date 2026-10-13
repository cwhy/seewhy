#import "/template.typ": *

= Background <sec-background>

This section defines everything the later sections lean on. A reader who writes
code and reads statistics, but has not trained a neural network, should be able
to follow the rest of the paper from here.

== What is being trained

A #gloss[language model][a program that reads text and predicts what comes next]
is trained on a stream of #gloss[tokens][integers standing for common fragments of
text; here 4,096 of them, found by byte-pair encoding]. Given the first $t$ tokens
of a sequence it outputs a probability for each of the 4,096 possible next tokens.
The per-token loss is the negative log of the probability it gave to the token that
actually came next,

$ ell_t = -log p(y_t | x_1, ..., x_t), $

and the loss $L$ is the mean of $ell_t$ over every token in a batch. A model that
guesses uniformly scores $ln 4096$; a model that has learned nothing but how often
each token occurs scores less; getting under that floor requires using context
(@sec-task).

The model is a #gloss[transformer][a stack of identical blocks; each block lets every
position look at earlier positions (_attention_) and then transforms each position on
its own (an _MLP_)] @vaswani2017. All of its behaviour is set by $d approx 3.8 times 10^7$
numbers, the #gloss[parameters][the weights; a vector $theta$]. Training is a search for
the $theta$ that minimises $L$. We use the architecture of the DUST paper @dahal2026dust,
a variant of the modded-nanoGPT recipe @jordan2024modded, described in @sec-method.

== The gradient, and how backpropagation gets it

If we knew the gradient $nabla L$, a vector that points uphill in $L$, we could step
the other way. Backpropagation @rumelhart1986 computes it exactly by applying the chain
rule from the output back to the input. It is a second pass through the network, run
in reverse, and it needs two things: every operation must be differentiable, and every
intermediate value from the forward pass must be kept until the backward pass uses it.
It is the only credit-assignment algorithm that has trained modern language models.

This paper asks what happens when the backward pass is removed.

== Gradients from forward passes alone

Evaluating $L$ at a slightly changed $theta$ costs one forward pass and needs no
derivative. Perturb the parameters by a random direction $epsilon$ and the loss changes
by about $sigma epsilon dot nabla L$. Do it in both directions (an #gloss[antithetic
pair][two evaluations at $theta + sigma epsilon$ and $theta - sigma epsilon$, which share
their random draw]) and the difference isolates that term:

$ f = L(theta + sigma epsilon) - L(theta - sigma epsilon) approx 2 sigma epsilon dot nabla L . $

Multiply by $epsilon$ and average over many draws and what survives is the gradient:

$ hat(g) = 1/(P sigma^2) sum_(i=1)^P f_i sigma epsilon_i, quad EE[hat(g)] = 2 nabla L . $

This is #gloss[evolution strategies][a family of optimisers that score a _population_ of
randomly perturbed copies of the parameters and move toward the better ones] @salimans2017es
in the form used here. The factor of two is harmless to Adam @kingma2015adam, which rescales
each coordinate by its own typical size.

*The cost is variance.* Each pass returns one number, and that number has to steer $d$
parameters. For a random $epsilon$ in $d$ dimensions, the cosine between $hat(g)$ and
$nabla L$ grows only as $sqrt(P / d)$ @nesterov2017. With $d approx 3.8 times 10^7$ and
$P = 1000$ passes, that is about $0.005$: the estimate is, to a very good approximation,
noise with a faint trace of the gradient in it. The reason evolution strategies work at
all on small networks is that $d$ is small. The reason they are not used to pretrain
language models is that $d$ is not.

== EGGROLL: make each population member cheap

A population of $P$ perturbed networks is expensive to hold: each member needs its own copy
of every weight matrix. EGGROLL @sarkar2026eggroll replaces the dense perturbation of an
$m times n$ matrix by a #gloss[rank-one][outer-product] one,

$ E = sigma a b^top, quad a in RR^m, quad b in RR^n, quad a_j, b_k tilde.op N(0, 1), $

which costs $m + n$ numbers rather than $m n$, is never written out, and is applied in the
forward pass as a side branch, $x W^top + sigma (x dot b) a^top$: two thin products per
layer. The update is a weighted sum of $P$ such terms, so its rank can reach $P$ even
though each term has rank one. Biases and other vectors get an ordinary Gaussian.

This is the estimator used throughout this repository (@sec-method): rank one, antithetic
pairs, raw loss differences, Adam. It makes a member cheap. It does nothing about how little
a member tells you: one scalar per member, steering $d$ parameters.

#fig(image("/diagrams/eggroll_update.svg", width: 100%),
  caption: [EGGROLL on one weight matrix. A member is a rank-one perturbation, an outer product of
    two random vectors, so it costs $m + n$ numbers rather than $m n$. A pair of forward passes with
    $plus.minus$ the perturbation yields one number. The update is the sum of $P$ perturbations
    weighted by those numbers, passed to Adam.])

#fig(image("/diagrams/pass_value.svg", width: 100%),
  caption: [What one forward pass is worth. EGGROLL perturbs weights, so a pass is one
    population member and returns one number, the mean loss. DUST perturbs activations
    independently at every token, so the same pass is one member per token and returns one
    loss per token. Layers are schematic; the real model has eight blocks.])

== DUST: perturb the activations, one member per token

DUST @dahal2026dust keeps the idea of scoring random perturbations but changes what is
perturbed. It adds noise to the *output of a linear layer*, $y_t arrow.r y_t + sigma a_t$,
with fresh $a_t tilde.op N(0, I)$ at every token $t$. This is #gloss[node perturbation][noise
injected into a layer's activations rather than its weights] @widrow1990 @werfel2003, and the
usual argument for it is dimension: a layer's output has $d_"out"$ entries and its weights
$d_"out" times d_"in"$.

The transformer adds something a classifier does not have. A batch is $8 times 2048$ tokens,
and the loss is a sum of one term $ell_t$ per token. If token $t$'s loss responds mainly to
token $t$'s own noise, then every token is an independent experiment, and one forward pass
runs about sixteen thousand of them. The authors call this a *virtual population*: it lives on
the token axis, orthogonal to the batch axis that EGGROLL's members already occupy.

For one linear layer $y = W x$, one #gloss[draw][one forward pass with fresh noise at every
token of the selected layers] gives each token a reward, and $K$ draws give an estimate of the
gradient of the loss with respect to that layer's output. With $r_t^((i))$ the reward and
$a_t^((i))$ the noise of draw $i$ at token $t$:

$ r_t^((i)) = overline(ell)_t - ell_t^((i)), quad
  hat(g)_t = - 1/(K sigma) sum_(i=1)^K r_t^((i)) a_t^((i)), quad
  hat(G)_W = sum_t hat(g)_t x_t^top . $

The bar is the mean over the draws evaluated together. The first equation is a reward: positive
when this draw's noise happened to lower token $t$'s loss. The second turns reward-weighted noise
into an estimate of the *error at the layer's output*, $partial L slash partial y_t$, the thing
backpropagation obtains from the chain rule. The third multiplies that error by the layer's
cached input to give the weight gradient. Backpropagation forms the same outer product with
the same input; the only difference is where $hat(g)_t$ comes from.

#fig(image("/diagrams/reward_to_gradient.svg", width: 100%),
  caption: [DUST on one linear layer. The clean forward pass caches the input $x_t$. Each
    draw adds fresh noise at the layer's output, reruns the rest of the network, and reads off
    each token's loss. Rewards weight the noise into an output-error estimate, and the outer
    product with the cached input is the weight gradient. For an embedding table, $x_t$ is
    one-hot and the outer product is a scatter-add into the token's row.])

Three details turn this into something that trains a transformer; the paper calls them the
whole algorithm beyond the two ideas above.

*Interference.* A draw that jitters every layer at every token makes each token's loss depend on
all the noise, not just its own. DUST therefore spends extra passes to keep the sources apart.
The two residual-stream writers of a block (the attention projection and the MLP output) are
jittered together; the MLP's first layer, the attention output, and the token embedding each get
their own passes, with their own noise scales. A draw that jitters block $l$ reruns only blocks
$l$ onward, because the clean pass cached everything before it.

*Attention internals.* A token's query, key and value matter through other tokens' outputs, so a
token's own loss barely registers them. DUST scores them against the *estimated error at the
attention output* from the previous item: it recomputes only the block's attention with the
jitter and takes $c_s = chevron.l hat(g)_s, Delta o_s chevron.r$ per head, the loss change the
estimate predicts. Token $t$'s jitter is credited with the scores of the tokens that can see it,

$ r_t = sum_(s >= t) gamma^(s - t) c_s , $

with a decay $gamma$ near one for keys, values, gates and value embeddings, and zero for queries,
which only affect their own position. Everywhere else $gamma = 0$.

*The head and the scalars.* The output head is jittered directly on the cached logits, one
256-wide slice of the 4,096-word vocabulary at a time, recomputing only the cross-entropy: far
cheaper than a forward pass. The $2L$ residual-mixing scalars are not activations, so they use
ordinary two-sided weight-space ES with eight draws.

*Population.* The paper counts $K$, the number of draws rewarded with token losses, and does not
count the head's slabs or the attention-internal draws, which it reports as a small fraction of
the update's compute. A draw is one forward pass, slightly cheaper because of the cache. For
EGGROLL the population is forward passes of the batch, two per antithetic pair. This paper keeps
both conventions and reports measured wall-clock beside them, because the headline claim is about
efficiency.

== Related work

Training networks with forward passes alone is old: node perturbation goes back to Widrow and
Lehr @widrow1990, with a learning-curve analysis by Werfel et al. @werfel2003, and evolution
strategies were revived for deep networks by Salimans et al. @salimans2017es. MeZO @malladi2023mezo
showed that language models can be fine-tuned with forward passes at inference-level memory; the
open question was pretraining from scratch, where the loss landscape is far from a good
initialisation. Ren et al. @ren2023 combine activation perturbation with local losses and
forward-mode differentiation. A separate line replaces the global backward pass with local
dynamics; one such method, predictive coding with Lagrange multipliers @seely2026pcalm, is
studied in the sibling project #raw("local-learning"). What remains open, and what this paper
tests, is whether DUST's claimed advantage over weight-space ES holds up outside the authors'
own implementation of the baseline.
