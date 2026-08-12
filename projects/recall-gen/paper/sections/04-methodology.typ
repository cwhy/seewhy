#import "/template.typ": *

// OBLIGATIONS
//  - Reimplementable from this section alone.
//  - Model: equations or pseudocode. Loss written out.
//  - Complete hyperparameter table. Seeds and repeats. Hardware, wall-clock, params.

= Methodology

== Notation

#notation(
  ($M$, [number of context images in an episode]),
  ($Q$, [number of masked query images in an episode]),
  ($R$, [number of image rows hidden from each query]),
  ($N$, [sequence length, $M + Q$]),
  ($d$, [width of the model's internal vector, 256]),
  ($H$, [number of parallel memory "heads", 4]),
  ($d_k$, [size of each head's key and value vectors, 64]),
  ($S_t$, [the memory state after reading token $t$; $H$ matrices of $d_k times d_k$]),
  ($alpha_t$, [forget gate — how much of the memory survives step $t$]),
  ($beta_t$, [write strength — how strongly token $t$ is written into memory]),
)

== The model

#v(3pt)
#fig(include "/diagrams/architecture.typ", caption: [
  The architecture. *(a)* An episode. The $M$ context images write into the
  recurrent state; the $Q$ masked queries only read from it, and the state is
  fixed at 16 384 numbers however large $M$ is — so it is both the only route
  from context to query and a bottleneck that tightens as the context grows.
  *(b)* The stack every token passes through, and one layer expanded. The state
  is written and read once per layer, and the four layers do not share one.
])

Each token — a context image or a masked query — is first turned into a vector
of width $d$ by two learned linear maps, one applied to the 784 pixel values and
one to the 784-entry binary mask, plus a learned vector marking whether the
token is context or query:

$ h^((0))_t = W_"pix" x_t + W_"msk" m_t + r_(c(t)) $

Those vectors then pass through four identical layers. Each layer is a memory
module followed by a two-layer feed-forward network, both wrapped in
#gloss[residual connections][the layer's output is added to its input rather
than replacing it, so information can skip past any layer] and preceded by
#gloss[layer normalisation][rescaling each vector to zero mean and unit variance,
which keeps the numbers in a stable range as they pass through many layers].

=== The memory

The memory module is Kimi Delta Attention @kda2025, a *linear recurrent
network*: it reads the sequence one token at a time, carrying a fixed-size state
forward, in contrast to #gloss[attention][the standard mechanism in which every
token compares itself against every other token, at a cost that grows with the
square of the sequence length], which keeps every token available and re-reads
them all on every query.

What makes it useful here is that the state is a *matrix* rather than a vector,
and behaves like an associative table. Each token produces a query, a key and a
value ($q_t, k_t, v_t$, each of size $d_k$ per head, with $q$ and $k$ scaled to
unit length), a per-channel forget gate $alpha_t in (0,1)^(d_k)$ and a write
strength $beta_t in (0,1)$. The state is updated by the *delta rule* — look up
what is currently stored under this key, and write only the difference:

$
tilde(S)_t &= S_(t-1) "diag"(alpha_t)               & quad & "forget" \
hat(v)_t   &= tilde(S)_t k_t                        & quad & "what is already stored at" k_t \
e_t        &= beta_t (v_t - hat(v)_t)               & quad & "the error" \
S_t        &= tilde(S)_t + e_t k_t^top              & quad & "write" \
o_t        &= S_N q_t \/ sqrt(d_k)                  & quad & "read"
$

Two gating choices make this an experiment about the state rather than about
attention. *Context tokens write; query tokens do not* ($beta_t = 0$ and
$alpha_t = 1$ for queries). And every token reads the *completed* state $S_N$,
not a running prefix. Together these mean the only path by which a query can
learn anything about the context is through $S_N$ — a hard bottleneck of
$H d_k^2 = 4 times 64 times 64 = 16 thin 384$ numbers, fixed no matter how
large $M$ is. There is no attention over the context images and no pooling; the
state *is* the aggregate.

Because queries never write, the $Q$ queries in an episode are independent
probes of one shared encoding of the context.

The forget gate is initialised so that its *horizon* — the number of tokens over
which a memory decays to $1 slash e$ of its strength — is eight times the
sequence length. The context is a set, not a sequence, so any decay is an
arbitrary bias against whichever images happen to come first; initialising the
horizon well beyond the episode makes that bias negligible at the start while
leaving the gate free to learn.

=== The output

The final vector of each query token is layer-normalised, mapped to 784 numbers
by a learned linear map and passed through a #gloss[sigmoid][a smooth function
squashing any number into the range $(0,1)$], matching the pixel range.

== The loss

Only hidden pixels of query tokens are scored. Writing $m$ for the binary mask
(1 on hidden pixels), $hat(y)$ for the model's output and $y$ for the true
image, the loss for a batch of $B$ episodes is

$ cal(L) = 1/(B Q sum_i m_i) sum_(b=1)^B sum_(q=1)^Q sum_(i=1)^784 m_i (hat(y)_(b q i) - y_(b q i))^2 $

The visible half of the output is never scored and is therefore unconstrained
noise; figures showing model completions paste the prediction into the query's
visible half rather than displaying the raw output.

== Training modes

The three arms of the paper differ only in how a query's target is chosen:

/ recall: the target is one of the $M$ context images, chosen uniformly. The
  answer is always present.
/ completion: the target is a fresh image from the pool, essentially never in the
  context. The answer is never present.
/ mixed: each query independently takes the recall target with probability 0.5.

Everything else — architecture, optimiser, data, evaluation — is shared.

== Training setup

#kv(
  ("optimiser", "AdamW (per-parameter step sizes, decoupled weight decay 0.01)"),
  ("learning rate", "3e-4, warm-up over 300 steps then cosine decay to 10%"),
  ("gradient clipping", "global norm 1.0"),
  ("steps", "12000"),
  ("batch", "256 episodes (M <= 64); 64 for M = 256"),
  ("parameters", "4.03M"),
  ("evaluation", "512 fixed episodes per condition, one fixed seed, shared by every run"),
)

Evaluation episodes are constructed once with a fixed seed and reused across
every experiment, so two runs are always compared on literally the same
episodes. Metrics are recorded every 500 steps; the reported number is the final
step unless explicitly labelled as the best over training.

*Seeds.* Every configuration is a single run unless stated otherwise. Recall
training at $M = 16$ is replicated at three seeds and completion training at
$M = 16$ at two; §6 reports the spread. Every other row in this paper is one
run, and the reader should treat differences smaller than the seed spread
reported there as noise.

*Hardware and wall-clock.* One NVIDIA RTX 4090. Sequence length is short, so the
per-token scan is dominated by kernel-launch overhead rather than arithmetic:
step time is nearly flat in batch size, and quadrupling the batch costs about
20% more per step. A 12 000-step run at $M = 16$ takes about 5 minutes; $M = 64$
takes about 20; $M = 256$ takes about 22.

== Reference points

Because a negative result is only as good as what it is measured against, four
model-free strategies are computed on the same evaluation episodes:

/ mean image: predict the average training image. This is the normaliser, 1.0 by
  construction.
/ ridge: a single linear map from the 392 visible pixels to the 392 hidden ones,
  fitted by ridge regression on the training pool, with the penalty chosen on a
  held-out sixth of it. This ignores the context entirely — it is the best a
  *distributional* answer can do with no in-context information.
/ nearest neighbour: find the context image whose visible half is closest to the
  query's, and copy its hidden half. The best a pure *look-up* can do.
/ soft look-up: a softmax-weighted average of all $M$ context images' hidden
  halves, weighted by visible-half similarity, with the temperature swept and the
  best reported. This is the strongest answer obtainable from the context alone,
  and it is precisely the shape of computation the linear-attention state can
  perform — so it is the ceiling a retrieval mechanism is being measured against.
