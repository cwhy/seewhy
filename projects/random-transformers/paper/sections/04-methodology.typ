#import "/template.typ": *

= Methodology <sec-methodology>

== Notation

#notation(
  ($d$, [hidden width — the length of the vectors flowing through the stack]),
  ($m$, [number of transformer blocks (depth)]),
  ($v$, [vocabulary size]),
  ($n$, [maximum context length, so the height of the positional table]),
  ($E_"tok"$, [token embedding matrix, $v times d$]),
  ($E_"pos"$, [positional embedding matrix, $n times d$]),
  ($U$, [unembedding matrix, $v times d$]),
  ($F$, [the frozen interior: the whole stack of $m$ blocks]),
  ($h^((i))$, [activations after block $i$; $h^((0))$ is the embedding output]),
  ($M$, [the scored mask — the positions whose prediction enters the loss]),
)

== The model

A GPT-2-style decoder-only transformer @radford2019language, pre-layernorm, no
dropout, no weight tying. Writing $"LN"$ for layer normalisation (rescaling a
vector to zero mean and unit variance, then applying a learned scale and shift),
one block is

$ h &<- h + "Attn"("LN"(h)) \
  h &<- h + "MLP"("LN"(h)) $

with causal multi-head attention and $"MLP"(x) = "GeLU"(x W_"fc" + b_"fc") W_"proj" + b_"proj"$
where $W_"fc"$ is $d times 4d$. After the last block a final $"LN"$, then
$"logits" = "LN"(h) U^top$. Attention uses 8 heads throughout.

Input embeddings are the sum of the two tables:

$ h^((0))_t = E_"tok" [x_t] + E_"pos" [t] $

*Initialisation.* Feed-forward weights are drawn from $cal(N)(0, (0.02 slash sqrt(2m))^2)$.
Every other weight matrix — query, key, value, the attention output projection,
and all three embedding tables — is drawn from $cal(N)(0, 0.02^2)$. Biases are
zero and layer-norm affines start at identity. The paper does not say which
group the attention output projection belongs to; we read it as an "other weight
matrix". This is our choice, not the paper's.

== Training regimes

The single axis that varies. All five optimise the same loss with the same
optimiser on the same data, and differ only in which parameters receive updates:

#kv(
  ("full", "everything — the paper's 'normal transformer'"),
  ("random", "E_tok, E_pos, U — the paper's 'random transformer'"),
  ("u_only", "U alone"),
  ("e_only", "E_tok and E_pos"),
  ("etoken_u", "E_tok and U, positional embeddings frozen"),
)

Frozen parameters are passed to the loss as non-differentiated arguments, so the
compiler never builds their weight-gradient computations. Gradients still
propagate *through* them to reach the embeddings.

== Loss

Cross-entropy over the scored positions only:

$ cal(L) = - 1/(|M|) sum_((b,t) in M) log p(x_(b,t+1) | x_(b, <= t)) $

Language modeling scores every position, so there $M$ is everything.

For circuit imitation the loss is instead the KL divergence from the target
model's next-symbol distribution $tilde(p)$ to the student's, averaged over
positions and over uniformly random inputs:

$ cal(L) = EE_x [ "KL"( tilde(p)(dot | x) || p(dot | x) ) ] $

== Metrics

*Sequence accuracy* — every scored position in a sequence correct. This is the
strict metric and the one the tables report; on decimal addition it means all
eleven or twelve output symbols at once. *Token accuracy* is logged beside it and
differs only on decimal addition. For memorization, an association counts as
stored if the model's highest-scoring output matches, and *bits per trainable
parameter* is $9$ bits times the number stored, divided by the trainable
parameter count.

For the subspace analysis we collect activations at $h^((0))$ and after each
block, over positions the model has actually read (padding excluded, since a
constant vector repeated across sequences would deflate the variance
denominator). We then report, from the same activations and against the same
total variance, the fraction explained by the top 10 principal components and
the fraction explained by the 10 highest-variance individual neurons.

== Hyperparameters

AdamW @loshchilov2019decoupled throughout, with gradient norms clipped at 1.0.

#kv(
  ("optimiser (synthetic tasks)", "AdamW, lr 1e-3, weight decay 1e-3"),
  ("optimiser (language modeling)", "AdamW, lr 6e-4, weight decay 0.1"),
  ("optimiser (LSTM baseline)", "AdamW, lr 5e-3, weight decay 1e-3"),
  ("gradient clipping", "global norm 1.0"),
  ("depth", "2 blocks (4 also swept for language modeling)"),
  ("attention heads", "8"),
  ("main-table widths", "1024 and 16"),
  ("width sweep", "16, 32, 64, 128, 256, 512, 1024"),
  ("evaluation interval", "every 250 steps (5000 for memorization)"),
)

Budgets are per task, and are the main place we depart from the paper — see
@sec-limitations. The paper uses 10,000 steps at batch 1000 for the streamed
tasks; a convergence probe found needle-in-a-haystack saturating after roughly
750,000 training examples regardless of how they were batched (step 750 at batch
1000, step 2000 at batch 250), so budgets were set from that measurement with a
two- to fourfold margin rather than copied:

#kv(
  ("modular addition", "20,000 steps x batch 1000"),
  ("needle in a haystack", "3,000 steps x batch 500"),
  ("decimal addition", "6,000 steps x batch 500"),
  ("parenthesis balancing", "4,000 steps x batch 500"),
  ("memorization", "60,000 steps x batch 8192"),
  ("circuit imitation", "6,000 steps x batch 256"),
  ("language modeling", "one pass over 100M tokens, batch 32 x 512 tokens"),
)

== Seeds, aggregation, hardware

The main table and the ablation use *5 seeds* per cell; the width sweep, the
memorization and circuit-imitation experiments use *3*; the language-modeling
sweep is a *single run* per cell. Reported numbers are medians across seeds,
following the paper, with the min–max range across seeds given alongside in
@sec-results. The paper uses 10 seeds; ours is a compute-driven reduction and
the medians are correspondingly noisier.

A seed determines both the random initialisation and the training data stream,
so two seeds differ in the frozen network *and* in what it sees. This is the
paper's protocol. It means seed spread bounds the combined variability rather
than isolating the initialisation, which matters when reading the width sweep.

Everything ran on a single machine with two NVIDIA RTX 4090s (24 GB each), in
JAX. A width-1024 run takes between two and four minutes per cell depending on
task.
