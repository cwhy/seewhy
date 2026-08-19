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
$"logits" = "LN"(h) U^top$. Attention uses *4 heads* throughout, which the paper
does not state and we took from the authors' released code. The choice is not
neutral for us: more heads means more independent random circuits for
embedding-only training to select among, so the larger value we first used would
have flattered the condition under test.

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
  ("learning-rate schedule", "500-step linear warmup, then cosine decay to zero"),
  ("gradient clipping", "global norm 1.0"),
  ("depth", "2 blocks (4 also swept for language modeling)"),
  ("attention heads", "4"),
  ("main-table widths", "1024 and 16"),
  ("width sweep", "16, 32, 64, 128, 256, 512, 1024"),
  ("evaluation interval", "every 250 steps (5000 for memorization)"),
)

#callout(title: [The schedule is not in the paper, and it decides the baseline])[
  Appendix D.3 of the paper gives only "AdamW optimizer with a learning rate
  $10^(-3)$ and weight decay $10^(-3)$ ... We clip all gradient norms at 1."
  There is no warmup and no decay in that description. Under it we could not
  reproduce the paper's fully trained results at all. A fully trained width-1024
  model plateaus at 0.14 on needle-in-a-haystack. It stays there for 10,000
  steps — the paper's own budget.

  The authors' released code sets `warmup_steps = 500` and
  `lr_scheduler_type = "cosine"`. With both, the published learning rate works.
  Embedding-only training reaches 1.00 with or without either — which is exactly
  why the omission leaves no trace in the paper's own tables. @sec-analysis
  returns to this, because it is a small piece of evidence for the paper's
  thesis rather than only an erratum.
]

Budgets are per task, and are the main place we depart from the paper — see
@sec-limitations. The paper uses 10,000 steps at batch 1000 for the streamed
tasks. A convergence probe found needle-in-a-haystack saturating after roughly
750,000 training examples, regardless of batching (step 750 at batch 1000, step
2000 at batch 250). Budgets were initially set from that measurement.
That was an error, and worth stating: a budget shared by several conditions has
to be calibrated on the *slowest* of them, and embedding-only training is the
fastest by an order of magnitude. The budgets below are set by the fully trained
condition instead:

#kv(
  ("modular addition", "20,000 steps x batch 1000"),
  ("needle in a haystack", "8,000 steps x batch 500"),
  ("decimal addition", "8,000 steps x batch 500"),
  ("parenthesis balancing", "6,000 steps x batch 500"),
  ("memorization", "60,000 steps x batch 8192"),
  ("circuit imitation", "6,000 steps x batch 256"),
  ("language modeling", "one pass over 100M tokens, batch 32 x 512 tokens"),
)

== Seeds, aggregation, hardware

The main table and the ablation use *5 seeds* per cell. The width sweep,
memorization and circuit imitation use *3*. The language-modeling sweep is a
*single run* per cell. Reported numbers are medians across seeds,
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
