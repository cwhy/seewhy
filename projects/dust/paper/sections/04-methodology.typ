#import "/template.typ": *

= Methodology <sec-method>

== Model

A 37,749,776-parameter decoder-only transformer, a JAX port of the model in the DUST release
@dahal2026dust. Eight blocks of width 512 with eight heads of 64; RMS normalisation without a
learned gain; no biases; a squared-ReLU MLP of width 2,048; rotary position embeddings followed by
RMS-normalised queries and keys; causal attention with a left window of 1,024 tokens in blocks 1-3
and 5-7 and the full 2,048 in blocks 4 and 8; an untied head with logits capped as
$15 tanh(z \/ 15)$. Every other block adds a #gloss[value embedding][a second token embedding added
to the block's values, scaled per head by $2 sigma(g)$ where the gate $g$ is a linear function of
the first 32 channels of the block input]. The residual stream entering block $i$ is
$lambda_i x + mu_i x_0$, with $x_0$ the normalised token embedding and $lambda_i, mu_i$ learned
scalars, initialised at 1 and 0.1.

We checked the port against the authors' PyTorch model on identical randomised weights. Per-token
losses agree to $3.5 times 10^(-3)$ at most, which is the size of the difference between the GPU's
TF32 matmuls and PyTorch's CPU float32; the autograd gradients of the two frameworks have cosine 1.00
on every layer type and equal norm.

== The three learning rules

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  align: (left, left, left, left),
  [], [*Backprop*], [*EGGROLL*], [*DUST*],
  [gradient], [chain rule], [$1\/(P sigma^2) sum_i f_i epsilon_i$, rank-1 $epsilon$], [$sum_t hat(g)_t x_t^top$ from per-token rewards],
  [optimiser], [SGD + momentum], [Adam], [SGD + momentum],
  [population], [--], [forward passes (2 per pair)], [draws $K$ (head, attention internals extra)],
  [source], [`lib/model.py`], [`lib/eggroll.py`], [`lib/dust.py`],
)

Equations for each are in @sec-background. DUST's draw allocation, noise scales and credit decay
are the authors' released values (their Appendix E), scaled by $K slash 256$ for populations of 256 and
up; their own 64-draw recipe is used at 64. EGGROLL is the setting of this repository's
`local-learning` project, unchanged: rank 1, antithetic pairs, raw loss differences, Adam. Its two
free constants, the Adam rate and the perturbation scale $sigma$, are tuned per population on one
seed.

=== DUST: what one update jitters

All counts are draws per update at $K = 256$ and 1M tokens (the authors' released recipe,
Appendix E of @dahal2026dust); other populations scale every count by $K slash 256$. "Reruns" says
how much of the network one draw recomputes: a draw that jitters block $l$ restarts from the clean
pass's cached stream entering that block.

#table(
  columns: (1.6fr, 1.5fr, auto, auto),
  align: (left, left, right, right),
  [*Site*], [*One draw reruns*], [*Draws*], [*Noise scale*],
  [Residual writers (attention and MLP output projections), jittered together], [blocks $l$ to 8], [110], [0.2],
  [MLP first layer], [blocks $l$ to 8], [110], [0.2],
  [Attention output (input of the projection)], [blocks $l$ to 8], [26], [0.2; blocks 5-8: 0.4],
  [Token embedding], [all blocks], [10], [0.2],
  [Queries, per block], [that block's attention], [2], [0.05],
  [Values, per block], [that block's attention], [20], [0.05],
  [Value-embedding gate, per block that has one], [that block's attention], [6], [0.05],
  [Keys, per block], [that block's attention], [12], [0.05],
  [Value embedding, per block that has one], [that block's attention], [8], [0.05],
  [Head], [one 256-column slab's cross-entropy], [1,024], [0.05],
  [Residual scalars (weight-space ES, two-sided)], [the whole network, twice], [8], [0.03],
)

The first four rows sum to the population, $K = 256$. The per-block split of the writer and MLP
draws is tilted toward the early blocks: $(38, 32, 16, 10, 6, 4, 2, 2)$ each, and the attention
output gets $(6, 6, 4, 2, 2, 2, 2, 2)$. Queries get no credit from later tokens ($gamma = 0$); keys,
values, gates and value embeddings use $gamma = 0.98$. Draws are evaluated $n$ at a time in one
batched forward, with $n = 1, 2, 8, 8, 16$ at $K = 64, 256, 1024, 4096, 16384$; the reward is
centred across those $n$.

=== Optimisers

#table(
  columns: (1.5fr, auto, auto, auto, 1.2fr),
  align: (left, right, right, right, left),
  [*Method*], [*Rate*], [*Momentum*], [*Other*], [*Source*],
  [Backprop, 1M], [0.175], [0.98], [token embedding 1000], [released baseline],
  [Backprop, 10M], [0.2], [0.95], [token embedding 1000], [released baseline],
  [DUST, 256 draws, 1M], [0.25], [0.95], [value embeddings 0.3; scalars 0.03; token embedding 1000], [released],
  [DUST, 1,024 draws, 1M], [0.225], [0.95], [as above], [released],
  [EGGROLL, Adam], [0.003], [$beta_1 = 0.5$], [$beta_2 = 0.95$, rank 1, $sigma = 0.003$], [`local-learning`; tuned in #raw("exp5"), the paper's Adam constants from #raw("exp8")],
)

SGD is PyTorch's: a momentum buffer $b arrow.l mu b + g$, then $theta arrow.l theta - "lr" dot b$. The
token embedding's rate of 1000 is not a typo: its gradient is tiny, because the embedding is
normalised immediately after lookup.

=== Adam and Muon <method-opt>

The last results subsection (@res-opt) repeats the 1M-token ladder under two other optimisers, with the groups and constants of the DUST
paper's Table 7 where it gives them. Adam is the usual one: bias-corrected, $beta_2 = 0.95$, $epsilon = 10^(-8)$, no weight decay.

#table(
  columns: (1.3fr, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  [*Method*], [*Matrices, head*], [*Embeddings*], [*Residual scalars*], [*$beta_1$*],
  [Backprop], [searched], [0.3], [0.2], [0.8],
  [DUST], [searched], [300 × matrix rate], [100 × matrix rate], [0.7],
  [EGGROLL], [0.003], [0.003], [0.003], [0.5],
)

Muon acts on the six hidden matrices of a block (queries, keys, values, the attention output projection, and the MLP in and out) and
leaves everything else on Adam at the rates above. It keeps a Nesterov momentum buffer ($mu = 0.95$), replaces it by the matrix with the same
singular vectors and singular values near one, with five steps of the quintic iteration $X arrow.l a X + (b A + c A^2) X$,
$A = X X^top$, $(a, b, c) = (3.4445, -4.7750, 2.0315)$, and steps by $"lr" dot 0.2 sqrt(max(m, n))$ times the result for an $m times n$ matrix.
The factor makes the per-entry size of the step about 0.2, which is AdamW's, so Muon's rate is read on the same scale as Adam's. EGGROLL under Muon
keeps 0.003 for everything but the matrices, whose rate is searched.

Under Adam the DUST paper also moves draws between sites. At population $K$ it gives the residual writers $0.25 K$ draws, the MLP first layer
$0.25 K$, the attention outputs $0.375 K$, the token embedding $0.5 K$ at noise 0.4, and the head $3 K$, where the SGD recipe gives them 0.43 $K$,
0.43 $K$, 0.10 $K$, 0.04 $K$ and $4 K$. Its split across blocks is not given. Ours keeps the SGD recipe's tilt toward early blocks, rescaled, so an
Adam or Muon update draws $1.375 K$ for the first four sites where the SGD one draws $K$. We used this recipe under Muon as well.

== Protocol

*Seeds.* The seed sets both the initialisation and the order of the training sequences. The 1M-token ladder and the
100K-token ladder use seeds 42, 43 and 44; the 10M-token cells and the 4,096-population rung use seed 42 only, except
backprop at 10M, which has all three. Every number in a table is a mean over the seeds stated beside it, with the sample
standard deviation; a single run is marked. Tuning uses seed 42 alone and judges by validation loss.

*Tuning.* Backprop and DUST use the authors' released settings at 1M and 10M tokens, except DUST at 64 draws and every cell
at 100K, which follow the paper's rule (Appendix F of @dahal2026dust): a 12-point grid of momentum and rate, seed 42, the
best point repeated on three seeds, the grid extended one step when its best point lies on an edge. EGGROLL's Adam rate and
noise scale are tuned the same way, on the grids in the experiments table; its 1,024-pass and 4,096-pass cells take the
256-pass winner, and its 10M-token cell takes the 1M-token winner.

*Hardware and cost.* Every run uses one RTX 4090. The runs of @res-opt sometimes shared the machine with another user's jobs, which slows them and does not change their losses (@sec-limits); the timings below are from runs that did not. JAX 0.8.1; matmuls in bfloat16 with float32
accumulation, float32 master weights. Cost is the median wall-clock of an update after the first, which carries
XLA compilation (about five minutes for a DUST update, once per batch shape, then cached). A DUST update takes
2.2, 7.6, 29.5 and 116.5 seconds at 64, 256, 1,024 and 4,096 draws, and an EGGROLL update 2.1, 8.2, 32.4 and 129.5 seconds at the
same populations; a backprop update takes 0.13 seconds. The model has 37,749,776 parameters.

*Checkpoints for the alignment measurement.* Backprop at the authors' 10M-token settings (rate 0.2, momentum 0.95) on a
100M-token stream disjoint from validation and test, 6,100 updates, weights kept after 610 and after 6,100. The paper
does not say how its own 100M-token checkpoint was trained. Alignment is measured on the first eight validation
sequences, with two independent noise draws at populations up to 1,024 and one above.

== Notation

#notation(
  ($theta$, [the model's parameters, a vector of $d = 37749776$ numbers]),
  ($L$, [mean per-token loss over a batch]),
  ($ell_t$, [loss at token $t$]),
  ($sigma$, [noise scale]),
  ($epsilon$, [a weight-space perturbation (EGGROLL)]),
  ($a_t$, [activation noise at token $t$ (DUST), standard normal]),
  ($K$, [DUST population: draws rewarded with token losses]),
  ($P$, [EGGROLL population: forward passes of the batch]),
  ($hat(g)_t$, [estimated gradient of the loss with respect to a layer's output at token $t$]),
  ($gamma$, [credit decay for attention internals]),
  ($d$, [number of parameters]),
)
