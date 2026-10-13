# DUST vs EGGROLL — Concepts

What this project tests: the claim of [Dahal et al., "Dust: Pretraining Transformers
Without Backpropagation"](https://qlabs.sh/research/dust) (Q Labs, October 2026) that a
zeroth-order method which perturbs *activations* trains a transformer language model far
more efficiently than one which perturbs *weights* — "on the order of 10³ to 10⁴ times
more efficient than EGGROLL". We re-implement DUST in JAX from the authors' released
PyTorch code, implement EGGROLL on the same network with the setting used in
[local-learning](../local-learning/concepts.md), and run the paper's comparison at the
scale two RTX 4090s allow.

## Task / data

Next-token prediction on FineWeb (`HuggingFaceFW/fineweb`, `sample-10BT`, the first
200,000 documents skipped), tokenised with the authors' byte-level BPE-4096
(`tokenizer/tokenizer.json`, MIT). Sequences are 2048 tokens plus one shifted target.

Training is **one epoch over a fixed prefix** of the stream, 8 sequences (16,384 tokens)
per update:

| budget | sequences | updates |
|---|---:|---:|
| 100K tokens | 48 | 6 |
| 1M | 488 | 61 |
| 10M | 4,880 | 610 |

Validation and test are 544 sequences each, disjoint from training and from each other.
**The reported number is the test loss at the checkpoint with the best validation loss.**
Chance is `ln 4096 = 8.318` nats per token. A model that knows only unigram frequencies
(counted on the 1M-token training prefix) scores **6.994** on the test split (7.027 from
the 100K prefix, 6.978 from 10M); `scripts/tmp/unigram.py`. That is the floor a network
must get under before it has learned anything about context.

## Model & loss

A 37,749,776-parameter decoder-only transformer of the modded-nanoGPT kind (8 blocks,
width 512, 8 heads of 64). The details that matter because the paper trains this exact
network:

- pre-norm, RMS norm with no gain, no biases; squared-ReLU MLP of width 2048;
- rotary embeddings, then RMS-normalised queries and keys;
- causal attention with a 1024-token left window in blocks 0–2 and 4–6, full 2048 in 3 and 7;
- a **value embedding** on every other block: a second token embedding added to the values,
  scaled per head by `2·sigmoid(gate)`, the gate reading 32 channels of the block input;
- the stream entering block `i` is `resid[i]·x + x0[i]·x₀` — 2L learned scalars;
- untied head, logits capped as `15·tanh(logit/15)`;
- loss is mean per-token cross-entropy. Compute runs in bfloat16; weights are kept in f32.

## The three learning rules

### Backprop

`jax.grad` of the mean loss, then SGD with momentum. One global backward pass. The
authors' tuned settings: lr 0.175, momentum 0.98 at 1M tokens (0.2 / 0.95 at 10M); the
token embedding gets lr 1000 (its gradient is tiny because the embedding is immediately
RMS-normalised).

### EGGROLL — perturb the weights (the baseline)

[Evolution strategies at the hyperscale](https://arxiv.org/abs/2511.16652) (Sarkar et
al.). A **population member is a perturbed copy of the whole network**. For every weight
matrix `W` (`m × n`) draw `a ~ N(0, I_m)`, `b ~ N(0, I_n)` and set

    ε = σ · a bᵀ                       rank one, never stored: the forward pass uses
                                       x Wᵀ + σ (x·b) aᵀ, a side branch of two thin products

Each member is scored on the whole batch, as an **antithetic pair**,

    f = L(θ + ε) − L(θ − ε)           two forward passes, one scalar

and the gradient estimate is the fitness-weighted sum of the perturbations,

    ĝ = 1/(P σ²) · Σᵢ fᵢ εᵢ            P pairs;   E[ĝ] = 2 ∇L

fed to Adam. This is the setting of `local-learning/lib/es.py`, unchanged: rank 1,
antithetic, raw loss differences (no rank shaping, no z-scoring), Adam. Vectors get a
dense Gaussian. **Population is counted in forward passes of the batch** (2 per pair),
which is how the DUST paper counts it.

*Why it is hopeless at scale.* One member returns **one number**. That number must
steer ~3.8×10⁷ parameters. The cosine between ĝ and the true gradient grows as
`√(P/d)`: with `d = 3.8×10⁷` and 16,384 forward passes that is **0.02**. Low rank makes a
member cheap to *store*; it does nothing for how little a member *tells* you.

### DUST — perturb the activations, one member per token

Add noise to the **output of a linear layer** instead of to its weights:

    y_t → y_t + σ a_t,        a_t ~ N(0, I),   independently at every token t

This is *node perturbation* (Widrow & Lehr; Werfel et al.). A layer's output has `d_out`
numbers, its weights `d_out × d_in`, so the noise space is `d_in` times smaller. And the
transformer has a trick a plain MLP does not: a batch is `8 × 2048` tokens, each with its
**own per-token loss** `ℓ_t`. If token `t`'s loss responds mainly to token `t`'s own
noise, then **one forward pass runs ~16,000 independent experiments at once**. That is
the *virtual population*: it lives on the token axis, orthogonal to the batch axis that
EGGROLL already uses.

The estimate for one set of layers, with `K` draws (each draw is one forward pass with
fresh noise at every token):

1. **Reward.** `rₜ = mean_draws(ℓₜ) − ℓₜ` — how much *lower* than average token `t`'s loss
   was in this draw. (Centred across the draws evaluated together; no rescaling.)
2. **Estimated error at the layer's output.**
   `ĝₜ = −(1/(Kσ)) · Σ_draws rₜ aₜ` — reward-weighted noise. A jitter that lowered the
   loss is evidence that the gradient points the other way.
3. **Weight gradient.** `Ĝ_W = Σₜ ĝₜ xₜᵀ` — the estimated output error times the layer's
   *cached* input, summed over tokens. **Backprop forms the same outer product with the
   same input; it gets `ĝ` from the chain rule, DUST gets it from the population.** For
   the embedding, `xₜ` is one-hot, so this is a scatter-add into the token's row.

Everything above happens without a backward pass and without differentiating anything.
Three details make it work for a real transformer:

**Interference.** If you jitter every layer at every token in one pass, the loss change
that is meant to reward one jitter also contains the effect of all the others. DUST
spends extra passes to keep them apart: the two residual writers of a block are jittered
together but each block gets its own passes; the MLP's first layer, the attention
output, and the token embedding each get theirs, with their own noise scale. Passes are
cheaper than a full forward because the clean pass is cached and a draw that jitters
block `l` reruns only blocks `l…L`.

**Attention internals.** Queries, keys, values, the value-embedding gate and the value
embedding act on *other tokens' outputs* through attention, so a token's own loss barely
registers them. They are scored **against the estimated error at the attention output**
(from the previous step, `ĝ`): recompute only the block's attention with the jitter, and
take `cₛ = ⟨ĝₛ, Δoₛ⟩` per head, the predicted loss change. Token `t`'s jitter is rewarded
with the scores of the tokens that can see it,

    rₜ = Σ_{s ≥ t, s−t ≤ window} γ^(s−t) · cₛ ,      γ = 0.98   (queries: γ = 0)

**The head and the scalars.** The head is jittered directly on the cached logits, one
256-wide slice of the vocabulary per draw, recomputing only the cross-entropy — a small
fraction of a forward pass. The 2L residual-mixing scalars are not activations, so they
use ordinary two-sided weight-space ES (8 draws).

*Population* `K` counts the **direct draws** (writers, MLP hidden, attention outputs,
token embedding). Head draws and attention-internal draws are extra, as in the paper.
The paper's tuned allocation across layers, noise scales (0.2 / 0.4 / 0.05) and
optimiser settings are used as released — see `lib/dust.py`.

### What the comparison is

| | EGGROLL | DUST |
|---|---|---|
| noise lives in | weight space (rank-1 per matrix) | activation space (per token) |
| one forward pass is | **1** population member | **~16,000** members (one per token) |
| member's reward | the batch-mean loss: 1 number | its own token's loss: 1 number *per token* |
| gradient is built as | `Σ f ε` over members | `Σₜ ĝₜ xₜᵀ` — exact local outer product |
| optimiser | Adam | SGD + momentum |
| population counted in | forward passes of the batch | draws (each also a pass, a little cheaper) |

## Metrics

- **test loss** — mean per-token cross-entropy on the 544 test sequences, at the
  checkpoint with the lowest validation loss. Lower is better; the paper's primary number.
- **cosine to backprop** — cosine between an estimate and the backprop gradient *on the same
  batch and weights*, per layer type (mean over blocks). 1.0 means the estimator
  reproduced backprop. The paper's law: `cos(K) = c_max / √(1 + c/K)`.
- **population** — forward passes of the batch per update. For DUST, direct draws `K`;
  for EGGROLL, 2 per antithetic pair.
- **step_s** — median wall-clock of one update after compilation, blocked on the device.
  The paper does not report compute; we do, because the headline claim is *efficiency*.

## Findings

Test loss at the best-validation checkpoint, 1M tokens, three seeds unless noted. Reports 1-5 and the paper
(`paper/`, published at a stable URL) carry the evidence.

1. **The port reproduces the published baselines.** Backprop 5.937 ± 0.027 (published 5.959) and 4.984 ± 0.008 at 10M (4.989); DUST
   6.315 / 6.074 / 5.939 at 64 / 256 / 1,024 draws (6.290 / 6.067 / 5.934). On identical weights the authors' PyTorch DUST and ours give
   the same layer cosines and total gradient norm.
2. **DUST beats EGGROLL at every population.** 64 DUST draws (6.315) beat 4,096 EGGROLL passes (6.970) by 0.65 nats at 59x less
   measured cost. EGGROLL (ours, `local-learning`'s setting) is above the token-frequency floor of 6.994 until 4,096 passes.
3. **DUST reaches backprop, under SGD.** Level at 1,024 draws (5.939 vs 5.937), 0.047 below at 4,096 on one seed (paper: 0.048 below).
   Under Adam and Muon this does not hold (finding 8).
4. **The cause is measured.** Cosine to the backprop gradient at 16,384 draws on a 100M-token checkpoint: DUST 0.99 head, 0.73 attention
   out, 0.63 MLP; EGGROLL 0.024, 0.010, 0.011. EGGROLL's values equal `s / sqrt(s^2 + d/n)`, the random-direction prediction, to within
   0.002, and at 131,072 passes are under 0.05 except the head (0.064), as the paper reports.
5. **The efficiency factor is an extrapolation and the ladder bends.** Matching DUST's 64 draws takes EGGROLL 3,554x (line through the
   top three) to 16,810x (top two) the population, or never by a power law whose floor (6.55) is above 6.315. A three-point line gave
   547x. Treat any such factor as an order of magnitude.
6. **Our EGGROLL is weaker than the paper's** by 0.34 nats at 64 passes and 0.08 at 4,096. Adam rate, sigma, the paper's Adam constants,
   rank shaping and float32 did not close it; its code is not released.
7. **Other budgets.** 10M tokens, one seed: DUST 256 draws 5.232 (paper 5.231), 0.248 behind backprop. 100K tokens (six updates): nothing
   beats the unigram floor of 7.027, ours or the paper's.
8. **Better optimisers help backprop more than DUST, and EGGROLL not at all.** 1M tokens. Backprop 5.937 (SGD) -> 5.358 (Adam, paper 5.361)
   / 5.372 (Muon). DUST 64 / 256 / 1,024 draws: Adam 6.036 / 5.860 / 5.836, Muon 5.864 / 5.617 / 5.466 (SGD 6.315 / 6.074 / 5.939), so DUST is
   +0.493 / +0.245 / +0.095 behind backprop under Muon, +0.036 at 4,096 draws (one seed; SGD: 0.047 ahead). The distance more than halves
   per quadrupling and has not reached zero. EGGROLL under Muon is slightly worse than under Adam (+0.052, +0.029, +0.015). The paper's
   Adam section reports a fitted DUST limit of 5.248 below its backprop's 5.361, not a measured crossing. Caveats: Adam cells carry ~0.08 nats
   of GPU nondeterminism, 1,024 and 4,096 are one seed at the 256-draw rate (a check at 1,024 found 0.003 best), the Adam draw split is our
   reconstruction. Code: `lib/optim.py`, `experiments19.py` to `experiments28.py`, `scripts/gpu_queue.py` (runs jobs on idle GPUs only).
