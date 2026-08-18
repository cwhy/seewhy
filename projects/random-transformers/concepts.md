# Random Transformers — Concepts

Algorithmic details for [workflow.md](workflow.md). Replication of
arXiv:2410.04368; motivation and claim list are in [proposal.md](proposal.md).

## The central object

A **random transformer** is a decoder-only transformer in which the attention
and feed-forward weights are sampled once at initialisation and never updated.
Only three matrices are trained:

| matrix | shape | role |
|--------|-------|------|
| `E_token` | (v, d) | one vector per vocabulary item |
| `E_pos`   | (n, d) | one vector per position |
| `U`       | (v, d) | maps the final hidden state to logits over the vocabulary |

Everything between them — `12d² + 13d` parameters per layer — stays at its
random initialisation. Training can choose *how inputs are written into* the
random stack and *how its output is read back*, and nothing else.

## Model

GPT-2 architecture, pre-layernorm, no dropout, no weight tying, GeLU (tanh
approximation). One block:

```
h ← h + Attn(LN(h))
h ← h + MLP(LN(h))
```

with causal attention, `n_head = 8`, MLP hidden width `4d`. Final `LN`, then
`logits = LN(h) @ U.T`.

**Initialisation** (from the paper, §3): MLP weights `~ N(0, (0.02/√(2m))²)`
for `m` layers; every other weight matrix `~ N(0, 0.02²)`, including query,
key, value, the attention output projection, and all three embedding matrices;
biases zero; layer-norm affines identity.

The attention output projection is not named in the paper's initialisation
paragraph. We read it as an "other weight matrix" (flat 0.02), not as a
feed-forward layer. This is recorded as ours.

## Training modes

The one axis the experiments vary. `TRAINABLE` in `lib/model.py`:

| mode | optimised | paper's name |
|------|-----------|--------------|
| `full` | everything | "normal transformer" |
| `random` | `E_token`, `E_pos`, `U` | "random transformer" |
| `u_only` | `U` | Table 2 ablation |
| `e_only` | `E_token`, `E_pos` | Table 2 ablation |
| `etoken_u` | `E_token`, `U` | Table 2 ablation |

Frozen parameters are passed to the loss as a non-differentiated argument, so
XLA never builds their weight-gradient matmuls. Gradients still flow *through*
them to reach the embeddings — that is the whole mechanism.

## Loss

Next-token cross-entropy, restricted to scored positions:

```
L = − (1/|M|) Σ_{(b,t) ∈ M} log p(x_{b,t+1} | x_{b,≤t})
```

`M` is the mask each task supplies. Prompt tokens and padding are excluded, so
loss and accuracy both see only the answer. AdamW, learning rate 1e-3, weight
decay 1e-3, global gradient-norm clipping at 1.0 (paper §D.3). Language
modeling uses 6e-4 / 0.1.

## Tasks

All encodings follow Appendix D.1 and are checked against independently written
decoders by `scripts/tmp/verify_tasks.py` before any run.

| task | vocab | `n_ctx` | sequence | scored |
|------|-------|---------|----------|--------|
| `mod_add` | 199 | 5 | `[a, b, (a+b) mod 199]` | the answer |
| `needle` | 256 | 100 | `[m₁,c₁,…,m_k,c_k,m_u, c_u]`, k~U[1,30] | the answer |
| `decimal` | 31 | 40 | `[a₀…a₉, +, b₀…b₉, =, r₀…r_l, EOS]`, digits reversed | every output token |
| `parens` | 4 | 80 | `[p₁…p_n, ?, label]`, n ≤ 60 | the label |
| `memorization` | 1024 | 5 | `[x, y+512, z]` | the answer |

Markers are distinct integers in [128, 157]; values in [1, 127]; the query
token is the asked marker + 30. Decimal input digits are tokens 0–9, `+` is 10,
`=` is 11, output digits are 20–29 and 30 ends the output. Parentheses are 1
and 2, `?` is 3, and the label reuses tokens 2 (balanced) and 1 (unbalanced).

**Splits.** Modular addition uses a fixed 95/5 split of all 199² = 39,601 pairs
(37,621 train / 1,980 test), identical across every run and architecture, so
test accuracy measures generalisation. The other tasks stream fresh training
data and hold out a fixed 4,096-example test set. Memorization has no test
split by design.

## Metrics

**Sequence accuracy** — every scored token in a sequence is correct. This is the
strict metric and is what the tables report. **Token accuracy** is logged
alongside it, and is the more informative one for decimal addition, where a
single wrong digit zeroes the sequence score.

**Chance levels**, derived rather than asserted:

- `mod_add`: 199 equally likely answers → **1/199 = 0.503%**.
- `needle`: the answer is uniform in [1, 127] and the query is uninformative to
  a model that cannot search → **1/127 = 0.787%**.
- `decimal`: the strict metric needs 10 or 11 correct digits at once. A uniform
  digit guesser scores 10⁻¹⁰; the honest floor is **0%**.
- `parens`: guessing the majority class. Measured on the fixed test set:
  30.6% of sequences are balanced, so the floor is **69.4%** — this is why the
  paper's width-16 figure of 87.3% is a much weaker result than it looks.
- `memorization`: 512 equally likely values → **1/512 = 0.195%**.

**Bits memorized** (H4), following the paper: an example counts as memorized if
its argmax output is correct, and each memorized example is worth
`log₂ 512 = 9` bits. Bits per parameter divides by the *trainable* count, which
is what differs between the two conditions.

**Explained variance** (H6): collect hidden states at the embedding output and
after each block, over a fixed set of inputs. Report the fraction of total
variance captured by (a) the top 10 principal components and (b) the 10
highest-variance individual neurons. Subspace selection means (a) is large;
sparsification would mean (b) is large too. The contrast between the two is the
claim — neither number means anything alone.

## Findings

_(appended as results land)_
