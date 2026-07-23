# Saturation in Small Language Models

Based on arxiv:2602.03685 — *"Scaling Laws for Precision"* / logit saturation analysis.

## Core idea

As a language model trains, its output logits grow in magnitude. The **inverse temperature** β measures how peaked the output distribution is:

```
β = std(logits over vocab)   [per token position, then averaged over the batch]
```

When β is small, the distribution is diffuse (uniform-ish). When β is large, the model is confident — most probability mass sits on a few tokens.

The paper identifies a critical threshold:

```
c₀ = sqrt(2 · ln V)
```

where V is the vocabulary size. For V = 4096: **c₀ ≈ 4.08**.

- **β < c₀** — sub-saturated regime. The distribution is still diffuse. Loss can decrease quickly.
- **β > c₀** — saturated / peaked regime. The model is overconfident. Loss decays much more slowly, following a power law: `loss ∝ τ^(−1/3)` where τ = cumulative learning rate = `batch × lr`.

## What we track

Each batch logs:

| Field | Meaning |
|-------|---------|
| `beta` | Mean β over response token positions in the batch |
| `tau` | Cumulative LR = `batch_idx × ADAM_LR` |
| `pl_exponent` | Rolling power-law exponent fit: slope of `log(loss)` vs `log(τ)` over the last 200 batches |

The theoretical value once saturated: `pl_exponent ≈ −0.333`.

## What we observed (exp-10, 2414 batches, random init)

β never exceeded c₀:

| Phase | Mean β |
|-------|--------|
| Early (1–500) | 0.851 |
| Mid (501–1500) | 1.225 |
| Late (1501–2414) | 1.422 |

This makes sense — the model is randomly initialized and small. It may take much longer training or a warmer LR to push β past c₀. The saturation regime is more relevant for large models or fine-tuning from a pre-trained checkpoint.

The `pl_exponent` was positive throughout (~1.96 at batch 2414), meaning loss was still actively decreasing faster than the τ^(−1/3) law predicts. We are clearly pre-saturation.

## Implications for this project

- With random init, don't expect to see β > c₀ in 3000 batches at lr=1e-4.
- Starting from a pre-trained checkpoint (e.g. `params_exp_kdyck_rope_phase1.pkl`) would give a higher starting β and might cross c₀ earlier.
- The saturation plot at the end of each run shows β and pl_exponent over training — the goal is to watch β grow and eventually cross the red dashed line.
- Once pl_exponent stabilises near −0.333, the model is in the scaling-law regime and further improvement requires more data or a larger model, not just more steps.
