# Report 4 — Gram VAE Experiments 10–14

## Summary

Experiments 10–14 systematically probed the remaining capacity limits after hitting the 88.7%
plateau in exps 6–9. The key finding: **LATENT_DIM is the dominant lever** — scaling from 64→512
gave the clearest and most consistent gains. The current best is **89.2%** (exps 11, 13, 14).
We appear to be approaching a ceiling where neither encoder nor decoder improvements move
things further.

---

## Results Table

| Exp | Acc    | Recon NLL | KL     | Time | Params   | Key change from prev best |
|-----|--------|-----------|--------|------|----------|---------------------------|
| 3   | 83.9%  | 0.7241    | 0.5365 |  94s |  168K    | baseline (direct val head) |
| 4   | 87.3%  | 0.6036    | 0.5082 | 136s |   3.2M   | D=125, decoder MLP, N_ENC_Q=8 |
| 5   | 88.7%  | 0.5354    | 0.5121 | 262s |   5.0M   | D=144, 2-layer val head |
| 6   | 88.6%  | 0.5424    | 0.3141 | 257s |   5.6M   | enc MLP, LATENT=128, FREE=0.3 ← FAILED |
| 7   | 88.6%  | 0.5438    | 0.5107 | 258s |   5.0M   | z-conditioned queries |
| 8   | 88.7%  | 0.5151    | 0.5103 | 666s |   7.3M   | 300 epochs, DEC_RANK=48 |
| 9   | 88.7%  | 0.5345    | 0.5115 | 386s |   5.1M   | 4-head decoder |
| 10  | 89.0%  | 0.4648    | 0.5115 | 255s |   5.5M   | LATENT_DIM=256 |
| 11  | **89.2%**  | 0.4296    | 0.5094 | 255s |   6.2M   | LATENT_DIM=512 |
| 12  | 89.1%  | 0.4508    | 0.5078 | 260s |   6.6M   | enc MLP (hidden=1024) + LATENT=256 |
| 13  | 89.2%  | 0.4198    | 0.5111 | 290s |  20.6M   | DEC_RANK=64, MLP_HIDDEN=1024 + LATENT=512 |
| 14  | **89.2%**  | 0.4249    | 0.5092 | 335s |   8.4M   | D=196 (D_R=D_C=7), LATENT=512 |

---

## Key Findings

### 1. Latent dimension is the strongest lever

The 88.7% plateau (exps 6–9) was purely an information bottleneck. With `FREE_BITS=0.5`, each
latent dim encodes a minimum of 0.5 nats. With `LATENT_DIM=64`, total capacity = 32 nats — not
enough for MNIST images at 32-bin resolution.

| LATENT_DIM | Min. capacity | Bin accuracy |
|------------|---------------|--------------|
| 64         | 32 nats       | 88.7%        |
| 256        | 128 nats      | 89.0%        |
| 512        | 256 nats      | 89.2%        |

Gains are real but diminishing: +0.3% per 4× increase. The KL per dim stays pinned at the floor
(~0.509–0.512) regardless of LATENT_DIM — all dimensions are saturated. This suggests the gram
encoder's representational capacity (not the latent size) is now the limiting factor.

### 2. FREE_BITS must stay at 0.5

Exp6 dropped FREE_BITS to 0.3 while increasing LATENT_DIM to 128. Result: partial posterior
collapse — most dims settled at KL≈0.3 (the new floor), effectively unused. Accuracy went from
88.7% → 88.6%. With FREE_BITS=0.5, all dims are forced to encode signal; cutting it even slightly
lets them coast at the lower floor.

**Rule**: never reduce FREE_BITS below 0.5 in this architecture.

### 3. Decoder rank is not a bottleneck

Exp13 doubled DEC_RANK (32→64) and widened MLP_HIDDEN (512→1024), ballooning params from 6.2M
to 20.6M. Result: same 89.2%. The pseudo-gram rank-32 is already sufficient for the decoder.
Future experiments should not pursue DEC_RANK > 32 — it's wasted compute.

### 4. Encoder MLP helps modestly but LATENT_DIM matters more

Exp12 added a non-linear MLP layer in the encoder (gram features → hidden=1024 → mu/lv) with
LATENT_DIM=256. Result: 89.1% vs exp10's 89.0%. A modest improvement, but exp11 (LATENT_DIM=512,
no enc MLP) scored 89.2%. The larger latent outweighs the non-linearity.

### 5. Larger D (gram dimension) marginally helps

Exp14 increased D from 144 to 196 (D_R=D_C=7) with LATENT_DIM=512. Result: 89.2% — matched
exp11 and possibly a hair better (0.8921 vs 0.8919). The larger gram gives slightly richer
per-pixel tokens. However, it also increased params from 6.2M to 8.4M and training time from
255s to 335s for the same accuracy. Not a meaningful gain.

---

## Architecture Progression

The winning sequence of improvements, from exp3 → exp14 best:

```
exp3  83.9%  D=64,  LATENT=64,  direct val head
        ↓ +3.4%  D=125, decoder MLP, N_ENC_Q=8
exp4  87.3%
        ↓ +1.4%  D=144, 2-layer val head
exp5  88.7%
        ↓ +0.3%  LATENT_DIM: 64 → 256
exp10 89.0%
        ↓ +0.2%  LATENT_DIM: 256 → 512
exp11 89.2%  ← current best (tied exp14)
```

---

## What Doesn't Work

All tested against exp5 (88.7%) or exp11 (89.2%) baseline:

- **More epochs** (exp8: 300 epochs): same accuracy at 3× cost
- **Z-conditioned position queries** (exp7): no improvement
- **Multi-head decoder** (exp9, 4 heads): no improvement
- **Higher DEC_RANK** (exp13: rank 64 vs 32): same accuracy at 3.3× params
- **Larger gram D alone** (exp14 vs exp11): negligible
- **Enc MLP alone** (exp12): +0.1% over exp10, worse than just scaling LATENT_DIM

---

## Diagnosis of the Current Ceiling

All latent dims (whether 256 or 512) saturate at exactly the FREE_BITS floor (KL ≈ 0.51).
This means the encoder is compressing images into the minimum allowed information — the
reconstruction gradient is not strong enough to push dims above the floor. Two interpretations:

**A. Task ceiling**: 32-bin quantization means adjacent bins (8 pixel values apart) are genuinely
ambiguous. Some pixel accuracy is irreducible, and we're near that limit. The VAE naturally
settles at the minimum KL needed to reach reconstruction quality sufficient for bin accuracy ≈ 89%.

**B. Encoder bottleneck**: The gram matrix M = Σ token_i ⊗ token_i is a permutation-invariant
second-order statistic. It loses pixel ordering except through position-tagged tokens. The gram
may not capture enough spatial structure for finer reconstruction, regardless of latent size.

Both likely contribute. The consistently flat KL across LATENT_DIM 256 and 512 (both at 0.51)
points toward the gram encoder's capacity or the task itself as the true ceiling.

---

## Next Directions (if continuing)

**Most promising**:
- More encoder queries: N_ENC_Q=16 or N_ENC_Q=32. Currently 8 query vectors summarize the gram;
  more queries might extract more of its information.
- Combine best factors: D=196 + LATENT_DIM=512 + enc MLP. Each individually is marginal, but
  together might push past 89.2%.
- More epochs with the best config (exp11/14): loss was still slowly declining at epoch 99.

**Probably not worth it**:
- Higher DEC_RANK (confirmed no-op)
- Larger D alone without a corresponding latent increase
- Any change that reduces FREE_BITS

---

## Visualizations

All plots at `seewhy/26-03-23/` on R2:

- exp11 reconstructions: https://media.tanh.xyz/seewhy/26-03-23/exp11_reconstructions_final.png
- exp14 reconstructions: https://media.tanh.xyz/seewhy/26-03-23/exp14_reconstructions_final.png
- exp14 learning curves: https://media.tanh.xyz/seewhy/26-03-23/exp14_learning_curves.svg
- exp14 latent PCA:     https://media.tanh.xyz/seewhy/26-03-23/exp14_latent_pca_final.svg
- exp14 entropy map:    https://media.tanh.xyz/seewhy/26-03-23/exp14_entropy_final.png
