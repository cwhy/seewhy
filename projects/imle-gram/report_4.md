# Report 4 — Tensor Product Decoder (exp6)

## Goal

Test whether replacing the gram decoder with a 3-way tensor product `r⊗c⊗h`
changes the 87% ceiling established in exp1–5.

## Architecture

```
z (LATENT_DIM=64)
→ h = GELU(z @ W_mlp1)              # (MLP_HIDDEN=64,)  shared across pixels

per pixel (r, c):
→ re = row_emb[r]                   # (D_R=6,)
→ ce = col_emb[c]                   # (D_C=6,)
→ f  = (re[:,None,None] * ce[None,:,None] * h[None,None,:]).reshape(-1)
                                     # (D_R*D_C*MLP_HIDDEN,) = (2304,)
→ h1 = GELU(W_val1 @ f)            # (MLP_HIDDEN=64,)
→ logits = W_val2 @ h1              # (N_VAL_BINS=32,)
```

**No gram. No concatenation. Multiplicative interaction between position and latent.**

| Hyperparameter | Value |
|----------------|-------|
| `LATENT_DIM` | 64 |
| `MLP_HIDDEN` | 64 |
| `D_R`, `D_C` | 6, 6 |
| `f_dim` | 6×6×64 = 2304 |
| `N_SAMPLES` | 1000 |
| Parameters | **154,096** |

## Results

| Epoch | Match L2 (train) | Eval Match L2 | Bin acc |
|-------|-----------------|---------------|---------|
| 4 | 4,415 | 4,012 | 82.6% |
| 9 | 3,440 | 2,933 | 84.8% |
| 14 | 2,674 | 2,352 | 85.9% |
| 19 | 2,405 | 2,121 | 86.4% |
| 29 | 2,243 | 1,967 | 86.6% |
| 44 | 2,164 | 1,885 | 86.8% |
| 64 | 2,128 | 1,849 | 86.9% |
| 99 | 2,092 | 1,821 | **86.9%** |

**Final: 86.9% bin accuracy, 490s, 154K params.**

## Comparison

| Exp | Architecture | Params | Bin acc | Time |
|-----|-------------|--------|---------|------|
| exp1 | gram decoder | 1.25M | 87.1% | 330s |
| exp2 | gram, N_SAMPLES=5000 | 1.25M | 87.2% | 550s |
| exp4 | gram, MLP=512, rank=32 | 4.7M | 87.1% | 323s |
| exp5 | gram, adaptive IMLE | 1.25M | 87.0% | 760s |
| **exp6** | **r⊗c⊗h tensor product** | **154K** | **86.9%** | **490s** |

## Observations

**Same ceiling, 8× fewer parameters.** The tensor product decoder reaches 86.9% —
essentially identical to the gram decoder ceiling — using only 154K parameters vs 1.25M.
The gram structure was adding complexity with zero accuracy benefit.

**The ceiling is robustly at 87%.** Six experiments across two fundamentally
different decoder architectures (gram rank-16 vs 3-way tensor product), varying
N_SAMPLES (1000–5000), LATENT_DIM (64–256), and matching strategies all land at
87.0–87.2%. The constraint is in the IMLE matching mechanism, not the decoder.

**Tensor product is parameter-efficient.** The 3-way interaction `r⊗c⊗h` captures
position×latent interactions multiplicatively — each logit depends on *how* the
latent code relates to the specific row and column, not just their concatenation.
This inductive bias achieves the same result at 8× lower parameter cost.

**Large param budget remaining.** With 154K params vs 1.25M gram baseline, there
is room to scale up the tensor product decoder significantly within the same budget.

## Visualizations

- Learning curves: https://media.tanh.xyz/seewhy/26-03-23/exp6_learning_curves.svg
- Recon epoch 20: https://media.tanh.xyz/seewhy/26-03-23/exp6_recon_epoch020.png
- Recon epoch 60: https://media.tanh.xyz/seewhy/26-03-23/exp6_recon_epoch060.png
- Recon final: https://media.tanh.xyz/seewhy/26-03-23/exp6_recon_epoch100.png
- Prior samples: https://media.tanh.xyz/seewhy/26-03-23/exp6_samples_5x8_seed0.png

## Next

Scale up the tensor product decoder within the 1.25M param budget:
- exp7: MLP_HIDDEN=128 → f_dim=4608, W_val1=(128,4608)=590K → ~600K total params.
