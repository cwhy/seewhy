# Report — exp3: Direct Val Head + More Capacity

## Config

| Param | Value |
|---|---|
| D_R, D_C, D_V | 4, 4, 4 |
| D (gram space) | 64 |
| N_ENC_Q | 4 |
| LATENT_DIM | 64 |
| DEC_RANK | 16 |
| N_VAL_BINS | 32 |
| FREE_BITS | 0.5 |
| ADAM_LR | 3e-4 (cosine decay) |
| KL_WEIGHT | 1.0 |
| KL_WARMUP_EPOCHS | 20 |
| BATCH_SIZE | 128 |
| MAX_EPOCHS | 100 |
| n_params | 168,384 |
| time | 93.8s |

**Key change from exp2**: decoder val head changed from `dec_val_emb @ mean_pool(f)` (D=64 → D_V=4 → 32 bins) to `W_val_head @ f` (D=64 → 32 bins directly), eliminating the pooling bottleneck.

## Results

| Epoch | β | recon NLL | KL | total | bin acc |
|---|---|---|---|---|---|
| 0 | 0.00 | 4.618 | 1.647 | 4.618 | — |
| 4 | 0.20 | 0.849 | 0.625 | 0.973 | 81.3% |
| 20 | 1.00 | 0.808 | 0.535 | 1.343 | — |
| 39 | 1.00 | 0.779 | 0.534 | 1.312 | 82.1% |
| 59 | 1.00 | 0.755 | 0.535 | 1.290 | 82.9% |
| 79 | 1.00 | 0.730 | 0.536 | 1.266 | 83.8% |
| **99** | 1.00 | **0.724** | **0.537** | **1.261** | **83.9%** |

## Visualizations

- [epoch 020](https://media.tanh.xyz/seewhy/26-03-23/exp3_reconstructions_epoch020.png)
- [epoch 040](https://media.tanh.xyz/seewhy/26-03-23/exp3_reconstructions_epoch040.png)
- [epoch 060](https://media.tanh.xyz/seewhy/26-03-23/exp3_reconstructions_epoch060.png)
- [epoch 080](https://media.tanh.xyz/seewhy/26-03-23/exp3_reconstructions_epoch080.png)
- [epoch 100](https://media.tanh.xyz/seewhy/26-03-23/exp3_reconstructions_epoch100.png)
- [reconstructions final](https://media.tanh.xyz/seewhy/26-03-23/exp3_reconstructions_final.png)
- [pixel entropy](https://media.tanh.xyz/seewhy/26-03-23/exp3_entropy_final.png)
- [latent PCA](https://media.tanh.xyz/seewhy/26-03-23/exp3_latent_pca_final.svg)
- [learning curves](https://media.tanh.xyz/seewhy/26-03-23/exp3_learning_curves.svg)

## Observations

- **83.9% bin accuracy** vs exp2 where KL was collapsed (0.08) and all reconstructions identical
- KL healthy at 0.537/dim → ~34 nats/sample total; latent is actively used (free bits=0.5 helped)
- Accuracy climbed steadily: 81.3% (ep4) → 83.9% (ep99), plateauing ~ep85
- Loss still slightly decreasing at ep99 — not fully converged but slowing significantly
- Cosine LR decay kept training stable through 100 epochs without oscillation
- Recon NLL 0.724 vs exp2's ~0.730 — modest improvement from direct head + more capacity

## What to Try Next

- Larger gram dimension: `D_R=D_C=D_V=6` → D=216, for richer per-pixel representations
- More encoder queries `N_ENC_Q=8` — currently only 4 queries into a 64×64 gram
- Attention-style pooling in encoder instead of mean-pool over queries
- Multi-layer decoder: project z → 2-layer MLP → A, B instead of direct linear
- Higher `N_VAL_BINS=64` for finer pixel quantisation (may help at high accuracy regime)
- Larger `DEC_RANK` (32) for more expressive pseudo-gram
- Longer run (200 epochs) — curve not fully converged
