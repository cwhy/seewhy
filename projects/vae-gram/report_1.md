# Report — exp1: Baseline Gram VAE

## Config

| Param | Value |
|---|---|
| D_R, D_C, D_V | 4, 4, 4 |
| D (gram space) | 64 |
| N_ENC_Q | 4 |
| LATENT_DIM | 32 |
| DEC_RANK | 8 |
| N_VAL_BINS | 32 |
| ADAM_LR | 1e-3 |
| KL_WEIGHT | 1.0 |
| KL_WARMUP_EPOCHS | 10 |
| BATCH_SIZE | 128 |
| MAX_EPOCHS | 30 |
| n_params | 50,340 |
| time | 31s |

## Results

| Epoch | β | recon NLL | KL | total |
|---|---|---|---|---|
| 0 | 0.00 | 1.2615 | 9.63 | 1.2615 |
| 1 | 0.10 | 0.8619 | 1.00 | 0.9619 |
| 5 | 0.50 | 0.8512 | 0.18 | 0.9436 |
| 10 | 1.00 | 0.8555 | 0.10 | 0.9601 |
| 20 | 1.00 | 0.8194 | 0.08 | 0.8996 |
| 29 | 1.00 | **0.8098** | **0.0804** | **0.8902** |

Random init baseline: `log(32) = 3.47` nats/pixel.

## Visualizations

- [epoch 05](https://media.tanh.xyz/seewhy/26-03-22/exp1_reconstructions_epoch05.png)
- [epoch 10](https://media.tanh.xyz/seewhy/26-03-22/exp1_reconstructions_epoch10.png)
- [epoch 15](https://media.tanh.xyz/seewhy/26-03-22/exp1_reconstructions_epoch15.png)
- [epoch 20](https://media.tanh.xyz/seewhy/26-03-22/exp1_reconstructions_epoch20.png)
- [epoch 25](https://media.tanh.xyz/seewhy/26-03-22/exp1_reconstructions_epoch25.png)
- [epoch 30](https://media.tanh.xyz/seewhy/26-03-22/exp1_reconstructions_epoch30.png)
- [reconstructions final](https://media.tanh.xyz/seewhy/26-03-22/exp1_reconstructions_final.png)
- [pixel entropy](https://media.tanh.xyz/seewhy/26-03-22/exp1_entropy_final.png)
- [latent PCA](https://media.tanh.xyz/seewhy/26-03-22/exp1_latent_pca_final.png)
- [learning curves](https://media.tanh.xyz/seewhy/26-03-22/exp1_learning_curves.png)

## Observations

- Loss was still decreasing at epoch 29 — not converged, more epochs likely help
- KL settled at ~0.08/dim → total ~2.6 nats/sample; latent is used but lightly
- Epoch 0 (β=0) drops recon from 3.47 → 1.26 immediately, showing the encoder learns fast without KL pressure
- KL spike at epoch 0 (9.6) then rapid collapse as β kicks in — expected with warmup

## What to Try Next

- More epochs (60–100) — loss still moving at epoch 29
- Larger D: `D_R=D_C=D_V=8` → D=512 for a richer gram
- More latent dims: `LATENT_DIM=64`
- Higher `DEC_RANK` (16 or 32) for a more expressive pseudo-gram
- More encoder queries: `N_ENC_Q=8`
- Finer bins: `N_VAL_BINS=64`
- Cosine LR decay instead of fixed Adam
