# Report 1 — Gram IMLE Baseline (exp1)

## Setup

Basic IMLE with the gram decoder from vae-gram/exp19. No encoder. No KL.

| Hyperparameter | Value |
|----------------|-------|
| `N_SAMPLES` | 1000 |
| `LATENT_DIM` | 64 |
| `MLP_HIDDEN` | 256 |
| `DEC_RANK` | 16 |
| `D` | 144 (D_R=D_C=6, D_V=4) |
| `ADAM_LR` | 3e-4 (cosine decay) |
| `BATCH_SIZE` | 128 |
| `MAX_EPOCHS` | 100 |
| `M_EVAL` | 5000 |

## Results

| Metric | Value |
|--------|-------|
| Bin accuracy (eval, M_EVAL=5000) | **87.1%** |
| Match L2 / pixel (eval) | 1702 |
| NLL (final epoch) | 0.6230 |
| Params | 1,252,784 |
| Time | 330s |

## Visualizations

**Learning curves**
https://media.tanh.xyz/seewhy/26-03-23/exp1_learning_curves.svg

**Prior samples** (z ~ N(0,I), soft mean, 5×8 grid)
https://media.tanh.xyz/seewhy/26-03-23/exp1_samples_5x8_seed0.png

**Nearest-neighbor reconstructions — final**
https://media.tanh.xyz/seewhy/26-03-23/exp1_recon_final.png

**Reconstructions during training**
- Epoch 20: https://media.tanh.xyz/seewhy/26-03-23/exp1_recon_epoch020.png
- Epoch 40: https://media.tanh.xyz/seewhy/26-03-23/exp1_recon_epoch040.png
- Epoch 60: https://media.tanh.xyz/seewhy/26-03-23/exp1_recon_epoch060.png
- Epoch 80: https://media.tanh.xyz/seewhy/26-03-23/exp1_recon_epoch080.png
- Epoch 100: https://media.tanh.xyz/seewhy/26-03-23/exp1_recon_epoch100.png

## Observations

- Bin accuracy plateaus at **87.1%** from around epoch 80 — training has converged.
- Match L2 steadily decreases throughout (2008 → 1989 train, 1714 → 1702 eval),
  suggesting the decoder is still improving coverage at the end of training.
- Prior samples (z ~ N(0,I)) are clean — no prior hole problem since there is
  no encoder and no posterior to misalign.

## Next Experiments

- **exp2**: increase `N_SAMPLES` (e.g. 5000) — better coverage per epoch,
  likely lower match L2 and higher bin accuracy.
- **exp3**: increase `LATENT_DIM` (e.g. 256 or 512) — unlike VAE, there is no
  FREE_BITS ceiling or prior hole; larger latent may improve expressivity freely.
- **exp4**: `lax.scan` over the batch loop for faster wall-clock time.
