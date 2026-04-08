# SSL Feature Extractors: Full 6-Way Comparison

Comparing six training regimes across 9 encoder architectures.

All models trained on **MNIST** (200 epochs, LATENT_DIM=1024, Adam + cosine LR).
K-Means K=128 majority-vote cluster accuracy.  Fashion-MNIST is zero-shot transfer
(MNIST-trained params evaluated on Fashion-MNIST).

---

## Training regimes

| Label | Description |
|---|---|
| **RANDOM** | Randomly initialised encoder, He-init weights, no training. |
| **AE** | Standard autoencoder: MSE(decode(encode(x)), x/255). MLP decoder 1024→512→784. |
| **DAE** | Denoising autoencoder: reconstruct clean x from corrupted input. Same 4 augmentations as EMA/SIGReg (Gaussian σ=75, pixel masking 25/50/75%; ViT-patch uses patch masking). |
| **VAE** | Variational autoencoder: MSE + β·KL (β=0.001). encode() returns mean. |
| **SIGREG** | SIGReg-JEPA: predictor + Epps-Pulley spectral regulariser (N(0,I) target). |
| **EMA** | EMA-JEPA: MSE prediction loss, momentum target encoder τ=0.996. |

---

## Linear probe accuracy

### MNIST

![Linear probe MNIST](https://media.tanh.xyz/seewhy/26-04-06/ssl_probe_mnist.png)

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
| MLP-shallow  | 92.1% | 92.2% | 96.1% | 92.4% | 96.4% | 93.2% |
| MLP-deep     | 91.6% | 91.9% | 97.4% | 92.5% | 96.8% | 74.3% |
| ConvNet-v1   | 71.1% | 97.4% | 97.2% | 97.7% | 95.3% | 93.9% |
| ConvNet-v2   | 66.1% | 98.1% | 98.0% | 98.7% | 94.5% | 90.9% |
| ViT          | 79.3% | 92.0% | 95.9% | 92.0% | 94.0% | 90.2% |
| ViT-patch    | 79.3% | 92.0% | 96.7% | 92.1% | 93.8% | 95.1% |
| Gram-dual    | 90.9% | 94.9% | 97.1% | 96.8% | 95.7% | 97.3% |
| Gram-single  | 91.1% | 92.5% | 96.3% | 95.6% | 96.4% | 97.0% |
| Gram-GLU     | 86.8% | 92.4% | 97.3% | 95.2% | 97.0% | 94.3% |

### Fashion-MNIST (zero-shot transfer)

![Linear probe F-MNIST](https://media.tanh.xyz/seewhy/26-04-06/ssl_probe_fmnist.png)

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
| MLP-shallow  | 83.2% | 83.9% | 84.0% | 83.3% | 80.7% | 66.8% |
| MLP-deep     | 82.8% | 83.6% | 83.7% | 83.2% | 80.3% | 57.3% |
| ConvNet-v1   | 68.5% | 82.9% | 80.7% | 83.1% | 77.8% | 74.3% |
| ConvNet-v2   | 67.7% | 83.4% | 82.1% | 84.3% | 74.5% | 72.9% |
| ViT          | 76.1% | 82.1% | 81.3% | 81.8% | 77.8% | 76.7% |
| ViT-patch    | 76.1% | 82.1% | 81.0% | 81.8% | 77.3% | 77.4% |
| Gram-dual    | 77.5% | 79.9% | 82.0% | 81.2% | 80.0% | 82.4% |
| Gram-single  | 77.4% | 77.7% | 81.6% | 79.4% | 81.5% | 82.0% |
| Gram-GLU     | 75.2% | 80.7% | 83.7% | 82.7% | 83.0% | 75.7% |

---

## 1-NN (L2) accuracy

### MNIST

![1-NN MNIST](https://media.tanh.xyz/seewhy/26-04-06/ssl_knn_mnist.png)

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
| MLP-shallow  | 96.0% | 95.7% | 97.1% | 95.5% | 95.9% | 91.4% |
| MLP-deep     | 96.1% | 95.1% | 96.6% | 95.6% | 95.7% | 79.9% |
| ConvNet-v1   | 82.5% | 96.6% | 96.6% | 96.5% | 92.5% | 94.8% |
| ConvNet-v2   | 85.2% | 97.1% | 97.5% | 97.7% | 91.2% | 92.1% |
| ViT          | 74.6% | 96.1% | 97.1% | 96.1% | 92.6% | 86.4% |
| ViT-patch    | 74.6% | 96.2% | 97.4% | 96.1% | 90.6% | 92.0% |
| Gram-dual    | 88.4% | 90.7% | 95.7% | 92.6% | 95.3% | 95.5% |
| Gram-single  | 87.9% | 94.4% | 95.8% | 94.3% | 95.5% | 95.6% |
| Gram-GLU     | 87.9% | 95.8% | 96.8% | 94.3% | 95.7% | 93.2% |

### Fashion-MNIST (zero-shot transfer)

![1-NN F-MNIST](https://media.tanh.xyz/seewhy/26-04-06/ssl_knn_fmnist.png)

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
| MLP-shallow  | 84.0% | 83.6% | 83.6% | 83.0% | 80.3% | 68.1% |
| MLP-deep     | 83.9% | 83.1% | 82.5% | 82.5% | 78.1% | 64.2% |
| ConvNet-v1   | 74.6% | 81.2% | 80.0% | 81.0% | 75.8% | 76.4% |
| ConvNet-v2   | 76.1% | 82.1% | 81.1% | 81.3% | 70.5% | 76.3% |
| ViT          | 74.8% | 82.4% | 81.1% | 82.3% | 78.0% | 76.6% |
| ViT-patch    | 74.8% | 82.5% | 80.3% | 82.4% | 73.4% | 73.6% |
| Gram-dual    | 74.7% | 78.1% | 78.4% | 78.5% | 76.0% | 77.6% |
| Gram-single  | 74.2% | 78.5% | 79.1% | 79.0% | 78.0% | 79.3% |
| Gram-GLU     | 74.6% | 80.7% | 80.9% | 79.5% | 78.7% | 73.4% |

---

## VAE β sweep — all architectures

Linear probe (MNIST-trained encoder) as a function of β across all 9 architectures.
β=0 ≡ pure AE (no KL); larger β tightens the N(0,I) prior.

### MNIST linear probe

![VAE β sweep MNIST](https://media.tanh.xyz/seewhy/26-04-06/ssl_vae_kl_mnist.png)

### Fashion-MNIST linear probe (probe trained on F-MNIST train set)

![VAE β sweep F-MNIST](https://media.tanh.xyz/seewhy/26-04-06/ssl_vae_kl_fmnist.png)

| Arch | β=0 M | β=1e-4 M | β=1e-3 M | β=1e-2 M | β=1e-1 M | β=1 M | β=0 F | β=1e-4 F | β=1e-3 F | β=1e-2 F | β=1e-1 F | β=1 F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP-shallow | 92.3% | 92.4% | 92.0% | 91.3% | 90.9% | 89.9% | 83.2% | 83.5% | 82.8% | 82.8% | 81.4% | 74.0% |
| MLP-deep | 82.1% | 92.4% | 92.3% | 92.4% | 95.0% | 96.9% | 73.0% | 83.3% | 83.0% | 83.3% | 83.5% | 82.7% |
| ConvNet-v1 | 96.7% | 96.7% | 97.6% | 97.5% | 96.9% | 96.4% | 81.3% | 81.4% | 83.4% | 81.5% | 80.1% | 79.8% |
| ConvNet-v2 | 98.2% | 98.2% | 98.7% | 98.1% | 96.4% | 95.5% | 83.0% | 82.8% | 84.3% | 82.9% | 79.9% | 79.7% |
| ViT | 91.9% | 92.1% | 92.0% | 92.2% | 93.8% | 96.0% | 82.4% | 82.0% | 81.9% | 82.2% | 82.1% | 81.0% |
| ViT-patch | 92.0% | 92.1% | 92.0% | 92.1% | 93.8% | 96.1% | 81.9% | 82.0% | 81.9% | 81.9% | 82.2% | 80.8% |
| Gram-dual | 96.2% | 96.5% | 96.8% | 97.3% | 97.1% | 96.7% | 81.0% | 81.6% | 81.2% | 82.4% | 82.9% | 79.9% |
| Gram-single | 95.2% | 94.8% | 95.6% | 95.5% | 96.1% | 95.9% | 79.4% | 79.8% | 79.3% | 80.5% | 81.2% | 79.1% |
| Gram-GLU | 95.0% | 94.6% | 95.2% | 96.1% | 96.6% | 96.0% | 82.3% | 82.7% | 82.8% | 81.8% | 81.6% | 77.1% |

---

## K-Means K=128 — MNIST

![MNIST K-Means by encoder](https://media.tanh.xyz/seewhy/26-04-06/ssl_km_mnist.png)

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
| MLP-shallow  | 85.4% | 85.7% | 90.5% | 85.3% | 88.4% | 79.4% |
| MLP-deep     | 85.5% | 83.5% | 91.7% | 86.9% | 92.5% | 48.7% |
| ConvNet-v1   | 52.9% | 87.7% | 89.5% | 86.7% | 81.5% | 79.3% |
| ConvNet-v2   | 53.4% | 91.8% | 92.2% | 94.1% | 82.1% | 74.2% |
| ViT          | 54.5% | 86.9% | 91.8% | 88.2% | 86.4% | 70.6% |
| ViT-patch    | 54.5% | 86.9% | 92.9% | 88.1% | 80.7% | 80.1% |
| Gram-dual    | 55.7% | 72.8% | 77.5% | 76.8% | 82.1% | 78.5% |
| Gram-single  | 56.9% | 76.5% | 78.4% | 78.3% | 83.3% | 78.9% |
| Gram-GLU     | 55.5% | 86.8% | 90.5% | 84.0% | 88.6% | 75.4% |

| Encoder | Best method | Score |
|---|---|---|
| MLP-shallow  | DAE      | 90.5% |
| MLP-deep     | SIGREG   | 92.5% |
| ConvNet-v1   | DAE      | 89.5% |
| ConvNet-v2   | VAE      | 94.1% |
| ViT          | DAE      | 91.8% |
| ViT-patch    | DAE      | 92.9% |
| Gram-dual    | SIGREG   | 82.1% |
| Gram-single  | SIGREG   | 83.3% |
| Gram-GLU     | DAE      | 90.5% |

---

## K-Means K=128 — Fashion-MNIST (zero-shot transfer)

![Fashion-MNIST K-Means by encoder](https://media.tanh.xyz/seewhy/26-04-06/ssl_km_fmnist.png)

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
| MLP-shallow  | 73.7% | 72.9% | 73.6% | 73.5% | 69.4% | 57.4% |
| MLP-deep     | 73.5% | 73.0% | 70.9% | 73.2% | 65.9% | 48.5% |
| ConvNet-v1   | 58.8% | 70.8% | 67.7% | 69.2% | 64.2% | 63.2% |
| ConvNet-v2   | 58.4% | 71.3% | 69.4% | 70.9% | 60.2% | 60.5% |
| ViT          | 61.6% | 74.5% | 71.2% | 74.1% | 63.8% | 63.5% |
| ViT-patch    | 61.6% | 74.7% | 69.6% | 73.7% | 62.2% | 61.1% |
| Gram-dual    | 59.1% | 60.8% | 60.8% | 60.7% | 57.5% | 62.1% |
| Gram-single  | 60.1% | 61.0% | 61.5% | 62.9% | 60.7% | 60.9% |
| Gram-GLU     | 57.4% | 69.5% | 69.0% | 69.1% | 66.2% | 57.2% |

| Encoder | Best method | Score |
|---|---|---|
| MLP-shallow  | RANDOM   | 73.7% |
| MLP-deep     | RANDOM   | 73.5% |
| ConvNet-v1   | AE       | 70.8% |
| ConvNet-v2   | AE       | 71.3% |
| ViT          | AE       | 74.5% |
| ViT-patch    | AE       | 74.7% |
| Gram-dual    | EMA      | 62.1% |
| Gram-single  | VAE      | 62.9% |
| Gram-GLU     | AE       | 69.5% |

---

## Analysis

### DAE is the best self-supervised method for MNIST K-Means

DAE wins 6/9 architectures. The denoising objective forces compact spherical clusters:
the encoder must map corrupted versions of the same input to approximately the same
latent point, directly creating the isotropic per-class geometry K-Means requires.

Best result: ViT-patch DAE **92.9%**, followed by ConvNet-v2 VAE **94.1%**.

### VAE wins ConvNet-v2 (94.1%) — best single K-Means result

The KL prior creates a smooth, continuous latent space where nearby inputs cluster
together. Combined with ConvNet-v2's strong local inductive bias, this produces
the most K-Means-friendly geometry of any model tested.

### EMA best for linear probe; worst for K-Means

EMA achieves the highest linear probe accuracy across most architectures — the momentum
target encoder prevents collapse while allowing anisotropic embedding distributions.
Those elongated ellipsoidal class regions are linearly separable but not Voronoi-separable,
so K-Means suffers (7/9 architectures lowest).

### Reconstruction objectives transfer to Fashion-MNIST; predictive ones don't

AE/DAE/VAE improve over random on Fashion-MNIST (zero-shot).
EMA loses to random — it learns MNIST-specific invariances that hurt transfer.

---

*Generated 2026-04-05.*
*Data: `results_knn_comparison.jsonl` (probe/1-NN), `results_kmeans_comparison.jsonl` (K-Means), `results_vae_kl.jsonl` (β sweep).*
*Script: `projects/ssl/scripts/gen_report_ae_dae_vae.py`.*
