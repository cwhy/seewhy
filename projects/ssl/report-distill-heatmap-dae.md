# Masked Distillation: Arch×Arch Heatmap — DAE (Denoising Autoencoder) Teachers

**Method**: Masked distillation — frozen DAE (Denoising Autoencoder) teacher encodes clean images;
student learns to predict teacher representations from augmented views (Gaussian noise,
25/50/75% pixel masks) via a FiLM predictor. 200 epochs, LATENT_DIM=1024.

**Grid**: 9 teacher architectures × 9 student architectures = 81 experiments.

---

## Linear Probe Accuracy

![Probe heatmap](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_dae_probe.png)

## 1-NN KNN Accuracy

![KNN heatmap](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_dae_knn.png)

## Δ vs Teacher Baseline

Each cell: student probe − teacher's own DAE (Denoising Autoencoder) probe accuracy.
Positive = student surpasses the teacher.

![Delta heatmap](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_dae_delta.png)

---

## Key Findings

**Best teacher** (avg probe): **Conv-2** (98.0%)

**Best student** (avg probe): **MLP-2** (98.0%)

### Top 5 combinations

| Teacher | Student | Probe | KNN |
|---------|---------|-------|-----|
| Conv-2 | MLP-2 | 98.6% | 97.3% |
| Conv | MLP-2 | 98.6% | 97.0% |
| Conv-2 | ViT-patch | 98.5% | 97.9% |
| Conv-2 | MLP | 98.2% | 97.5% |
| Conv-2 | Gram-GLU | 98.2% | 97.2% |

### Bottom 5 combinations

| Teacher | Student | Probe | KNN |
|---------|---------|-------|-----|
| Gram-single | ViT | 95.1% | 96.9% |
| Gram-dual | ViT | 95.5% | 96.8% |
| MLP | ViT | 95.8% | 97.2% |
| ViT | ViT | 96.4% | 97.4% |
| Gram-dual | Conv | 96.4% | 96.0% |

### Best student per teacher vs DAE (Denoising Autoencoder) baseline

| Teacher | DAE (Denoising Autoencoder) baseline | Best student | Best student probe | Δ |
|---------|-----------------|--------------|-------------------|---|
| MLP | 96.1% | Gram-GLU | 97.8% | +1.6 pp |
| MLP-2 | 97.4% | Gram-GLU | 97.9% | +0.5 pp |
| Conv | 97.2% | MLP-2 | 98.6% | +1.4 pp |
| Conv-2 | 98.0% | MLP-2 | 98.6% | +0.7 pp |
| ViT | 95.9% | Conv-2 | 98.0% | +2.1 pp |
| ViT-patch | 96.7% | Conv-2 | 98.1% | +1.4 pp |
| Gram-dual | 97.1% | MLP-2 | 98.0% | +1.0 pp |
| Gram-single | 96.3% | Gram-GLU | 97.9% | +1.6 pp |
| Gram-GLU | 97.3% | MLP-2 | 97.9% | +0.6 pp |

### Average probe by teacher

| Teacher | Avg probe |
|---------|-----------|
| Conv-2 | 98.0% |
| Conv | 97.7% |
| ViT-patch | 97.5% |
| MLP-2 | 97.4% |
| ViT | 97.3% |
| Gram-GLU | 97.1% |
| MLP | 97.1% |
| Gram-dual | 97.0% |
| Gram-single | 97.0% |

### Average probe by student

| Student | Avg probe |
|---------|-----------|
| MLP-2 | 98.0% |
| Gram-GLU | 97.8% |
| Conv-2 | 97.7% |
| ViT-patch | 97.4% |
| MLP | 97.3% |
| Gram-dual | 97.3% |
| Conv | 97.0% |
| Gram-single | 97.0% |
| ViT | 96.5% |
