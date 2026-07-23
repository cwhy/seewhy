# Masked Distillation: Arch×Arch Heatmap — Random (untrained) Teachers

**Method**: Masked distillation — frozen Random (untrained) teacher encodes clean images;
student learns to predict teacher representations from augmented views (Gaussian noise,
25/50/75% pixel masks) via a FiLM predictor. 200 epochs, LATENT_DIM=1024.

**Grid**: 9 teacher architectures × 9 student architectures = 81 experiments.

---

## Linear Probe Accuracy

![Probe heatmap](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_random_probe.png)

## 1-NN KNN Accuracy

![KNN heatmap](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_random_knn.png)

---

## Key Findings

**Best teacher** (avg probe): **MLP** (96.6%)

**Best student** (avg probe): **MLP-2** (96.6%)

### Top 5 combinations

| Teacher | Student | Probe | KNN |
|---------|---------|-------|-----|
| MLP | MLP-2 | 97.9% | 96.8% |
| Gram-single | Gram-GLU | 97.8% | 96.0% |
| Gram-single | MLP-2 | 97.6% | 96.0% |
| MLP-2 | MLP-2 | 97.5% | 96.8% |
| MLP | Conv-2 | 97.4% | 96.9% |

### Bottom 5 combinations

| Teacher | Student | Probe | KNN |
|---------|---------|-------|-----|
| Conv-2 | Gram-GLU | 85.4% | 93.3% |
| Conv-2 | ViT-patch | 86.2% | 85.1% |
| Conv-2 | ViT | 86.8% | 85.4% |
| Conv-2 | Conv | 87.1% | 91.5% |
| Conv-2 | Conv-2 | 87.3% | 93.2% |

### Average probe by teacher

| Teacher | Avg probe |
|---------|-----------|
| MLP | 96.6% |
| Gram-single | 96.6% |
| MLP-2 | 96.4% |
| Gram-dual | 96.0% |
| ViT-patch | 95.7% |
| ViT | 95.7% |
| Gram-GLU | 95.0% |
| Conv | 94.4% |
| Conv-2 | 89.9% |

### Average probe by student

| Student | Avg probe |
|---------|-----------|
| MLP-2 | 96.6% |
| Gram-dual | 96.3% |
| MLP | 96.3% |
| Gram-single | 95.9% |
| Gram-GLU | 95.3% |
| Conv-2 | 94.7% |
| Conv | 94.4% |
| ViT-patch | 94.0% |
| ViT | 93.0% |
