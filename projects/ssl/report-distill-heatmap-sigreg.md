# Masked Distillation: Arch×Arch Heatmap — SigReg Teachers

**Method**: Masked distillation — frozen SigReg teacher encodes clean images;
student learns to predict teacher representations from augmented views (Gaussian noise,
25/50/75% pixel masks) via a FiLM predictor. 200 epochs, LATENT_DIM=1024.

**Grid**: 8 teacher architectures × 9 student architectures = 72 experiments.

---

## Linear Probe Accuracy

![Probe heatmap](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_sigreg_probe.png)

## 1-NN KNN Accuracy

![KNN heatmap](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_sigreg_knn.png)

## Δ vs Teacher Baseline

Each cell: student probe − teacher's own SigReg probe accuracy.
Positive = student surpasses the teacher.

![Delta heatmap](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_sigreg_delta.png)

---

## Key Findings

**Best teacher** (avg probe): **Conv** (97.6%)

**Best student** (avg probe): **Gram-GLU** (98.1%)

### Top 5 combinations

| Teacher | Student | Probe | KNN |
|---------|---------|-------|-----|
| Conv | Gram-GLU | 98.4% | 96.7% |
| Conv-2 | Gram-GLU | 98.2% | 95.9% |
| Conv | MLP-2 | 98.2% | 96.3% |
| Conv | ViT-patch | 98.2% | 97.1% |
| ViT-patch | Gram-GLU | 98.1% | 96.2% |

### Bottom 5 combinations

| Teacher | Student | Probe | KNN |
|---------|---------|-------|-----|
| Gram-single | ViT | 95.9% | 96.6% |
| ViT | ViT | 95.9% | 94.1% |
| Conv-2 | Conv-2 | 96.0% | 94.1% |
| Gram-GLU | ViT | 96.0% | 96.5% |
| MLP | Gram-single | 96.1% | 95.8% |

### Best student per teacher vs SigReg baseline

| Teacher | SigReg baseline | Best student | Best student probe | Δ |
|---------|-----------------|--------------|-------------------|---|
| MLP | 96.4% | Gram-GLU | 97.9% | +1.6 pp |
| MLP-2 | 96.8% | Gram-GLU | 98.0% | +1.2 pp |
| Conv | 95.3% | Gram-GLU | 98.4% | +3.1 pp |
| Conv-2 | 94.5% | Gram-GLU | 98.2% | +3.7 pp |
| ViT | 94.0% | Gram-GLU | 98.0% | +4.0 pp |
| ViT-patch | 93.8% | Gram-GLU | 98.1% | +4.2 pp |
| Gram-single | 96.4% | Gram-GLU | 97.9% | +1.5 pp |
| Gram-GLU | 97.0% | Gram-GLU | 98.0% | +1.0 pp |

### Average probe by teacher

| Teacher | Avg probe |
|---------|-----------|
| Conv | 97.6% |
| MLP-2 | 97.3% |
| ViT-patch | 97.2% |
| Conv-2 | 97.2% |
| Gram-GLU | 97.0% |
| MLP | 97.0% |
| Gram-single | 96.9% |
| ViT | 96.8% |

### Average probe by student

| Student | Avg probe |
|---------|-----------|
| Gram-GLU | 98.1% |
| MLP-2 | 97.4% |
| ViT-patch | 97.4% |
| Gram-dual | 97.3% |
| MLP | 97.2% |
| Conv-2 | 96.9% |
| Gram-single | 96.9% |
| Conv | 96.6% |
| ViT | 96.6% |
