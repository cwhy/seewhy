# Masked Distillation: Arch×Arch Heatmap

**Method**: Masked distillation — frozen EMA-JEPA teacher encodes clean images;
student learns to predict teacher representations from augmented views (Gaussian noise,
25/50/75% pixel masks) via a FiLM predictor. 200 epochs, LATENT_DIM=1024.

**Grid**: 9 teacher architectures × 9 student architectures = 81 experiments.
All teachers are pre-trained EMA-JEPA encoders at LATENT_DIM=1024.

---

## Linear Probe Accuracy

![Probe heatmap](https://media.tanh.xyz/seewhy/26-04-06/ssl_distill_heatmap_probe.png)

## 1-NN KNN Accuracy

![KNN heatmap](https://media.tanh.xyz/seewhy/26-04-06/ssl_distill_heatmap_knn.png)

## Δ vs EMA-JEPA Baseline

Each cell shows how much the student's probe accuracy differs from the **teacher's** own
EMA-JEPA baseline. Positive = student surpasses the teacher; negative = student falls short.

![Delta heatmap](https://media.tanh.xyz/seewhy/26-04-06/ssl_distill_heatmap_delta.png)

---

## Key Findings

**Best teacher** (avg probe across all students): **ViT-patch** (97.4%)

**Best student** (avg probe across all teachers): **Gram-GLU** (97.3%)

### Top 5 combinations (linear probe)

| Teacher | Student | Probe | KNN |
|---------|---------|-------|-----|
| Conv | MLP-2 | 98.1% | 96.6% |
| MLP | Gram-GLU | 97.9% | 96.7% |
| Conv | MLP | 97.9% | 96.5% |
| ViT-patch | Gram-GLU | 97.9% | 96.4% |
| Conv-2 | MLP-2 | 97.7% | 95.9% |

### Bottom 5 combinations (linear probe)

| Teacher | Student | Probe | KNN |
|---------|---------|-------|-----|
| MLP-2 | MLP-2 | 92.3% | 92.3% |
| ViT | ViT | 93.4% | 88.9% |
| MLP | MLP-2 | 94.0% | 91.6% |
| MLP-2 | ViT | 94.7% | 95.6% |
| Conv-2 | Conv-2 | 94.9% | 94.8% |

### Best student per teacher vs teacher's own baseline

| Teacher | Teacher baseline | Best student | Best student probe | Δ |
|---------|-----------------|--------------|-------------------|---|
| MLP | 93.2% | Gram-GLU | 97.9% | +4.8 pp |
| MLP-2 | 74.3% | Gram-dual | 97.1% | +22.8 pp |
| Conv | 93.9% | MLP-2 | 98.1% | +4.2 pp |
| Conv-2 | 90.9% | MLP-2 | 97.7% | +6.8 pp |
| ViT | 90.2% | Gram-dual | 97.2% | +7.1 pp |
| ViT-patch | 95.1% | Gram-GLU | 97.9% | +2.8 pp |
| Gram-dual | 97.3% | Gram-GLU | 97.6% | +0.3 pp |
| Gram-single | 96.9% | Gram-GLU | 97.7% | +0.8 pp |
| Gram-GLU | 94.3% | Conv-2 | 97.1% | +2.9 pp |

### Average probe accuracy by teacher

| Teacher | Avg probe (all students) |
|---------|--------------------------|
| ViT-patch | 97.4% |
| Conv | 97.3% |
| Gram-dual | 96.6% |
| MLP | 96.6% |
| Gram-single | 96.6% |
| Gram-GLU | 96.6% |
| Conv-2 | 96.6% |
| ViT | 96.0% |
| MLP-2 | 95.9% |

### Average probe accuracy by student

| Student | Avg probe (all teachers) |
|---------|--------------------------|
| Gram-GLU | 97.3% |
| Gram-dual | 97.2% |
| MLP | 96.8% |
| Gram-single | 96.8% |
| Conv-2 | 96.6% |
| ViT-patch | 96.5% |
| Conv | 96.4% |
| MLP-2 | 96.3% |
| ViT | 95.7% |

### Self-distillation diagonal (same arch as teacher and student)

| Arch | Self-distill probe | EMA-JEPA baseline | Δ |
|------|--------------------|--------------------|---|
| MLP | 96.7% | 93.2% | +3.5 pp |
| MLP-2 | 92.3% | 74.3% | +18.0 pp |
| Conv | 96.7% | 93.9% | +2.7 pp |
| Conv-2 | 94.9% | 90.9% | +4.0 pp |
| ViT | 93.4% | 90.2% | +3.2 pp |
| ViT-patch | 96.9% | 95.1% | +1.9 pp |
| Gram-dual | 97.1% | 97.3% | -0.2 pp |
| Gram-single | 96.8% | 96.9% | -0.1 pp |
| Gram-GLU | 96.9% | 94.3% | +2.6 pp |
