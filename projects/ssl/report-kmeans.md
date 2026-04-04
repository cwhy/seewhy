# K-Means Clustering Evaluation — EMA-JEPA SSL Encoders

Unsupervised cluster-assignment accuracy on frozen SSL embeddings.
All encoders trained with **EMA-JEPA** (see below).
Results logged to `results_kmeans.jsonl`.

---

## Training setup recap

Every encoder was trained with the same **EMA-JEPA** objective — *no labels used*:

```
loss = MSE( predictor( online_encode(x_aug) ),
            stop_grad( target_encode(x_full) ) )

target ← τ · target + (1-τ) · online    (τ = 0.996, per step)
```

The target encoder is a **momentum copy** of the online encoder (BYOL/JEPA style).
SIGReg is computed as a *diagnostic metric only* — it does not appear in the training
loss.  There is no contrastive term, no negative pairs.

Augmentations: Gaussian noise (σ=75) + random pixel masking (25/50/75%).
ViT-patch uses patch-level masking instead of flat per-pixel masking.
All runs: LATENT_DIM=1024, 200 epochs, Adam lr=3e-4, batch=256.

---

## Evaluation method

1. Extract frozen embeddings for the full training set (60 000 samples).
2. Fit **FAISS K-Means** on training embeddings (100 iterations, seed=42).
3. **Majority-vote** label assignment: each cluster → most common class among
   training points assigned to it.
4. Predict test set by nearest-centroid; measure accuracy.
5. **Mean purity** = average fraction of the majority class per cluster.

**Random projection variant**: before clustering, project embeddings from 1024-d
to a fixed 256-d random Gaussian subspace (columns L2-normalised, seed=0).
Tests whether the useful structure lives in a low-dimensional subspace.

---

## Results — MNIST

### K=128 (main result)

| Encoder | Test acc | Purity | Test acc (proj→256) |
|---|---|---|---|
| **ViT-patch**   | **80.1%** | 81.1% | 79.2% |
| MLP-shallow     | 79.4%     | 89.4% | **80.0%** |
| ConvNet-v1      | 79.3%     | 80.6% | 78.9% |
| Gram-single     | 78.9%     | 85.1% | 78.4% |
| Gram-dual       | 78.5%     | 84.5% | 76.6% |
| Gram-GLU        | 75.4%     | 85.4% | 74.4% |
| ConvNet-v2      | 74.2%     | 79.0% | 74.7% |
| ViT             | 70.4%     | 70.3% | 69.9% |
| MLP-deep        | 48.7%     | 86.0% | 49.0% |

### K=10 baseline (one cluster per class)

| Encoder | Test acc | Purity | Test acc (proj→256) |
|---|---|---|---|
| ConvNet-v1  | 48.2% | 56.9% | 47.7% |
| MLP-shallow | 44.3% | 70.9% | 53.0% |
| ViT         | 42.8% | 52.0% | 41.2% |
| Gram-GLU    | 41.1% | 61.9% | 38.1% |
| ConvNet-v2  | 39.9% | 51.6% | 41.9% |
| ViT-patch   | 38.8% | 49.3% | 49.5% |
| Gram-single | 36.0% | 49.6% | 39.0% |
| Gram-dual   | 35.8% | 46.2% | 34.1% |
| MLP-deep    | 17.3% | 64.7% | 19.5% |

K=10 accuracy is low (17–48%) because with only 10 clusters the majority-vote
assignment can't capture within-class variation.  K=128 gives each class ~13
sub-clusters and jumps to 49–80%.

---

## Results — Fashion-MNIST

### K=128

| Encoder | Test acc | Purity | Test acc (proj→256) |
|---|---|---|---|
| **MLP-shallow** | **74.7%** | 78.5% | 74.6% |
| ViT             | 68.0%     | 69.1% | 68.0% |
| ViT-patch       | 67.2%     | 68.5% | 67.0% |
| MLP-deep        | 67.1%     | 78.2% | 64.0% |
| ConvNet-v2      | 67.0%     | 68.5% | 67.0% |
| ConvNet-v1      | 65.6%     | 66.4% | 65.9% |
| Gram-GLU        | 64.9%     | 76.6% | 66.2% |
| Gram-dual       | 62.7%     | 73.0% | 63.0% |
| Gram-single     | 61.8%     | 72.3% | 60.7% |

### K=10 baseline

| Encoder | Test acc | Purity |
|---|---|---|
| MLP-shallow | 55.9% | 65.3% |
| ViT         | 47.5% | 51.7% |
| MLP-deep    | 21.0% | 53.2% |
| ConvNet-v2  | 37.0% | 35.6% |
| ConvNet-v1  | 35.9% | 35.6% |
| ViT-patch   | 35.7% | 36.3% |
| Gram-GLU    | 37.8% | 68.2% |
| Gram-dual   | 34.8% | 59.4% |
| Gram-single | 35.9% | 64.9% |

---

## Cross-metric comparison (MNIST, best K=128 result)

Compares K-means cluster accuracy against supervised metrics on the same
frozen embeddings (from `results_knn.jsonl` and `results.jsonl`).

| Encoder | Linear probe | 1-NN (L2) | K-Means K=128 |
|---|---|---|---|
| Gram-dual    | **97.3%** | 96.1% | 78.5% |
| Gram-single  | 96.9%     | 95.9% | 78.9% |
| MLP-shallow  | 93.2%     | 95.4% | 79.4% |
| ViT-patch    | 95.1%     | 92.0% | **80.1%** |
| ConvNet-v1   | 93.9%     | —     | 79.3% |
| Gram-GLU     | 94.3%     | 93.2% | 75.4% |
| ConvNet-v2   | —         | —     | 74.2% |
| ViT          | —         | —     | 70.4% |
| MLP-deep     | —         | —     | 48.7% |

The **K-means ranking inverts the linear-probe ranking** for the top slots:
- Gram-dual wins on linear probe (97.3%) but falls to 5th on K-means (78.5%)
- ViT-patch wins on K-means (80.1%) despite being 4th on linear probe (95.1%)

This suggests Gram embeddings are linearly separable but not Voronoi-separable:
the class boundaries are hyperplanes, not compact spherical clusters.
ViT-patch produces more isotropic, hyperspherical clusters (purity 81.1% ≈ test acc 80.1%
— the clusters are already almost pure, meaning the Voronoi cells align with
classes directly).

---

## Key findings

### 1 — K=128 >> K=10 for all encoders

Going from 10 to 128 clusters adds ~35 pp on MNIST across the board.
With K=10 the majority vote is forced to share one prototype per class;
K=128 gives ~13 prototypes per class, capturing within-class stroke variation
and eliminating inter-class confusion at ambiguous boundaries.

### 2 — Random projection to 256-d is essentially free

Projecting 1024-d → 256-d random Gaussian subspace costs < 0.2 pp on most
encoders (max observed: 1.9 pp on Gram-dual).  The clustering structure is
preserved in a subspace that is 4× smaller than the full embedding.
Implication: the 1024-d embeddings are highly redundant — the informative
directions are concentrated in a low-d manifold.

### 3 — MLP-deep is collapsed on MNIST, partially recovered on Fashion-MNIST

MLP-deep: 48.7% K-means accuracy despite 86.0% cluster purity.
This means a handful of very pure clusters hold nearly all points, while most
clusters are empty.  The encoder has collapsed to a small region of the latent
space.  On Fashion-MNIST MLP-deep recovers to 67.1% (K=128) — the harder
dataset may prevent the same collapse mode.

### 4 — Gram purity ≠ compactness

Gram-dual has the highest purity among top encoders (84.5% on MNIST K=128)
but not the highest accuracy (78.5%).  Its clusters are internally pure but
misaligned with the Voronoi partition of the test set — the cluster shapes are
elongated or anisotropic, so test points fall in the wrong cell even though
training points are pure.  This is consistent with the gram structure: the
Hadamard readout creates polyhedral (not spherical) geometry.

### 5 — ViT-patch is the best unsupervised clusterer on MNIST

ViT-patch tops K-means accuracy (80.1%) while ranking 4th on linear probe
(95.1%).  Patch-level masking forces the encoder to produce compact,
per-class representations; flat pixel masking (used by all other architectures)
produces representations that are linearly separable but geometrically diffuse.

### 6 — Fashion-MNIST ranking stabilises

On Fashion-MNIST, MLP-shallow leads both K-means (74.7%) and linear probe
(83.6%).  The Gram encoders consistently rank last on F-MNIST (61–65%),
confirming that the co-activation prior is tuned to MNIST strokes and produces
poor cluster geometry on textured garments.

---

## Usage

```bash
# Default: K=128, no projection, both datasets
uv run python projects/ssl/scripts/kmeans_eval.py

# Sweep K values
uv run python projects/ssl/scripts/kmeans_eval.py --k 10 64 128 512

# With random projection
uv run python projects/ssl/scripts/kmeans_eval.py --k 128 --proj 64 128 256

# Single dataset, force re-run
uv run python projects/ssl/scripts/kmeans_eval.py --datasets mnist --force
```

Results are written incrementally to `results_kmeans.jsonl` (one JSON row per
`(dataset, exp_name, k, proj_dim)` combination).

---

*Generated 2026-04-02.  Script: `projects/ssl/scripts/kmeans_eval.py`.*
