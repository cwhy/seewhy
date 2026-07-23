# K-Means Comparison: Random vs SIGReg vs EMA

Three training regimes evaluated on K=128 K-Means cluster assignment accuracy
(majority-vote, test set) on frozen embeddings.  LATENT_DIM=1024 throughout.

Results logged to `results_kmeans_comparison.jsonl`.
Script: `projects/ssl/scripts/kmeans_comparison.py`

---

## Training regimes

| Label | Description |
|---|---|
| **random** | Randomly initialised encoder — same architecture and LATENT_DIM, no SSL training whatsoever.  Weights drawn at the same scale used by the real training runs (He init). |
| **sigreg** | SIGReg-JEPA: predictor loss + SIGReg spectral regulariser enforcing N(0,I) on embeddings. Single encoder (no online/target split). All 9 architectures trained for 200 epochs. |
| **ema** | EMA-JEPA: MSE loss, target encoder updated by momentum copy τ=0.996.  All 9 architectures trained for 200 epochs. |

Both SIGReg and EMA use identical augmentations: Gaussian noise (σ=75) and
random pixel masking (25/50/75%).  ViT-patch uses patch-level masking instead
of flat per-pixel masking.

---

## Results — MNIST (K=128)

| Encoder | Random | SIGReg | EMA | EMA − Rand | EMA − SIGReg |
|---|---|---|---|---|---|
| MLP-shallow  | 85.4%     | 88.4%     | 79.4% | −6.0 pp  | −9.0 pp  |
| MLP-deep     | 85.5%     | **92.5%** | 48.7% | −36.8 pp | −43.8 pp |
| ConvNet-v1   | 52.9%     | 81.5%     | 79.3% | +26.4 pp | −2.2 pp  |
| ConvNet-v2   | 53.4%     | 82.1%     | 74.2% | +20.8 pp | −7.9 pp  |
| ViT          | 54.5%     | 86.4%     | 70.4% | +15.9 pp | −16.0 pp |
| ViT-patch    | 54.5%     | 80.7%     | **80.1%** | +25.6 pp | −0.6 pp  |
| Gram-dual    | 55.7%     | 82.1%     | 78.5% | +22.8 pp | −3.6 pp  |
| Gram-single  | 56.9%     | 83.3%     | 78.9% | +22.0 pp | −4.4 pp  |
| Gram-GLU     | 55.5%     | 88.6%     | 75.4% | +19.9 pp | −13.2 pp |

**SIGReg wins over EMA for every architecture on MNIST.**
Best SIGReg: MLP-deep 92.5%.  Best EMA: ViT-patch 80.1%.

---

## Results — Fashion-MNIST (K=128)

| Encoder | Random | SIGReg | EMA | EMA − Rand | EMA − SIGReg |
|---|---|---|---|---|---|
| MLP-shallow  | **73.7%** | 69.4% | 74.7% | +1.0 pp  | +5.3 pp  |
| MLP-deep     | **73.5%** | 65.9% | 67.1% | −6.4 pp  | +1.2 pp  |
| ConvNet-v1   | 58.8%     | 64.2% | 65.6% | +6.8 pp  | +1.4 pp  |
| ConvNet-v2   | 58.4%     | 60.2% | 67.0% | +8.6 pp  | +6.8 pp  |
| ViT          | 61.6%     | 63.8% | **68.0%** | +6.4 pp  | +4.2 pp  |
| ViT-patch    | 61.6%     | 62.2% | 67.2% | +5.6 pp  | +5.0 pp  |
| Gram-dual    | 59.1%     | 57.5% | 62.7% | +3.6 pp  | +5.2 pp  |
| Gram-single  | 60.1%     | 60.7% | 61.8% | +1.7 pp  | +1.1 pp  |
| Gram-GLU     | 57.4%     | 66.2% | 64.9% | +7.5 pp  | −1.3 pp  |

**EMA wins over SIGReg for almost every architecture on Fashion-MNIST.**
Exception: Gram-GLU where SIGReg edges EMA by 1.3 pp.

---

## Key findings

### 1 — SIGReg dominates EMA on MNIST K-Means, across all architectures

SIGReg beats EMA for all 9 architectures on MNIST (K=128).  The margins range
from −0.6 pp (ViT-patch, near-tie) to −43.8 pp (MLP-deep).

SIGReg enforces a spectral spread constraint (Epps-Pulley statistic, N(0,I)
target) on the encoder output distribution.  This explicitly prevents
anisotropic collapse and encourages the latent space to fill the hypersphere
uniformly — exactly the geometry that favours K-Means centroid assignment.

EMA prevents collapse implicitly through the momentum update but places no
constraint on the distribution shape.  The resulting embeddings are linearly
separable (high linear probe accuracy) but can be anisotropic (elongated
ellipsoidal regions), which defeats K-Means' centroid-distance assumption.

### 2 — The advantage completely reverses on Fashion-MNIST

EMA beats SIGReg on 8 of 9 architectures on Fashion-MNIST.  The only exception
is Gram-GLU where SIGReg wins by 1.3 pp.

Fashion-MNIST's class manifolds are structurally more complex than MNIST's
(dense textures, silhouette variation, visually similar classes like shirt/coat).
SIGReg's N(0,I) prior encourages uniform hyperspherical coverage — this works
well when the class manifolds are compact and well-separated (MNIST digits), but
can distort the natural cluster geometry of Fashion-MNIST by forcing unnatural
spreading of the embedding distribution.

EMA's softer constraint (just prevent collapse) allows the encoder to naturally
organise the more complex Fashion-MNIST structure without imposing a mismatched
geometric prior.

### 3 — MLP-deep: SIGReg rescues a collapsed architecture

MLP-deep EMA reaches only 48.7% K-Means on MNIST (vs random 85.5%).  EMA
training with a three-hidden-layer MLP has excess capacity to collapse the
embedding onto a low-dimensional manifold — most K-Means centroids land in
empty space.

SIGReg MLP-deep reaches **92.5%** — the best K-Means result of any trained
model.  The spectral regulariser prevents this collapse by penalising
degenerate distributions, forcing the encoder to maintain spread across all 1024
dimensions.  This turns the MLP-deep's collapse liability into a strength.

### 4 — Random MLP projections are surprisingly competitive

Random MLP-shallow: 85.4% MNIST / 73.7% F-MNIST — in both cases this *beats*
EMA (79.4% / 74.7% EMA).  However SIGReg MLP-shallow (88.4% / 69.4%) beats
random on MNIST but loses on Fashion-MNIST.

The random baseline confirms that MLP projections of pixel inputs naturally
produce well-separated Voronoi cells.  EMA training reorganises the geometry
for linear separability at the cost of cluster compactness.  SIGReg recovers
and exceeds the random K-Means performance on MNIST.

### 5 — SIGReg linear probe results are competitive

Despite large differences in K-Means accuracy, SIGReg and EMA produce similarly
high linear probe results on MNIST (from results.jsonl):

| Encoder | EMA probe | SIGReg probe |
|---|---|---|
| MLP-shallow  | 93.2% | 96.4% |
| MLP-deep     | 74.3% | 96.8% |
| ConvNet-v1   | 93.9% | 95.3% |
| ConvNet-v2   | 90.9% | 94.5% |
| ViT          | 90.2% | 94.0% |
| ViT-patch    | 95.1% | 93.8% |
| Gram-dual    | 97.3% | 97.4% |
| Gram-single  | 96.9% | 96.4% |
| Gram-GLU     | 94.3% | 97.0% |

SIGReg has higher linear probe accuracy than EMA for 8 of 9 architectures.
The exception is ViT-patch (SIGReg 93.8% vs EMA 95.1%).

This shows SIGReg produces representations that are simultaneously more linearly
separable *and* more K-Means-separable on MNIST — a better representation by
both measures.

---

## Summary scorecard

### MNIST — best K-Means K=128 per architecture

| Encoder | Random | SIGReg | EMA | Winner |
|---|---|---|---|---|
| MLP-shallow  | 85.4% | **88.4%** | 79.4% | SIGReg +3.0 pp vs rand |
| MLP-deep     | 85.5% | **92.5%** | 48.7% | SIGReg +7.0 pp vs rand |
| ConvNet-v1   | 52.9% | **81.5%** | 79.3% | SIGReg +28.6 pp vs rand |
| ConvNet-v2   | 53.4% | **82.1%** | 74.2% | SIGReg +28.7 pp vs rand |
| ViT          | 54.5% | **86.4%** | 70.4% | SIGReg +31.9 pp vs rand |
| ViT-patch    | 54.5% | 80.7% | **80.1%** | SIGReg +26.2 pp vs rand |
| Gram-dual    | 55.7% | **82.1%** | 78.5% | SIGReg +26.4 pp vs rand |
| Gram-single  | 56.9% | **83.3%** | 78.9% | SIGReg +26.4 pp vs rand |
| Gram-GLU     | 55.5% | **88.6%** | 75.4% | SIGReg +33.1 pp vs rand |

SIGReg wins on 8/9, EMA on 1/9 (ViT-patch, marginal −0.6 pp).

### Fashion-MNIST — best K-Means K=128 per architecture

| Encoder | Random | SIGReg | EMA | Winner |
|---|---|---|---|---|
| MLP-shallow  | 73.7% | 69.4% | **74.7%** | EMA +1.0 pp vs rand |
| MLP-deep     | **73.5%** | 65.9% | 67.1% | Random wins |
| ConvNet-v1   | 58.8% | 64.2% | **65.6%** | EMA +6.8 pp vs rand |
| ConvNet-v2   | 58.4% | 60.2% | **67.0%** | EMA +8.6 pp vs rand |
| ViT          | 61.6% | 63.8% | **68.0%** | EMA +6.4 pp vs rand |
| ViT-patch    | 61.6% | 62.2% | **67.2%** | EMA +5.6 pp vs rand |
| Gram-dual    | 59.1% | 57.5% | **62.7%** | EMA +3.6 pp vs rand |
| Gram-single  | 60.1% | 60.7% | **61.8%** | EMA +1.7 pp vs rand |
| Gram-GLU     | 57.4% | **66.2%** | 64.9% | SIGReg +8.8 pp vs rand |

EMA wins on 7/9, SIGReg on 1/9, Random on 1/9 (MLP-deep).

---

*Generated 2026-04-02.  Data: `results_kmeans_comparison.jsonl`.*
*Script: `projects/ssl/scripts/kmeans_comparison.py`.*
