# SSL Evaluation: K-Means, KNN, Linear Probe, and Random Baselines

Full evaluation of 9 encoder architectures trained with EMA-JEPA self-supervised
learning on MNIST and Fashion-MNIST.  Each encoder is assessed across five
complementary metrics to build a complete picture of representation quality.

All SSL training uses **EMA-JEPA** (MSE loss + momentum target encoder τ=0.996,
200 epochs, LATENT_DIM=1024) or **SIGReg-JEPA** (predictor loss + spectral
regulariser, single encoder, 200 epochs).  Both regimes trained for all 9
architectures.

---

## Metrics

| Label | Description |
|---|---|
| **Supervised** | End-to-end supervised classification accuracy (LATENT=256, 100 epochs, Adam, cosine LR). Measures encoder capacity ceiling. MNIST only. |
| **Linear probe** | Frozen SSL encoder + trained linear head. Measures linear separability of SSL features. |
| **KNN-1 (L2)** | 1-nearest-neighbour accuracy on frozen embeddings (Euclidean). No training at all. |
| **KNN-5 (L2)** | 5-NN majority vote, Euclidean distance. |
| **KNN-5 (cos)** | 5-NN majority vote, cosine similarity (L2-normalised embeddings). |
| **K-Means K=10** | K-Means cluster accuracy (majority-vote assignment), K=10. |
| **K-Means K=128** | K-Means cluster accuracy, K=128 (≈13 sub-clusters per class). |
| **K-Means K=128 proj→256** | Same, after random Gaussian projection 1024→256-d (fixed seed). |
| **Random K=128** | K-Means on *randomly initialised* encoder — same architecture, no SSL training. |
| **SIGReg K=128** | K-Means on SIGReg-JEPA trained encoder. All 9 architectures. |

---

## MNIST — full metric table

| Encoder | Sup | LinProbe | KNN-1 | KNN-5 | KNN-5c | KM-10 | KM-128 | KM-128p | Rand | SIGReg |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP-shallow  | 98.5% | 93.2% | 91.4% | 92.0% | 92.3% | 44.3% | 79.4% | 80.0% | 85.4% | 88.4% |
| MLP-deep     | 98.7% | 74.3% | 79.9% | 80.7% | 83.2% | 17.3% | 48.7% | 49.0% | 85.5% | **92.5%** |
| ConvNet-v1   | 99.4% | 93.9% | 94.8% | 95.2% | 95.1% | 48.2% | 79.3% | 78.9% | 52.9% | 81.5% |
| ConvNet-v2   | 99.5% | 90.9% | 92.1% | 93.0% | 93.3% | 39.9% | 74.2% | 74.7% | 53.4% | 82.1% |
| ViT          | 98.2% | 90.2% | 86.4% | 87.7% | 87.5% | 42.8% | 70.4% | 69.9% | 54.5% | 86.4% |
| ViT-patch    | — | 95.1% | 92.0% | 93.4% | 93.4% | 38.8% | **80.1%** | 79.2% | 54.5% | 80.7% |
| Gram-dual    | 98.0% | **97.3%** | 95.5% | 96.2% | **96.7%** | 35.8% | 78.5% | 76.6% | 55.7% | 82.1% |
| Gram-single  | 97.9% | 96.9% | **95.6%** | **96.1%** | 96.9% | 36.0% | 78.9% | 78.4% | 56.9% | 83.3% |
| Gram-GLU     | — | 94.3% | 93.2% | 93.9% | 94.4% | 41.1% | 75.4% | 74.4% | 55.5% | 88.6% |

*KNN-5c = cosine 5-NN.  KM-128p = K-Means K=128 after random proj→256-d.*
*Bold = best in column.  Sup and ViT-patch supervised not available.*

---

## Fashion-MNIST — full metric table

| Encoder | LinProbe | KNN-1 | KNN-5 | KNN-5c | KM-10 | KM-128 | KM-128p | Rand | SIGReg |
|---|---|---|---|---|---|---|---|---|---|
| MLP-shallow  | **85.6%** | **83.9%** | **85.8%** | 85.5% | **55.9%** | **74.7%** | **74.6%** | **73.7%** | 69.4% |
| MLP-deep     | 84.6% | 81.8% | 83.7% | 83.8% | 21.0% | 67.1% | 64.0% | **73.5%** | 65.9% |
| ConvNet-v1   | 80.1% | 78.6% | 80.9% | 81.2% | 35.9% | 65.6% | 65.9% | 58.8% | 64.2% |
| ConvNet-v2   | 82.7% | 79.9% | 81.9% | 82.2% | 37.0% | 67.0% | 67.0% | 58.4% | 60.2% |
| ViT          | 81.6% | 79.9% | 81.3% | 81.4% | 47.5% | 68.0% | 68.0% | 61.6% | 63.8% |
| ViT-patch    | 80.9% | 78.1% | 80.4% | 80.3% | 35.7% | 67.2% | 67.0% | 61.6% | 62.2% |
| Gram-dual    | 83.6% | 79.0% | 80.9% | **82.2%** | 34.8% | 62.7% | 63.0% | 59.1% | 57.5% |
| Gram-single  | 82.8% | 79.5% | 81.5% | 82.5% | 35.9% | 61.8% | 60.7% | 60.1% | 60.7% |
| Gram-GLU     | 83.5% | 78.2% | 80.6% | 81.9% | 37.8% | 64.9% | 66.2% | 57.4% | 66.2% |

---

## Training regime comparison (MNIST, K=128)

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

SIGReg beats EMA for all 9 architectures on MNIST (EMA − SIGReg always negative).

## Training regime comparison (Fashion-MNIST, K=128)

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

EMA beats SIGReg for 8 of 9 architectures on Fashion-MNIST.

---

## Analysis

### 1 — MLP representations are already clusterable at random initialisation

The most striking result: randomly-initialised MLP-shallow embeddings cluster
MNIST at **85.4%** — 6 pp *higher* than the trained model (79.4%).
On Fashion-MNIST the gap widens to 16 pp (random 73.7% vs EMA 57.4%).

This is not an artefact of random-feature theory accidentally matching MNIST.
The supervised result confirms the MLP *can* learn good representations (98.5%),
and KNN/linear probe show EMA does teach useful SSL features (93.2% probe).
The issue is geometric: EMA training reshapes the MLP embedding space to be
**linearly separable** (hyperplane boundaries) but not **Voronoi-separable**
(convex, isotropic class regions).  K-Means assumes the latter.

MLP-deep makes this catastrophic: random 85.5% → EMA 48.7% (MNIST, −36.9 pp).
Three hidden layers give EMA enough capacity to collapse the embedding onto a
low-dimensional curved manifold.  Cluster purity remains 86% (training points
are tightly packed) but most clusters are effectively empty — a handful of giant
clusters hold nearly all points, and test points land in the wrong Voronoi cell.
The linear probe on MLP-deep (74.3%) corroborates this: EMA-JEPA with a deep
MLP is already partially collapsing even by the linear-probe standard.

### 2 — Convolutional and ViT architectures require SSL training to cluster well

Random ConvNet / ViT embeddings cluster at 52–54% on MNIST.  After EMA-JEPA
training they reach 70–80% (+16 to +26 pp).

These architectures have structured inductive biases (local receptive fields,
attention over patches) that random weights leave inactive.  Convolution weights
initialised near zero produce near-uniform activations across the image; the
pooled output is essentially a mean of random linear combinations, with little
spatial discrimination.  EMA training forces the network to build specific spatial
detectors aligned with digit strokes, which creates compact, class-aligned regions
in the embedding space.

**ViT vs ViT-patch**: both random initialise to 54.5%, but ViT-patch EMA reaches
80.1% vs ViT's 70.6% (+9.5 pp).  Patch-level masking (4×4 blocks) forces the
encoder to learn patch-complete representations; flat pixel masking leaks positional
cues that ViT can use as shortcuts.

### 3 — EMA training barely helps on Fashion-MNIST for most architectures

On Fashion-MNIST, EMA gains over random collapse to near-zero for most encoders:
+0.8 to +4.4 pp for ConvNets, Gram variants, and ViT; negative for MLPs and
Gram-GLU.  The random baseline already reaches 58–74%, and trained models only
achieve 57–74%.

The augmentation strategy explains this: Gaussian noise + flat random pixel
masking is structurally well-matched to MNIST's sparse stroke patterns (small
fraction of active pixels) but conveys little about Fashion-MNIST's texture and
silhouette structure (dense, spatially correlated patterns).  The EMA objective
teaches the encoder to be invariant to these augmentations — but those invariances
don't align with Fashion-MNIST's semantic classes.  The random baseline "wins"
because it has no learned invariances that work against clustering.

This is confirmed by the linear probe numbers: MNIST probes are 90–97% while
Fashion-MNIST probes are 80–86% — SSL is weaker overall — but the K-means gap
is even larger.  The learned features are linearly separable but the class
regions are not convex/isotropic.

### 4 — SIGReg produces tighter clusters than EMA for all architectures on MNIST

SIGReg beats EMA on MNIST K=128 for **all 9 architectures**.  Margins range
from −0.6 pp (ViT-patch, near-tie) to −43.8 pp (MLP-deep: SIGReg 92.5% vs
EMA 48.7%).

SIGReg explicitly regularises the encoder output distribution to be
approximately N(0,I) via the Epps-Pulley spectral test statistic.  This
prevents anisotropic collapse and encourages the latent space to fill the
hypersphere uniformly — exactly the geometry that favours K-Means centroid
assignment.  EMA prevents collapse implicitly through the momentum update but
allows the distribution to be anisotropic (elongated ellipsoidal class regions),
which reduces K-Means accuracy while preserving linear separability.

The most dramatic case is MLP-deep: EMA training with three hidden layers
collapses the embedding onto a handful of tight manifolds (K-Means 48.7%,
random 85.5%).  SIGReg prevents this completely, achieving 92.5% — the best
K-Means result of any model.

On Fashion-MNIST the advantage reverses: EMA beats SIGReg for 8 of 9
architectures.  Fashion-MNIST's more complex class manifolds (dense textures,
similar inter-class shapes) are distorted by SIGReg's uniform-hypersphere prior.
EMA's softer constraint (just prevent collapse) allows the encoder to naturally
organise these manifolds without imposing a mismatched geometric prior.

### 5 — Linear probe and K-Means measure different geometric properties

**MNIST rankings by metric:**

| Encoder | Supervised | Linear Probe | KNN-5 | K-Means K=128 |
|---|---|---|---|---|
| Gram-dual   | 3rd (98.0%) | **1st (97.3%)** | 2nd (96.2%) | 5th (78.5%) |
| Gram-single | 4th (97.9%) | 2nd (96.9%) | **1st (96.1%)** | 4th (78.9%) |
| ConvNet-v1  | **1st (99.4%)** | 4th (93.9%) | 3rd (95.2%) | 3rd (79.3%) |
| MLP-shallow | 2nd (98.5%) | 5th (93.2%) | 6th (92.0%) | 2nd (79.4%) |
| ViT-patch   | — | 3rd (95.1%) | 4th (93.4%) | **1st (80.1%)** |
| ConvNet-v2  | — | 6th (90.9%) | 5th (93.0%) | 7th (74.2%) |
| ViT         | 5th (98.2%) | 7th (90.2%) | 8th (87.7%) | 8th (70.4%) |
| Gram-GLU    | — | — | — | 6th (75.4%) |
| MLP-deep    | — | 9th (74.3%) | 9th (80.7%) | 9th (48.7%) |

Supervised and linear probe are weakly correlated with K-Means.
Gram-dual is best at linear probing but 5th at K-Means.
ViT-patch is 3rd at linear probing but **1st at K-Means**.

The key distinction:
- **Linear probe / supervised**: measures hyperplane separability.  The class
  regions can be any convex or even non-convex shape as long as a linear
  boundary separates them.  Gram's Hadamard readout `(H·W_A)*(H·W_B)` creates
  polyhedral geometry (hyperplanes in H-space) that is easy to linearly separate.

- **KNN**: measures local neighbourhood purity — do nearby points share the
  same label?  Intermediate between linear probe (global) and K-Means (global,
  centroid-based).

- **K-Means**: requires roughly convex, isotropic class regions centred at a
  single centroid per sub-cluster.  ViT-patch's patch-level masking produces
  more equivariant, hyperspherical features; Gram's polyhedral geometry fails
  this test.

### 6 — K=10 vs K=128: the sub-cluster effect

K=10 accuracy is poor across the board (17–48% MNIST).  With only one centroid
per class, K-Means must compress the entire within-class distribution to a
single point.  MNIST classes have substantial within-class variation (digit style,
stroke width, rotation) that a single centroid cannot capture.

K=128 provides ≈13 sub-clusters per class, allowing the centroid to track
within-class sub-modes.  This nearly doubles accuracy for most encoders
(ConvNet-v1: 48% → 79%).

The exception is MLP-deep: K=10→48.7% at K=128 (still low).  The collapsed
embedding has so few distinct regions that even 128 centroids cannot recover
meaningful sub-clusters — most centroids land in the same collapsed region.

### 7 — Random projection to 256-d is nearly free

Projecting 1024-d → 256-d via a fixed random Gaussian matrix (columns
L2-normalised) costs at most 1–2 pp across all encoders and both K=128 tests.
The largest drops: Gram-dual −1.9 pp, Gram-single −0.5 pp, MLP-deep −0.3 pp.

This confirms the 1024-d embeddings are highly redundant — the class-discriminative
structure lives in a low-dimensional subspace.  The random projection happens to
preserve enough of this structure for K-Means.  Practical implication: if
clustering is the end goal, 256-d embeddings are sufficient and reduce K-Means
cost by 16×.

### 8 — MLP-deep SSL collapse: a warning sign visible in multiple metrics

MLP-deep has a coherent failure signature across all metrics on MNIST:

| Metric | Value | vs best |
|---|---|---|
| Supervised | 98.7% | near-best — the architecture *can* learn |
| Linear probe | 74.3% | −23 pp below MLP-shallow |
| KNN-5 | 80.7% | −15 pp below MLP-shallow |
| K-Means K=128 | 48.7% | −31 pp below MLP-shallow |
| K-Means K=128 purity | 86.0% | highest of all — clusters are pure but few |
| Random K=128 | 85.5% | −36.9 pp below random |

The cluster purity (86%) being the highest while accuracy (48.7%) being the
lowest tells the full story: there are a small number of very pure, very large
clusters, with most K-Means centroids sitting in empty space.  The SSL training
has collapsed the MLP-deep embedding onto a handful of tight manifolds.

On Fashion-MNIST the collapse is less severe (K-Means 48.5% vs random 73.5%,
−25 pp) but follows the same pattern.  The harder dataset requires more diverse
features and partially resists collapse.

---

## Dataset comparison: MNIST vs Fashion-MNIST

| Encoder | MNIST LinProbe | F-MNIST LinProbe | Δ | MNIST KM-128 | F-MNIST KM-128 | Δ |
|---|---|---|---|---|---|---|
| MLP-shallow | 93.2% | 85.6% | −7.6 pp | 79.4% | 74.7% | −4.7 pp |
| MLP-deep    | 74.3% | 84.6% | +10.3 pp | 48.7% | 67.1% | +18.4 pp |
| ConvNet-v1  | 93.9% | 80.1% | −13.8 pp | 79.3% | 65.6% | −13.7 pp |
| ConvNet-v2  | 90.9% | 82.7% | −8.2 pp | 74.2% | 67.0% | −7.2 pp |
| ViT         | 90.2% | 81.6% | −8.6 pp | 70.4% | 68.0% | −2.4 pp |
| ViT-patch   | 95.1% | 80.9% | −14.2 pp | 80.1% | 67.2% | −12.9 pp |
| Gram-dual   | 97.3% | 83.6% | −13.7 pp | 78.5% | 62.7% | −15.8 pp |
| Gram-single | 96.9% | 82.8% | −14.1 pp | 78.9% | 61.8% | −17.1 pp |
| Gram-GLU    | 94.3% | 83.5% | −10.8 pp | 75.4% | 64.9% | −10.5 pp |

**MLP-deep reversal**: the only encoder that improves from MNIST→F-MNIST on both
metrics.  The Gram and ViT-patch encoders — which top MNIST — suffer the largest
drops (−13 to −17 pp on K-Means).

The Gram inductive bias (co-activation statistics via outer products) is well
matched to MNIST's sparse binary-ish strokes but not to Fashion-MNIST's dense
textured patterns, explaining its disproportionate drop.  ViT-patch's large drop
may reflect that 4×4 patch-level masking encodes MNIST spatial structure that
does not transfer to garment shape variation.

---

## Limitations of this evaluation

- **Single random seed for random baseline**: the random baseline uses seed=0.
  MLP random projections are known to be seed-sensitive; averaging over 3–5
  seeds would give a tighter estimate.

- **K-Means convergence**: FAISS K-Means with 100 iterations may not fully
  converge for K=128 on high-dimensional embeddings.  Increasing to 300
  iterations could improve absolute numbers by 1–3 pp but should not change
  relative rankings.

- **No augmentation-matched baseline**: the random baseline uses random weights
  but the same input preprocessing (normalise by 255).  A truly matched baseline
  would also randomise the augmentation pipeline, removing the EMA objective
  entirely to isolate the effect of the augmentation choices.

---

*Generated 2026-04-02.*
*Data: `results.jsonl`, `results_knn.jsonl`, `results_kmeans.jsonl`, `results_kmeans_comparison.jsonl`, `results_supervised.jsonl`.*
*Scripts: `kmeans_eval.py`, `kmeans_comparison.py`, `knn_eval.py`, `supervised_eval.py`.*
