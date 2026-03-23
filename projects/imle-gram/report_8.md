# Report 8 — Clustering Experiments (exp16–exp17c)

## Setup

Pure clustering: N_VARS=1, a single Hebbian rank-1 matrix M = u_k ⊗ v_k per image.
Each training image is assigned to one of N_CODES learned cluster prototypes.

Architecture identical to exp13–15 (Hebbian aggregation) with N_VARS=1:

```
z_k  = codebook[k]                       # (D_EMB=16,)
u_k  = z_k @ W_u + b_u                  # (D_U=16,)
v_k  = z_k @ W_v + b_v                  # (D_U=16,)
M    = u_k[:, None] * v_k[None, :]      # (16, 16) — rank-1 matrix
z    = M.reshape(-1)                     # (256,)
h    = GELU(z @ W_mlp1 + b_mlp1)       # (64,)
# → r⊗c⊗h pixel decoder
```

167,440 parameters throughout.

---

## Experiments

| Exp | N_CODES | Assignment | Active | Bin acc | lbl_hard |
|-----|---------|-----------|--------|---------|----------|
| exp16  | 32 | Nearest-neighbour (random pool 1000) | ~16/32 | 84.0% | — |
| exp17  | 32 | Load-balanced (median split) | **32/32** | 85.0% | — |
| exp17b | 50 | Load-balanced (median split) | **50/50** | **85.2%** | 79.7% |
| exp17c | 50 | Load-balanced (mean split)   | **50/50** | **85.3%** | 79.1% |

---

## Codebook Collapse (exp16)

With exp16's random-sampling + IMLE nearest-neighbour matching, roughly half
the 32 clusters receive no gradient. The random pool of 1000 samples is too
small: a cluster's prototype must happen to be the nearest of all 1000 pool
candidates for a given image. Prototypes starting far from training data never
win → no gradient → stay far → collapse.

## Load-Balanced Assignment (exp17–17c)

Replaces the random pool with direct prototype decoding (one image per cluster
decoded per epoch) and splits each batch assignment in two phases:

**Phase 1 — well-covered images (top group):** keep standard nearest-neighbour.

**Phase 2 — poorly-covered images (bottom group):** greedily reassign to
least-used clusters. The least-used cluster picks its closest available
(unassigned) image from the bottom group. Repeat until all bottom images
are assigned.

```python
# O(N × N_CODES) distance matrix + O(n_bot × N_CODES) pointer steps
rank = np.argsort(bot_dists, axis=0)      # per-cluster ranking of bottom images
ptr  = np.zeros(N_CODES, dtype=int)
for _ in range(n_bot):
    c = argmin(counts)
    local = rank[ptr[c], c]               # best unassigned image for c
    assignment[bot_idx[local]] = c
    counts[c] += 1
```

**Median split (exp17/17b):** top = images with nn_dist below the median
(exactly N//2 images).

**Mean split (exp17c):** top = images with nn_dist below the mean. The top
group size floats per batch — fewer images get load-balanced on batches where
the distance distribution is right-skewed (most are well-matched). Results are
essentially identical to median split (~0.1% difference).

**N_CODES 32→50 (exp17b/17c):** +0.2% bin_acc and +4.7% hard label accuracy
over exp17 (32 clusters). More clusters = purer clusters = better label
separability.

---

## Label Accuracy Analysis

Each cluster is evaluated as a classifier by mapping it to a label via test
set assignment statistics.

### Formulation

Let c_i = nearest prototype for test image i. For each cluster c:

```
P(y=k | c) = count(test images assigned to c with label k) / count(c)
```

**Hard accuracy** — majority-vote label per cluster:

```
label(c) = argmax_k P(y=k | c)
hard_acc = mean_i [ label(c_i) == y_i ]
```

**Bayes accuracy** — soft cluster assignment weighted label prediction:

```
P(c | x) = softmax(-dist(x, proto_c) / mean_dist)   over all c
P(y | x) = sum_c P(c|x) * P(y|c)
bayes_acc = mean_i [ argmax_y P(y|y_i) == y_i ]
```

Hard (79.7%) consistently outperforms Bayes (~51–57%) because the softmax
spreads weight across dissimilar clusters in high-dimensional pixel space,
diluting the label signal. Hard nearest-prototype is the better clustering
classifier here.

### Results (exp17b, final epoch)

| Metric | Value |
|--------|-------|
| Hard label acc | 79.7% |
| Bayes label acc | 51.2% |
| Bin acc (pixel) | 85.2% |

---

## Cluster Usage

The cumulative assignment line plot shows all 50 clusters receive consistent
coverage from epoch 0. Lines are coloured by majority label (tab10 colormap —
one colour per digit), so clusters capturing the same digit class appear as
a bundle. Parallel bundles indicate the model has learned multiple prototypes
per digit, corresponding to style/orientation variants.

---

## Visualisations

### exp17b (median split, N_CODES=50)
- Cumulative usage (coloured by majority label): https://media.tanh.xyz/seewhy/26-03-24/exp17b_cluster_usage_lineplot.png
- Label accuracy curves: https://media.tanh.xyz/seewhy/26-03-24/exp17b_label_accuracy.png
- Label distributions + reconstruction overlay: https://media.tanh.xyz/seewhy/26-03-24/exp17b_cluster_label_dist.png
- Prototype reconstructions: https://media.tanh.xyz/seewhy/26-03-24/exp17b_cluster_reconstructions.png

### exp17c (mean split, N_CODES=50)
- Cumulative usage: https://media.tanh.xyz/seewhy/26-03-24/exp17c_cluster_usage_lineplot.png
- Label accuracy curves: https://media.tanh.xyz/seewhy/26-03-24/exp17c_label_accuracy.png
- Label distributions + reconstruction overlay: https://media.tanh.xyz/seewhy/26-03-24/exp17c_cluster_label_dist.png
- Prototype reconstructions: https://media.tanh.xyz/seewhy/26-03-24/exp17c_cluster_reconstructions.png

### Earlier clustering experiments
- exp17 all 32 prototypes: https://media.tanh.xyz/seewhy/26-03-24/exp17_prototypes.png
- exp17 reconstructions: https://media.tanh.xyz/seewhy/26-03-24/exp17_recon_final.png
- exp16 all 32 prototypes (collapse visible): https://media.tanh.xyz/seewhy/26-03-24/exp16_prototypes.png

---

## Context: Accuracy Ceiling

| Prior / architecture | Bin acc |
|---------------------|---------|
| exp13: Hebbian, N_VARS=16, P_ZERO=0.8 | 87.0% |
| exp15: Hebbian, N_VARS=2, 1–2 active | 86.4% |
| exp17c: clustering, N_CODES=50, mean split | **85.3%** |
| exp17b: clustering, N_CODES=50, median split | 85.2% |
| exp17: clustering, N_CODES=32, load-balanced | 85.0% |
| exp16: clustering, N_CODES=32, random pool | 84.0% |

The ~1.4% gap between 50-cluster hard assignment and the 1-active soft
combination (exp15) reflects the decoder's smoother gradient signal under
soft codebook lookup vs. hard cluster assignment.
