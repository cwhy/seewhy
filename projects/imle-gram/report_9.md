# Report 9 — Scaling Clusters and Factored Codebooks (exp18–exp23)

## Overview

This report covers six experiments extending the load-balanced clustering work
from Report 8. The main questions are:

1. Can factored (multi-variable) codebooks match flat clustering at equal capacity?
2. Does the new delta_sum assignment improve over mean-split load-balancing?
3. How does accuracy scale with cluster count (50 → 128 → 512)?

---

## Architecture Recap

All experiments share the same decoder:

```
z_k  = codebook[k]             # (D_EMB=16,) per cluster
u_k  = z_k @ W_u + b_u        # (D_U=16,)
v_k  = z_k @ W_v + b_v        # (D_U=16,)
M    = Σ_k u_k ⊗ v_k          # (16,16) Gram matrix — sum of rank-1 outer products
h    = GELU(M.reshape(-1) @ W_mlp1 + b_mlp1)   # (64,)
# → row⊗col⊗h pixel decoder → 784×32 bin logits
```

For N_VARS > 1, the Gram matrix sums rank-1 contributions from each variable's
codebook lookup. For N_VARS = 1, it reduces to a single outer product M = u ⊗ v.

---

## Experiment Summary

| Exp | N_CODES | Vars | Assignment | bin_acc | lbl_hard | time | params |
|-----|---------|------|------------|---------|----------|------|--------|
| exp17c _(baseline)_ | 50 | 1 | mean-split | 85.3% | 79.1% | 434s | 167,728 |
| exp18  | 49  | 1 (flat)     | mean-split   | 85.2% | 79.7% | 432s | 167,712 |
| exp19  | 49  | 2 (7×7 shared) | mean-split | 84.8% | 74.1% | 430s | 167,040 |
| exp19b | 49  | 2 (7×7 indiv)  | mean-split | 85.2% | 78.9% | 433s | 167,152 |
| exp20  | 128 | 1 (flat)     | mean-split   | 85.6% | 83.7% | 441s | 168,976 |
| exp21  | 128 | 1 (flat)     | delta_sum    | **85.8%** | **87.0%** | 457s | 168,976 |
| exp22  | 512 | 3 (8×8×8 indiv) | delta_sum | 86.4% | 90.0% | 3094s | 167,312 |
| exp23  | 512 | 1 (flat)     | delta_sum    | **86.5%** | **91.1%** | 3011s | 175,120 |

All experiments: MNIST, 100 epochs, D_EMB=16, D_U=16, MLP_HIDDEN=64, LR=3e-4 cosine.

---

## Part 1: Flat vs Factored at 49 Capacity (exp18/19/19b)

### exp18 — Flat 49

Direct replication of exp17c with N_CODES=49 (≈ 7²). Essentially identical
to exp17c: 85.2% bin_acc, 79.7% lbl_hard.

### exp19 — Shared 7×7 Codebook

One codebook of 7 embeddings shared across two variables. Each image maps to
a pair (i, j): `M = u_i⊗v_i + u_j⊗v_j` where both i and j index the same
codebook.

Result: **only 84.8% bin_acc, 74.1% lbl_hard** — worse than flat.

**Why**: the shared codebook introduces symmetry. Slots 0 and 1 draw from the
same 7 embeddings with the same W_u, W_v projections, so `(i, j)` and `(j, i)`
produce identical Gram matrices. Of the 49 nominal combinations only ~28 are
geometrically distinct. The model is effectively underpowered — it has 49
cluster slots but only ~28 unique prototypes.

### exp19b — Individual 7×7 Codebooks

Separate `codebook_0` and `codebook_1`, each of size 7. Now all 49 combinations
are fully distinct: code i in slot 0 is a different embedding from code i in
slot 1.

Result: **85.2% bin_acc, 78.9% lbl_hard** — matches flat exp18.

This confirms: with equal capacity and individual codebooks, the factored
representation does not hurt. The shared codebook's symmetry was the culprit.

---

## Part 2: Delta-Sum Assignment (exp20 vs exp21)

### Mean-Split Baseline (exp20, N_CODES=128)

Top half of images (below-mean distance to NN) keep their nearest-neighbour
assignment. Bottom half are load-balanced via a least-used-cluster greedy scan.
Result: 85.6% bin_acc, 83.7% lbl_hard.

### Delta-Sum Assignment (exp21, N_CODES=128)

Replaces the top/bottom split with a unified priority-queue approach.

**Algorithm**: each cluster maintains an accumulated assignment cost `delta_sum[c]`.
The effective cost of assigning image i to cluster c is:

```
cost(i, c) = delta_sum[c] + dist(i, c)
```

At each step, the globally cheapest (image, cluster) pair is selected. After
assignment, `delta_sum[c] += dist(i, c)`. This means a cluster that already
received many cheap assignments will have high delta_sum and will be outbid by
under-used clusters on future steps — natural load-balancing emerges without
any explicit top/bottom partition.

**Optimised implementation** (O(N·N_CODES) instead of O(N²·N_CODES)):

```python
rank = np.argsort(dists, axis=0)   # sort images per cluster once
ptr  = np.zeros(N_CODES)           # per-cluster pointer to best unassigned image
for _ in range(N):
    cur_imgs  = rank[ptr, cidx]                   # one candidate per cluster
    cur_costs = delta_sum + dists[cur_imgs, cidx] # N_CODES values only
    c   = argmin(cur_costs)
    img = cur_imgs[c]
    assignment[img] = c
    delta_sum[c] += dists[img, c]
    advance ptrs for clusters that pointed to img
```

Result: **85.8% bin_acc (+0.2%), 87.0% lbl_hard (+3.3%)** vs mean-split.

The improvement in label accuracy is larger than in bin_acc, suggesting that
delta_sum achieves more semantically consistent cluster assignments — each
prototype captures a tighter region of image space.

---

## Part 3: Scaling to 512 — Flat vs Factored (exp22/23)

### exp22 — 8×8×8 Factored (N_VARS=3)

Three individual codebooks of 8 each. 8³ = 512 combination prototypes.
Only 3×8 = 24 codebook entries; the decoder capacity is the same as exp23
but the latent space is structured.

**Flat index**: `flat = i × 64 + j × 8 + k`

Assignment: delta_sum with N_COMBOS=512.

Result: **86.4% bin_acc, 90.0% lbl_hard**, 3094s.

### exp23 — Flat 512

512 independent codebook entries. Same decoder, same delta_sum assignment.

Result: **86.5% bin_acc, 91.1% lbl_hard**, 3011s.

### Comparison

At 512 capacity, flat and 8×8×8 factored are essentially tied on bin_acc
(0.1% gap). The flat model has marginally better label accuracy (+1.1%).

The factored model uses only 24 codebook entries to represent 512 prototypes.
Each prototype is a composition of 3 independently-learned style axes. The
similar accuracy suggests the Gram matrix (`M = u_i⊗v_i + u_j⊗v_j + u_k⊗v_k`)
can represent the same diversity as 512 independent rank-1 matrices, but the
flat model's 512 free codebook slots give a slight label-purity advantage.

**Timing**: both take ~50 minutes — the bottleneck is the delta_sum assignment
inner loop over 512 clusters per batch (vs ~7s per epoch for N_CODES=128).
The 3× longer assignment per step is the main cost.

---

## Scaling Summary: Label Accuracy vs N_CODES

| N_CODES | Assignment | lbl_hard |
|---------|------------|----------|
| 50 | mean-split | 79.1% |
| 128 | mean-split | 83.7% |
| 128 | delta_sum  | 87.0% |
| 512 (8×8×8) | delta_sum | 90.0% |
| 512 (flat)  | delta_sum | 91.1% |

Label accuracy scales consistently with cluster count. More clusters = smaller,
purer clusters = less label mixing. Delta_sum adds ~3% over mean-split at equal
N_CODES.

---

## Visualisations

| Exp | Clustering analysis |
|-----|---------------------|
| exp21 (128 flat, delta_sum) | https://media.tanh.xyz/seewhy/26-03-25/exp21_clustering.html |
| exp22 (8×8×8 factored) | https://media.tanh.xyz/seewhy/26-03-25/exp22_clustering.html |
| exp23 (512 flat, delta_sum) | https://media.tanh.xyz/seewhy/26-03-25/exp23_clustering.html |

Each page includes: per-epoch cluster usage (coloured by majority label), loss
curve, label accuracy curves, per-cluster label distribution grid (click-through
to all 128/512 clusters), and prototype reconstruction grid sorted by majority
label.

---

## Next Steps

- Fashion-MNIST: replicate best config (512-flat delta_sum, 128-flat delta_sum)
  to test whether the clustering approach generalises to harder 10-class data
  with more within-class variation.
- Larger decoder: current bottleneck may be decoder capacity (MLP_HIDDEN=64),
  not cluster count.
