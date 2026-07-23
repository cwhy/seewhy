# Report 10 — Fashion-MNIST and Codebook Diversity (exp24–exp33)

## Overview

This report covers ten experiments across two themes:

1. **Fashion-MNIST transfer** (exp24–exp25): applying the best MNIST clustering
   configuration to a harder dataset.
2. **Codebook diversity** (exp26–exp33): a series of attempts to reduce prototype
   redundancy — clusters that produce visually similar reconstructions.

Baseline throughout: exp21 (128-cluster flat delta_sum, MNIST, 85.8% bin_acc,
86.6% lbl_hard).

---

## Part 1 — Fashion-MNIST (exp24–exp25)

Fashion-MNIST is a drop-in replacement for MNIST with 10 clothing categories.
exp24 clones exp21 (128-cluster), exp25 clones exp23 (512-cluster), with only
the dataset loader swapped.

| Exp | N_CODES | bin_acc | lbl_hard | time |
|-----|---------|---------|----------|------|
| exp24 | 128 | 56.4% | 73.5% | 847s |
| exp25 | 512 | 57.0% | 79.0% | 3064s |

bin_acc drops sharply relative to MNIST (~85% → ~57%) — Fashion-MNIST pixel
distributions are more complex and harder to reconstruct with a Gram decoder.
lbl_hard stays reasonable (73–79%), meaning the clustering still captures
semantic structure even when per-pixel reconstruction is imperfect.

The 512-cluster model improves lbl_hard by 5.5pp over 128-cluster at 3.6× the
compute cost, consistent with the MNIST scaling trend from Report 9.

---

## Part 2 — Codebook Diversity Experiments (exp26–exp33)

### Motivation

The correlation heatmap (N_CODES×N_CODES cosine similarity of prototype images,
sorted by majority label) revealed blocks of near-duplicate prototypes — clusters
assigned to the same digit class with visually similar reconstructions. The goal
was to reduce this redundancy organically.

### exp26 — Cosine Repulsion Loss

Added a codebook repulsion penalty to the training loss:

```
L_rep = mean(G_ij²)  for i≠j,  G = Z_n @ Z_n.T  (normalised cosine Gram)
REP_LAMBDA = 0.1
```

**Result:** 85.9% bin_acc, 87.1% lbl_hard — no meaningful change from baseline.
The repulsion loss operates in embedding space; two embeddings can be orthogonal
while still decoding to visually similar prototypes. The heatmap showed negligible
improvement.

### exp27 — Merge+Reinit (cosine threshold)

After each epoch, check off-diagonal cosine similarity; if any pair exceeds
MERGE_THRESHOLD=0.9, discard the lighter cluster (by assignment count) and
reinitialise from scratch. Zero Adam mu/nu for reinited rows.

**Result:** 85.8% bin_acc, 86.6% lbl_hard — threshold never triggered.
Diagnostic (`check_cosine_sim.py`) showed max off-diagonal cosine sim = 0.85,
below the threshold throughout training. The codebook is naturally less collapsed
than expected in embedding space.

### exp28 — Newtonian Gravity (embedding space)

Applied Newtonian gravity between all codebook pairs after each gradient step.
Mass = fraction of epoch assignments (data-dense clusters attract more strongly).
Pairs pulled within MERGE_DIST=0.5 in embedding space are merged and the lighter
cluster is reinitialised.

```
F_ij = G · mass_i · mass_j · (z_j − z_i) / ||z_j − z_i||³
GRAVITY_G = 500  (calibrated from check_embed_dist.py)
```

**Result (second run):** 85.4% bin_acc, 77.5% lbl_hard — 9pp regression on
lbl_hard. Gravity was very active (grav_disp ~0.1–0.2/epoch, 8–13 merges/epoch
throughout all 100 epochs). The constant churn of ~10 reinits per epoch prevents
clusters from converging to stable identities.

### exp29 — Gravity + Pixel-Space Merge Threshold

Same as exp28 but the merge decision uses decoded prototype distances in pixel
space (784-dim, values in raw intensity range) rather than embedding-space L2.
Motivation: two embeddings can be far apart while decoding to similar images.

First attempt used MERGE_DIST_PIX=2.0 — completely wrong scale. Diagnostic
(`check_proto_dist.py`) revealed pixel-space L2 distances range 56–2520, mean
1510. The threshold of 2.0 fired zero merges.

**Result:** 85.6% bin_acc, 83.4% lbl_hard — essentially baseline (no merges,
gravity acted but MERGE_DIST_PIX never triggered).

### exp30 — Calibrated Pixel-Space Threshold

Raised MERGE_DIST_PIX to 200 (between min=56 and p1=488 from check_proto_dist).
Result: 25–63 merges/epoch in epochs 0–9 (unconverged prototypes are noisy),
tapering to 1–5/epoch by epoch 20+.

**Result:** 85.4% bin_acc, 78.9% lbl_hard — regression driven by early churn.
lbl_bayes recovered to 59.5% by epoch 99 but lbl_hard never recovered fully.

### exp31 — Warmup + Tighter Threshold

Added MERGE_WARMUP=20 (skip gravity+merge for first 20 epochs) and tightened
MERGE_DIST_PIX=100.

**Diagnosis of exp31:** At epoch 20, gravity suddenly activates on converged
clusters and fires 23 merges immediately (warm start causes a burst). After that,
gravity keeps grav_disp at 0.1–0.3 indefinitely, match_l2 stays elevated at
~3000 (vs ~2500 baseline), and lbl_hard fluctuates.

Root cause identified: **gravity fights gradient learning**. The gradient pulls
each cluster toward its assigned data; gravity pulls it toward other clusters.
These opposing forces prevent convergence.

**Result:** 85.2% bin_acc, 76.7% lbl_hard — worst lbl_hard of the series.

### exp32 — No Gravity, Pixel-Space Merge Only

Dropped gravity entirely. Kept warmup=20 and MERGE_DIST_PIX=100. Gradient
learning runs undisturbed; we only intervene when two prototypes genuinely
converge to the same visual output.

**Result:** 85.8% bin_acc, 86.7% lbl_hard — matches baseline exactly.
Zero merges fired: converged prototype min pixel-L2 = 411 (threshold was 100).
Clean training; match_l2 reached 2370, the best of the series.

### exp33 — Aggressive Threshold (800)

Raised MERGE_DIST_PIX to 800 (above the converged p1=701) to force merges on
visually similar pairs observed in the heatmap.

**Result:** 50+ merges/epoch from epoch 20 onwards — nearly half the clusters
reinited every epoch. 85.4% bin_acc, 79.4% lbl_hard.

The fixed absolute threshold is inherently problematic: there will always be a
fixed fraction of pairs below any threshold, producing permanent churn at a
constant rate regardless of how diverse the codebook becomes.

---

## Summary Table

| Exp | Description | bin_acc | lbl_hard | merges/epoch (post-warmup) |
|-----|-------------|---------|----------|---------------------------|
| exp21 _(baseline)_ | flat delta_sum | 85.8% | 86.6% | — |
| exp24 | Fashion-MNIST 128 | 56.4% | 73.5% | — |
| exp25 | Fashion-MNIST 512 | 57.0% | 79.0% | — |
| exp26 | cosine repulsion | 85.9% | 87.1% | — |
| exp27 | merge+reinit (cosine 0.9) | 85.8% | 86.6% | 0 |
| exp28 | gravity G=500, emb merge | 85.4% | 77.5% | ~10 |
| exp29 | gravity + pix merge (dist=2) | 85.6% | 83.4% | 0 |
| exp30 | gravity + pix merge (dist=200) | 85.4% | 78.9% | 1–50 |
| exp31 | gravity + warmup + pix merge | 85.2% | 76.7% | 1–5 |
| **exp32** | **pix merge only (dist=100)** | **85.8%** | **86.7%** | **0** |
| exp33 | pix merge only (dist=800) | 85.4% | 79.4% | ~50 |

---

## Key Findings

1. **Repulsion in embedding space has no visible effect on prototype diversity.**
   Embeddings can be far apart while decoding to similar images.

2. **Gravity fights gradient learning.** Any persistent force pulling clusters
   toward each other degrades convergence; the two objectives conflict.

3. **Prototypes are already more diverse than expected.** Converged pixel-space
   L2 distances (min=411, mean=1577) are large relative to naive expectations.
   The visual similarity in the heatmap is real but not extreme.

4. **Fixed absolute merge thresholds produce permanent churn.** There is always
   a fraction of pairs below any fixed threshold. A relative criterion (e.g.
   merge only the single closest pair per N epochs) would be more surgical.

5. **exp32 is the cleanest result.** No diversity mechanism, no performance cost,
   best match_l2 of the series (2370). The pixel-space merge framework is correct
   but needs a smarter trigger than a fixed threshold.

---

## Next Directions

- **Relative threshold**: merge only the K closest pairs per epoch (e.g. K=1 or
  K=2), regardless of absolute distance. Guarantees a bounded reinit rate.
- **Merge frequency**: apply merge check every N epochs rather than every epoch
  to give reinited clusters time to converge before re-evaluation.
- **Larger model / more epochs**: exp32's match_l2=2370 suggests room to push
  accuracy further with more capacity or training time.
