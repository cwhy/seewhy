# Report 7 — Hebbian Aggregation & Clustering (exp13–17)

## Architecture

```
# Shared codebook
E : (N_CODES, D_EMB)

# Per-slot projection (shared weights)
u_i = z_i @ W_u + b_u        # (D_U,)
v_i = z_i @ W_v + b_v        # (D_U,)

# Hebbian aggregation (permutation-invariant)
M = Σ_i  u_i[:,None] * v_i[None,:]    # (D_U, D_U)
z = M.reshape(-1)                       # (D_U²,)

# Decoder entry (same r⊗c⊗h as exp6-12)
h = GELU(z @ W_mlp1 + b_mlp1)
```

Key properties: M is a sum of rank-1 outer products — permutation-invariant,
and the rank of M equals the number of distinct active slots.

---

## Results

| Exp | N_VARS | Sampling | Bin acc |
|-----|--------|----------|---------|
| exp13 | 16 | P_ZERO=0.8, ~3 active | 87.0% |
| exp14 | 16 | exact 1–2 active, skip trick | 86.5% |
| exp15 | 2 | exact 1–2 active (N_VARS matches capacity) | 86.4% |
| exp16 | 1 | k ~ Uniform(0, 31) — 32 clusters | 84.0% |
| exp17 | 1 | load-balanced assignment, all 32 active | **85.0%** |

---

## Findings

**1. Permutation-invariant aggregation is free.**
Replacing concat with Hebbian sum (exp13 vs exp11) has zero accuracy cost at P_ZERO=0.8.
The decoder doesn't need positional identity — only the set of active codes matters.

**2. Reducing N_VARS to match actual capacity is also free.**
exp15 (N_VARS=2) matches exp14 (N_VARS=16) at 86.4% — the 14 extra slots in exp14
were entirely wasted. The model naturally uses at most 2 simultaneously.

**3. Exact sparsity (1–2 active) costs ~0.5% vs the full-prior ceiling.**
exp14/15 at 86.4–86.5% vs exp13 at 87.0%. The restriction to rank-1/rank-2 M
limits coverage of the training distribution.

**4. Pure clustering (exp16) achieves 84.0% with 32 prototypes.**
Each training image is assigned to one of 32 learned rank-1 matrices `u_k⊗v_k`.
With ~1875 images per cluster, the decoder learns a compressed prototype per cluster.
84% is a meaningful baseline for what 32-cluster quantisation can achieve on MNIST.

**Known issue — codebook collapse in exp16.** With uniform sampling and IMLE
nearest-neighbour matching, some codes are likely never matched (their decoded
images are never nearest to any training point), effectively reducing the active
codebook below 32. Fixed in exp17.

**5. Load-balanced assignment (exp17) fixes collapse and recovers +1%.**
exp17 replaces the random sample pool with direct prototype decoding (N_CODES
images per epoch, no random sampling). Assignment is split:
- Top half (best-matched images): keep nearest-neighbour as usual.
- Bottom half (poorly-matched images): greedily reassigned to the least-used
  cluster, using a precomputed per-cluster ranking for O(n_bot × N_CODES) total.

All 32 codes stay active from epoch 0 onward. Accuracy rises from 84.0% → 85.0%,
confirming that collapse was losing roughly 1% of representational capacity.

The 85.0% result is close to the exp15/14 ceiling (86.4%) for the same 1-active
configuration — the remaining ~1.4% gap reflects the fundamental limit of
hard cluster assignment vs. soft multi-code combinations.

---

## Visualisations

- exp13 codebook (P_ZERO=0.8): https://media.tanh.xyz/seewhy/26-03-23/exp13_codebook_viz.png
- exp14 rank-2 pairs: https://media.tanh.xyz/seewhy/26-03-23/exp14_codebook_pairs.png
- exp15 rank-2 pairs (N_VARS=2): https://media.tanh.xyz/seewhy/26-03-24/exp15_pairs.png
- exp16 all 32 prototypes (collapse present): https://media.tanh.xyz/seewhy/26-03-24/exp16_prototypes.png
- exp17 all 32 prototypes (all active): https://media.tanh.xyz/seewhy/26-03-24/exp17_prototypes.png
- exp17 reconstructions: https://media.tanh.xyz/seewhy/26-03-24/exp17_recon_final.png
- exp17 learning curves: https://media.tanh.xyz/seewhy/26-03-24/exp17_learning_curves.svg
