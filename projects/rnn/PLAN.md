# MNIST Linear RNN — Research Plan

## What we have

A pixel-by-pixel MNIST classifier using an **outer-product linear RNN**:

```
emb_t  = pos_emb[t] + val_emb[bin(pixel_t)]     # (d,)  learned embeddings
k_t    = W_k @ emb_t                             # (d_k,)
v_t    = W_v @ emb_t                             # (d_v,)
M_t    = M_{t-1} + k_t ⊗ v_t                    # accumulate outer products
logits = MLP(LayerNorm(vec(M_T)))
```

**Result: 95.3% test accuracy in 3 epochs, ~12 seconds on CPU.**

The state M_T = K^T V is the second-moment matrix of the embedded sequence —
it captures pairwise co-occurrences of learned features across all 784 positions.
This is equivalent to causal linear attention at the final token, or a
Hebbian fast-weight programmer with A=I.

---

## Core insight to preserve

The power of this model is that M_T is a **matrix** (not a vector).
A vector state can only accumulate weighted sums of features (first-order).
A matrix state accumulates products of features across positions (second-order) —
that is what makes classification easy.

Every planned change must preserve or improve this second-order accumulation.

---

## Plan: simplify first

The current model has more moving parts than necessary. Before adding anything,
we should find the minimal version that still works. Experiments in order:

### S1. Symmetric outer product  (k = v)

Replace the asymmetric key/value projections with a single shared projection:

```
e_t    = W @ emb_t                               # one projection instead of two
M_t    = M_{t-1} + e_t ⊗ e_t                    # symmetric: M_T = E^T E
```

M_T becomes a Gram matrix — symmetric positive semi-definite.
Half the projection parameters (W_k, W_v → W). M_T still has d² entries.

**Hypothesis:** loses <1% accuracy. A 4x4 image's Gram matrix still
separates the digit classes — asymmetry is not fundamental here.

**If it works:** removes W_v entirely, cleaner interpretation.

---

### S2. No projection at all  (k = v = emb)

Skip W entirely. Feed the raw embeddings straight into the outer product:

```
M_t    = M_{t-1} + emb_t ⊗ emb_t               # M_T = emb^T emb
logits = MLP(LayerNorm(vec(M_T)))
```

The state is now purely a function of the embeddings. All discriminative
capacity lives in pos_emb and val_emb.

**Hypothesis:** accuracy drops 1-3%. If <1% drop, this is the preferred model —
it has no projection parameters at all between embedding and state.

---

### S3. Linear readout  (remove MLP)

Replace the 2-layer MLP with a single linear layer:

```
logits = W_out @ LayerNorm(vec(M_T)) + b
```

Drops ~500K parameters. The state is 4096-dimensional, which is plenty
for a linear classifier over 10 classes.

**Hypothesis:** accuracy drops ~1-2%. If drop is ≤1%, use the linear readout —
it removes an entire layer and makes the model end-to-end interpretable:
logits are a linear function of the outer product of learned embeddings.

---

### S4. Reduce embedding dimension

Try d = 32 (state = 32×32 = 1024 instead of 4096).

**Hypothesis:** marginal drop. The rank of M_T is bounded by min(d, 784)
and we only need to separate 10 classes — d=32 gives 1024 features,
far more than necessary.

**If works at d=32:** smaller model, faster to train.

---

## One meaningful upgrade: the delta rule

If simplifications plateau below 95%, or to push toward 97-98%:

Replace the Hebbian update (always add) with the **delta rule** (add only the error):

```
# Hebbian (current):
M_t = M_{t-1} + k_t ⊗ v_t

# Delta rule:
error_t = v_t - M_{t-1} @ k_t             # what M currently predicts for k_t
M_t     = M_{t-1} + beta * error_t ⊗ k_t  # only write the surprise
```

The delta rule reduces **interference**: when two pixels map to similar keys,
the Hebbian rule corrupts their stored values. The delta rule first erases
the old value at key k_t, then writes the new one.

This is one additional matrix-vector multiply per step (M @ k), which is cheap.
Beta can be a fixed scalar (0.5) or a learned parameter.

**Why only this upgrade:** every other proposed addition from the literature
(multi-head, surprise weighting, DeltaProduct, Titans memory, gating) adds
significant complexity for marginal gain on a task we already solve at 95%.
The delta rule is the single most principled improvement — it directly
addresses the interference problem that limits the Hebbian model's capacity —
and it adds exactly one line of code.

---

## What we are NOT doing (and why)

| Idea | Why skip |
|---|---|
| Multi-head | Adds params and heads to tune; rank-64 is already sufficient for 10 classes |
| Forget gate γ_t M_{t-1} | All 784 pixels matter equally; no temporal ordering to exploit |
| Surprise weighting (Titans) | Complex; MNIST pixels are not surprising in a meaningful sense |
| DeltaProduct (multi-step Householder) | Overkill — solving a 10-class problem with machinery for language models |
| Bidirectional scan | pos_emb already encodes absolute position; no need to process backwards |
| TTT / gradient-as-update | Entirely different training regime; the gain is for very long sequences |
| Discrete val bins > 32 | 32 bins is already finer than MNIST stroke variation warrants |

---

## Success metrics

| Experiment | Target | Pass condition |
|---|---|---|
| S1 symmetric | ≥ 95.0% | keep symmetric, drop W_v |
| S2 no projection | ≥ 94.5% | keep projection-free model |
| S3 linear readout | ≥ 94.5% | drop MLP |
| S4 d=32 | ≥ 94.0% | use smaller embedding |
| Delta rule | ≥ 97.0% | justified complexity add |

The ideal end state: a model where every component earns its place —
probably `M_T = emb^T emb` with a linear readout and optionally the delta rule,
achieving 95-97% with fewer parameters and cleaner interpretation than the
current baseline.
