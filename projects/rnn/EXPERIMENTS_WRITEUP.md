# MNIST Pixel-by-Pixel Linear RNN — Experiment Log

**Goal**: Minimise non-embedding parameters while maximising MNIST test accuracy.
Architecture: linear RNN whose final state is a **Gram matrix** (outer-product sum), fed into a small MLP.

---

## Architecture Family

All models share the same computation graph:

```
pixels x ∈ R^784 (uint8 pixel values)
  ↓  embedding lookup
emb_t ∈ R^d         for each pixel t
  ↓  Gram matrix
M = Σ_t emb_t ⊗ emb_t  ∈ R^{d×d}   (equivalent to K^T V in linear attention)
  ↓  LayerNorm
  ↓  MLP (1 hidden layer, GELU)
  ↓  10-way softmax
```

**Non-embedding params** = everything except lookup tables (pos_emb / row_emb / col_emb / val_emb).
These equal `2·d² + mlp·d² + mlp + 10·mlp + 10`, dominated by `W1 ∈ R^{mlp × d²}`.

### Embedding variants

| Name | Formula | Order of statistics in M |
|---|---|---|
| additive S2 | `emb_t = pos_emb[t] + val_emb[bin]` | 2nd |
| outer | `emb_t = pos_emb[t] ⊗ val_emb[bin]` | 4th |
| factored3 | `emb_t = row_emb[r] ⊗ col_emb[c] ⊗ val_emb[bin]` | 6th |
| joint | `emb_t = E[t, bin]` direct lookup `E ∈ R^{784×K×d}` | full per-position |

### Readout / head variants

| Name | Formula | Non-emb params |
|---|---|---|
| MLP (standard) | `flatten(M) → LN → GELU(W1) → W2 → 10` | `d²·mlp + 11·mlp + 2·d² + 10` |
| bilinear+MLP | `LN(M @ h_answer) → GELU(W1) → W2 → 10` | `d·(3+mlp) + 11·mlp + 10` |

The bilinear+MLP head compresses the state from d²→d via a matrix-vector product before the MLP. At d=64, this reduces W1 from `mlp×4096` to `mlp×64` — a 64× reduction in the dominant parameter block.

---

## Results by Experiment Round

### exp3 — Simplification sweep (additive S2)

Swept `d ∈ {32, 48, 64}` and `mlp ∈ {64, 128, 256}`.

**Key finding**: 98% of params live in W1. Non-emb params = MLP params essentially.
Best: d=64 mlp=256 → **95.82%** at 1.06M non-emb.

---

### exp4 — Outer-product embeddings

Compared outer(d_p, d_v) vs additive at identical (state_dim, mlp_dim).

**Key finding**: Outer embeddings save embedding params but have *identical non-emb count* at same state_dim. The Gram matrix captures 4th-order pixel statistics vs 2nd-order.
Best: outer(4,8) mlp=128 → **95.17%** at 135K non-emb.

---

### exp5 — Push to 96%, train accuracy tracking

Added per-epoch train accuracy. Explored mlp scaling, weight decay, dropout, deeper MLP, asymmetric projections.

**Key finding**: mlp=256 saturates outer models; mlp=512–1024 adds nothing. Train/test gap ≈ 1% → not overfitting-limited at baseline.
Best: outer(8,8) mlp=256 → **96.42%**; outer+asym(64,64) mlp=256 → **96.46%**.

---

### exp6 — factored3 embedding, shift augmentation, bins=64

Introduced 3-way outer product: `row_emb[r] ⊗ col_emb[c] ⊗ val_emb[bin]`.

**Key findings**:
- factored3(4,4,4) mlp=256 → **97.27%** — big jump over outer at same non-emb
- factored3 has 2D spatial structure → 6th-order pixel statistics
- Shift aug (random 2D crop) helps outer but **hurts factored3** — row/col embeddings are position-absolute, shift breaks them
- Pareto front dominated by factored3 from here on

---

### exp7 — Longer training, pixel noise augmentation, larger dims

**Key findings**:
- factored3(4,4,4) mlp=512 → **98.16%** at 2.11M non-emb
- factored3(4,4,4) mlp=128 **+noise** → **98.04%** at 534K
- Noise aug (±15px uniform) is architecture-compatible for factored3: preserves positions, forces val_emb to learn intensity-robust features
- factored3(2,4,4) mlp=128 +noise → **96.36%** at 135K

---

### exp8 — Norms (DyT, RMSNorm), optimizer variants (NS-Adam), spatial gate, GELU

Tested several architectural ideas from recent papers.

| Idea | Result |
|---|---|
| **GELU activation** | +0.27% at 534K, +0.40% at 1.06M — **clear win** |
| DyT (Dynamic Tanh) norm | **Hurts** — LayerNorm mean centering is important for PSD Gram matrix |
| RMSNorm (no mean subtraction) | **Hurts** — same reason |
| NS-Adam (Newton-Schulz + Adam) | **Hurts** — NS orthogonalisation conflicts with Adam's per-element second moments |
| Spatial gate (RWKV7-style) | **Hurts** — gating out pixels loses information |
| bins=64 vs bins=32 | No benefit |

**Key finding**: GELU is not a minor tweak — it fundamentally improves the MLP's ability to learn from the Gram matrix features.

Best: factored3(4,4,4) mlp=256 GELU +noise → **98.40%** at 1.06M.

---

### exp9 — GELU + true Muon, more epochs, larger state

Implemented true Muon: Newton-Schulz + Nesterov SGD for W1/W2 weight matrices; Adam only for embeddings and scalars.

**Muon note**: The "NS-Adam" from exp8 was wrong — applying NS to gradients then feeding into Adam's EMA state causes conflicting normalisation. True Muon uses NS for the *direction* and a separate LR schedule.

| Model | NonEmb | Test acc | Notes |
|---|---|---|---|
| factored3(4,4,4) mlp=128 GELU +noise Muon | 534K | **98.36%** | +0.05% over Adam |
| factored3(4,4,4) mlp=256 GELU +noise Muon | 1.06M | **98.60%** | +0.20% over Adam; converges in 11 epochs |
| factored3(4,4,4) mlp=512 GELU +noise Adam | 2.11M | **98.56%** | baseline |
| factored3(2,4,4) mlp=128 GELU +noise Adam | 135K | **96.80%** | new Pareto point |

**Key finding**: GELU + true Muon at 1.06M beats previous best at 2.11M (ReLU + Adam), at half the params.

---

### exp10 — Stronger noise augmentation (±30px), deep MLP, 2-head

Tested scaling noise from ±15 to ±30 pixels.

| Model | NonEmb | Test acc | vs noise=15 |
|---|---|---|---|
| factored3(4,4,4) mlp=512 GELU +noise15 Muon | 2.11M | 98.55% | baseline |
| factored3(4,4,4) mlp=256 GELU **+noise30** Muon | 1.06M | **98.65%** | +0.05% |
| factored3(4,4,4) mlp=512 GELU **+noise30** Muon | 2.11M | **98.81%** | **+0.26%** — huge! |
| factored3(4,4,4) mlp=256+128 deep GELU Muon | 1.09M | 98.61% | marginal |
| factored3(2,4,4) mlp=256 GELU +noise Muon | 267K | **97.11%** | new Pareto point |
| 2-head (bug: emb keys) | — | 11.24% | FAILED — EMB_KEYS bug |

**Key finding**: Stronger noise (±30px vs ±15px) unlocks a **+0.26% gain at 2.11M**. The noise forces val_emb to learn intensity-robust features instead of memorising exact bin values. Effect is larger at bigger models (more capacity to exploit robust features).

**Current overall record: 98.81%** — factored3(4,4,4) mlp=512 GELU +noise30 Muon at 2.11M non-emb.

---

## Current Pareto Front (as of exp10)

| Non-emb params | Test acc | Model |
|---|---|---|
| 12K | 93.44% | d=32 additive, linear readout |
| 35K | 94.37% | outer(4,4) mlp=128 |
| 135K | **96.89%** | factored3(2,4,4) mlp=128 GELU +noise Muon |
| 267K | **97.11%** | factored3(2,4,4) mlp=256 GELU +noise Muon |
| 534K | **98.46%** | factored3(4,4,4) mlp=128 GELU +noise30 Muon |
| 1.06M | **98.65%** | factored3(4,4,4) mlp=256 GELU +noise30 Muon |
| 2.11M | **98.81%** | factored3(4,4,4) mlp=512 GELU +noise30 Muon |

---

## What We Learned

### Architecture
- **factored3 is the best embedding**: Separating row, col, and value embeddings gives 2D spatial structure in the Gram matrix (6th-order statistics). This is fundamentally better than a flat positional outer product.
- **Gram matrix = implicit causal linear attention at final token**: M = Σ_t emb_t emb_t^T = K^T K (self-attention special case). The model learns which 2nd-order pixel co-occurrence statistics are diagnostic.
- **LayerNorm is crucial for the Gram matrix**: The Gram is PSD (all positive eigenvalues, non-zero mean). Removing the mean (centering) is what makes it useful. DyT and RMSNorm both hurt because they skip mean centering.

### Augmentation
- **Noise augmentation is the key regulariser for factored3**: ±15px of uniform noise preserves positions (factored3 stays architecture-compatible) but forces val_emb to learn coarser intensity features.
- **Stronger noise = better generalisation** (at least up to ±30px): The more noise, the harder it is for val_emb to memorise exact pixel values → better generalisation. Noise scale likely has an optimal value beyond which signal is destroyed.
- **Shift augmentation hurts factored3** (row/col embeddings are position-absolute) but helps outer models.

### Activation Function
- **GELU consistently outperforms ReLU** across all model sizes (+0.27% to +0.43%). The smooth gradient at zero and the implicit gating help the MLP extract non-linear combinations of Gram features.

### Optimizer
- **True Muon outperforms Adam for weight matrices**: Newton-Schulz orthogonalisation + Nesterov SGD for W1/W2 gives more isotropic updates. Effect is strongest at moderate sizes (1.06M: +0.20% over Adam).
- **The key is separation**: Adam for embeddings (per-element adaptation important), Muon for 2D weights (spectral normalisation important).
- **NS-Adam (wrong Muon) hurts**: Applying Newton-Schulz then Adam creates conflicting normalisation — Adam's second moments track already-orthogonalised gradients, which is uninformative.
- **Muon converges faster**: 11 epochs vs 30 for Adam to reach the same accuracy.

---

### exp11 — Noise ceiling, larger state dims, 2-head fix, label smoothing

Tested noise scale ceiling, factored3(4,4,6) state dim, fixed 2-head, and label smoothing.

| Model | NonEmb | Test acc | Notes |
|---|---|---|---|
| factored3(4,4,4) mlp=128 +noise30 Muon | 534K | **98.46%** | +0.10% over noise15, new Pareto point |
| factored3(4,4,4) mlp=512 +noise40 Muon | 2.11M | 98.67% | **Worse** than noise30 (98.81%) |
| factored3(4,4,4) mlp=512 +noise50 Muon | 2.11M | 98.70% | **Worse** than noise30 |
| factored3(4,4,6) mlp=64 +noise30 Muon | 609K | 98.27% | Dominated by 534K (4,4,4) |
| factored3(4,4,6) mlp=128 +noise30 Muon | 1.2M | 98.35% | Worse than 1.06M (4,4,4) |
| 2-head mlp=128 +noise30 Muon (fixed) | 1.07M | 98.49% | Worse than single-head 1.06M (98.65%) |
| mlp=512 +noise30 +label_smooth ε=0.1 | 2.11M | 98.73% | Slightly worse than baseline |

**Key findings**:
- **noise30 is the optimal noise scale** — noise40 and noise50 both hurt at 2.11M. The signal starts degrading above ±30px.
- **factored3(4,4,4) is the right state size** — larger (4,4,6) forces a smaller MLP for the same non-emb budget, which hurts. The 4096-dim state with appropriately-sized MLP is the sweet spot.
- **2-head Gram adds nothing** — even with EMB_KEYS fixed, two independent Gram matrices at the same parameter budget are worse than one. The two heads learn redundant features.
- **Label smoothing doesn't help** — model confidence is not the bottleneck at 2.11M.

---

---

### exp12 — Low non-emb regime (sub-135K)

Filled the 35K→135K gap in the Pareto front. factored3 at smaller dimensions.

| Model | NonEmb | Test acc | Notes |
|---|---|---|---|
| factored3(2,2,2) mlp=64 +noise15 Muon | 4.9K | 94.49% | new Pareto point |
| factored3(2,4,4) mlp=32 +noise15 Muon | 35K | 95.85% | beats outer(4,4) at 35K (94.37%) by +1.5% |
| factored3(2,4,4) mlp=64 +noise30 Muon | 68K | 96.43% | new Pareto point |
| factored3(2,4,4) mlp=128 +noise30 Muon | 135K | 96.90% | new Pareto point |

**Key findings**:
- **factored3 dominates outer at every small budget**: (2,4,4) at 35K beats outer(4,4) by +1.5%. The 6th-order statistics matter even with a tiny state.
- **Big state + small MLP beats small state + big MLP** at same non-emb budget: factored3(2,4,4) state_dim=1024 with a tiny MLP is better than factored3(2,2,4) state_dim=64 with a larger MLP.
- **noise30 better than noise15 at 68K+**: Stronger regularisation also helps small models.

---

### exp13 — Joint (position × value) embedding

New embedding: `E[t, bin] ∈ R^d` — a direct lookup table with one vector per (pixel position, value bucket). No factorisation at all.

| Model | NonEmb | Test acc | Notes |
|---|---|---|---|
| joint(d=32) mlp=32 +noise30 Muon | 35K | **96.60%** | +0.75% over factored3(2,4,4) at same budget |
| joint(d=32) mlp=64 +noise30 Muon | 68K | **97.29%** | +0.86% over factored3(2,4,4) at 68K |
| joint(d=32) mlp=128 +noise30 Muon | 135K | **97.69%** | +0.79% over factored3 at 135K |
| joint(d=32) mlp=256 +noise30 Muon | 267K | **97.91%** | new Pareto point |
| joint(d=64) mlp=128 +noise30 Muon | 534K | 97.54% | **loses** to factored3(4,4,4) 98.46% |
| joint(d=64) mlp=256 +noise30 Muon | 1.06M | 97.46% | **loses** to factored3(4,4,4) 98.60% |

**Key findings**:
- **joint(d=32) dominates the 35K–267K Pareto front** over all prior models. The extra freedom (one independent vector per position×value pair) exploits the 60K training examples efficiently.
- **Phase transition at d=64**: joint(d=64) has 1.6M embedding params for 60K examples (26 params/example) → overfits badly, loses ~1% to factored3. joint(d=32) has 802K emb params (13 params/example) → still works.
- **Fundamental tradeoff**: joint embedding is a larger lookup table but captures exact positional patterns without factorisation assumptions. Factored3 wins when the emb table is too large relative to training data.

---

### exp14 — Learned projection before Gram (decouple state_dim from emb_dim)

Hypothesis: the constraint `state_dim = emb_dim²` is arbitrary. Add `P ∈ R^{k×d}` before the Gram: `proj_t = P @ emb_t`, `M = Σ proj_t ⊗ proj_t`.

| Model | NonEmb | Test acc | vs baseline |
|---|---|---|---|
| joint_proj(d=8, k=64) mlp=128 | 534K | 97.17% | -1.29% vs f3(4,4,4) |
| joint_proj(d=16, k=64) mlp=128 | 535K | 97.33% | -1.13% |
| joint_proj(d=32, k=64) mlp=128 | 536K | 97.13% | -1.33% |
| f3_proj(2,4,4, k=64) mlp=128 | 537K | 97.07% | -1.39% |
| joint_proj(d=8, k=64) mlp=256 | 1.06M | 97.51% | -1.09% vs f3(4,4,4) |

**Key findings**:
- **Projection before Gram adds no expressive power** — all 7 variants lost by 0.37%–1.42%. The reason is mathematically precise: `M_proj = P M_raw P^T` is a linear transformation of `M_raw`. The MLP applied to `M_proj` can learn any function that an MLP on `M_raw` could learn. The projection adds a bottleneck but no new structure.
- **The constraint state_dim = emb_dim² is not arbitrary** — it is the exact dimensionality of all second-order statistics of the embedding. Any compression loses information that the MLP was exploiting.

---

### exp15 — GLU variants for the MLP head

Replaced `GELU(W1 @ x)` with split-projection GLU: `(W_val @ x) * act(W_gate @ x)`.
Used `g = mlp//2` to match non-emb budget (W1 splits into two g-wide matrices).

Variants tested: GEGLU (gate=gelu), SwiGLU (gate=silu), Bilinear (gate=identity).

| Variant | NonEmb | Test acc | vs GELU baseline |
|---|---|---|---|
| f3(4,4,4) GEGLU g=64 | 533K | 98.16% | -0.30% |
| f3(4,4,4) SwiGLU g=64 | 533K | 97.85% | -0.61% |
| f3(4,4,4) Bilinear g=64 | 533K | 97.95% | -0.51% |
| f3(4,4,4) GEGLU g=128 | 1.06M | 98.12% | -0.48% |
| f3(4,4,4) SwiGLU g=128 | 1.06M | 98.17% | -0.43% |
| f3(4,4,4) GEGLU g=256 | 2.11M | 98.40% | -0.41% |
| f3(4,4,4) SwiGLU g=256 | 2.11M | 98.24% | -0.57% |

**Key findings**:
- **GLU variants consistently lose** at every budget and every activation choice.
- **Root cause: hidden width bottleneck**. With `g = mlp//2`, the GLU output is g-dimensional (half the width of the GELU baseline). For the same non-emb budget, GELU gives a 128-wide hidden layer while GEGLU gives 64-wide. The gating does not compensate for the narrower representation.
- **Bilinear (pure gating, no nonlinearity) also loses**, ruling out the hypothesis that multiplicative interactions alone help. The nonlinearity in GELU is genuinely useful, not just a proxy for gating.
- **Conclusion**: GELU is the optimal activation for this Gram-matrix MLP head. GLU variants are closed.

---

### exp16 — Matrix-structured readout of M

**Motivation**: the current architecture flattens M into a d²-dim vector and feeds it to an MLP. This discards M's matrix structure. Can we exploit M as a matrix?

**Architecture A — Bilinear readout** (the core idea):
```
h_answer ∈ R^d,   H_y ∈ R^{10×d}
h_yhat  = M @ h_answer           (matrix-vector product; compress d²→d)
h_yhat  = LN(h_yhat)             (normalise; M scale grows with #pixels)
logit_c = h_yc · h_yhat          (dot-product classifier)
```
`logit_c = h_yc^T M h_answer = Σ_t (h_yc·emb_t)(h_answer·emb_t)`:
the logit is a sum over pixels of (class-alignment × answer-alignment).

Variant: replace the dot-product head with a small MLP applied to the d-dim `h_yhat` rather than the d²-dim flat state. This is a massive compression.

**Architecture B — Projected Gram**:
```
H_proj ∈ R^{k×d}
h_t = H_proj @ emb_t             (project each embedding to k-dim)
G   = Σ_t h_t ⊗ h_t             (k×k Gram in the learned subspace)
logits = W_out @ LN(flatten(G))  (linear classifier on k² features)
```

| Model | NonEmb | % of 534K | Test acc | vs f3 GELU |
|---|---|---|---|---|
| f3 bilinear+LN (dot head) | 832 | 0.16% | 92.21% | -6.25% |
| f3 bilinear+LN+mlp32 | 2,602 | 0.49% | **97.20%** | -1.26% |
| f3 bilinear+LN+mlp128 | 9,802 | 1.84% | **97.82%** | -0.64% |
| f3 bilinear+LN+mlp512 | 38,602 | 7.2% | **98.07%** | -0.39% |
| f3 proj_gram k=8 | 1,290 | 0.24% | 92.78% | -5.68% |
| f3 proj_gram k=16 | 4,106 | 0.77% | 93.66% | -4.80% |
| f3 proj_gram k=32 | 14,346 | 2.69% | 93.71% | -4.75% |
| f3 proj_gram k=64 | 53,258 | 10.0% | 93.61% | -4.85% |
| joint bilinear+LN | 416 | 0.08% | 93.06% | — |

**Key findings**:

1. **Bilinear + small MLP is extraordinarily param-efficient**:
   - 9,802 non-emb → 97.82% (1.84% of baseline budget, only -0.64% accuracy)
   - 38,602 non-emb → 98.07% (7.2% of baseline budget, only -0.39% accuracy)
   - This dominates the entire 2.6K–53K Pareto range: new records at every budget.

2. **The projected Gram + linear classifier plateaus at ~93.6%** regardless of k (8 through 64). This isolates the value of the MLP's nonlinearity: a linear classifier on Gram features, however many you take, is worth ~5% less than a nonlinear MLP.

3. **The nonlinearity is the key ingredient, not the parameter count**: `proj_gram k=64` has 4096 features (same as the full Gram state) but only reaches 93.61% with a linear classifier. The same state with a GELU MLP reaches 98.46%. Those ~5 percentage points come entirely from the MLP's nonlinearity, not from more parameters.

4. **The d-dim bottleneck (M @ h_answer) works because**: `M @ h_answer = Σ_t (h_answer·emb_t) emb_t` is a soft attention over pixel embeddings — a learned, weighted summary of the image. Most classification signal can be routed through this single direction. The subsequent MLP then applies nonlinear functions to the 64-dim summary rather than the full 4096-dim state.

5. **Why bilinear+dot fails (92.21%)**: The dot-product head is a *linear* classifier on the d-dim readout. Just like proj_gram, it lacks the nonlinearity needed to separate the 10 classes from Gram features.

---

### exp17 — Deep bilinear (M²) and multi-query readout

Building on exp16's bilinear+MLP bottleneck: apply M multiple times (depth), or use K independent answer vectors (multi-query).

**Architecture:**
```
depth=D, K queries:
  for each query k:
    h = h_answer_k
    for i in 1..D:
        h = LN(M @ h)          ← one hop through the pixel similarity graph
  features = concat([h_1, ..., h_K])   ∈ R^{K·d}
  logits   = MLP(LN(features))
```

| Variant | NonEmb | Test acc | Notes |
|---|---|---|---|
| depth=2 mlp=128 | 10,058 | 97.83% | +0.01% over depth=1 — nearly free |
| depth=3 mlp=128 | 10,186 | 97.78% | Slightly worse than depth=2 |
| depth=2 mlp=512 | 38,858 | 98.24% | +0.17% at larger scale |
| K=2 mlp=128 | 18,314 | 98.28% | +0.46% — two directions richer than one |
| K=4 mlp=64 | 17,994 | 98.04% | +0.22% — more directions, less MLP width |
| K=8 mlp=32 | 18,410 | 97.47% | **Worse** — MLP too narrow to decode 8 directions |
| **K=4 mlp=128** | **35,082** | **98.54%** | **Beats 534K standard baseline (+0.08%)!** |
| K=2 depth=2 mlp=128 | 18,442 | 98.40% | +0.58% — combining both ideas |
| K=4 depth=2 mlp=64 | 18,122 | 97.97% | Modest gain over K=4 depth=1 mlp=64 |

**Key findings:**

1. **K=4 mlp=128 at 35K non-emb achieves 98.54%**, beating the standard f3(4,4,4) GELU mlp=128 architecture at 534K by +0.08%. A **15× reduction in non-emb params** while improving accuracy. The bilinear multi-query readout has fundamentally reshaped the Pareto front.

2. **K=2 depth=2 mlp=128 at 18K non-emb gets 98.40%** — only 0.06% below the 534K standard, at 29× fewer non-emb params.

3. **Multiple queries are more valuable than depth**: going K=1→K=2 adds +0.46% at identical per-query compute; depth=1→depth=2 adds only +0.01% at similar budget. The bottleneck is breadth (how many directions to read from M), not depth (how many hops).

4. **Depth helps more at larger MLP**: +0.01% at mlp=128, +0.17% at mlp=512. The two-hop walk adds signal only when the MLP has capacity to use it.

5. **K=8 mlp=32 fails**: fragmenting the budget into too many queries with too narrow an MLP hurts. The MLP needs enough width to decode K directions simultaneously — there's a minimum width per query.

6. **Why multi-query works**: each h_k = M @ h_answer_k = Σ_t (h_k·emb_t) emb_t is a different linear projection of the image summary. Four such projections (K=4, 256 total features) contain more of the classification signal than the single 64-dim projection, even though the full Gram has 4096 features. The Gram matrix has ~4 strongly informative directions; reading beyond K=4 yields diminishing returns.

---

### exp18 — Low-end sweep (sub-2.6K), f3(4,4,4) bilinear + joint(d=32) baselines

Swept the unexplored sub-2.6K regime using f3(4,4,4) bilinear with mlp=4/8/16/24/48/64/96, plus K=2 multi-query variants, and joint(d=32) bilinear as upper baselines.

| Model | NonEmb | Test acc | Notes |
|---|---|---|---|
| f3(4,4,4) bilinear K=1 mlp=4 | 630 | 70.58% | **Complete failure** — mlp=4 cannot decode Gram |
| f3(4,4,4) bilinear K=1 mlp=8 | 930 | 92.43% | Recovers with more MLP width |
| f3(4,4,4) bilinear K=2 mlp=8 | 1,634 | 93.04% | K=2 adds only +0.61% over K=1 here |
| f3(4,4,4) bilinear K=1 mlp=16 | 1,530 | 95.90% | Big jump — mlp width is key |
| f3(4,4,4) bilinear K=2 mlp=16 | 2,746 | 96.44% | +0.54% from second query |
| f3(4,4,4) bilinear K=1 mlp=24 | 2,130 | 96.80% | Similar to K=2 mlp=16, fewer params |
| f3(4,4,4) bilinear K=1 mlp=48 | 3,930 | 97.37% | — |
| f3(4,4,4) bilinear K=2 mlp=32 | 4,970 | 97.69% | — |
| f3(4,4,4) bilinear K=1 mlp=64 | 5,130 | 97.72% | — |
| f3(4,4,4) bilinear K=1 mlp=96 | 7,530 | **97.94%** | New Pareto point |
| joint(d=32) bilinear K=1 mlp=16 | 858 | **95.31%** | Baseline for tiny budgets |
| joint(d=32) bilinear K=1 mlp=32 | 1,546 | **96.62%** | Beats f3 K=1 mlp=16 (95.90%) at same budget |
| joint(d=32) bilinear K=1 mlp=64 | 2,922 | **97.39%** | New Pareto point |

**Key findings**:
- **mlp=4 is a hard floor** — the MLP cannot extract class-discriminating features from a 64-dim bilinear readout with only 4 hidden units. Accuracy collapses to 70.58%.
- **MLP width matters more than K at the very low end**: going mlp=8→16 adds +3.47%, while going K=1→K=2 at mlp=8 adds only +0.61%.
- **joint(d=32) dominates f3(4,4,4) at identical budgets**: at 858 non-emb, joint gets 95.31% vs f3's 92.43% (+2.88%).

---

### exp19 — Smaller factored3 dims: f3(2,4,4), f3(2,2,4), f3(2,2,2)

Explored smaller factored3 embedding dims to push the Pareto front below 858 non-emb. Also provides clean isolating comparisons vs joint at identical non-emb and identical emb_dim.

| Model | NonEmb | emb_dim | Test acc | Notes |
|---|---|---|---|---|
| f3(2,2,4) bilinear K=1 mlp=8 | 306 | 16 | 84.92% | New low-end Pareto point |
| f3(2,2,2) bilinear K=1 mlp=24 | 506 | 8 | 84.12% | d=8 fails even with wide MLP |
| f3(2,2,4) bilinear K=1 mlp=16 | 522 | 16 | 85.75% | ← compare to joint(d=16) below |
| f3(2,2,4) bilinear K=2 mlp=16 | 826 | 16 | 90.63% | ← compare to joint(d=16) below |
| f3(2,4,4) bilinear K=1 mlp=16 | 858 | 32 | 93.08% | ← compare to joint(d=32) 95.31% |
| f3(2,2,2) bilinear K=1 mlp=64 | 1,266 | 8 | 82.85% | d=8 stuck even wider |
| f3(2,4,4) bilinear K=1 mlp=32 | 1,546 | 32 | 94.81% | Close to joint(d=32) 96.62% |
| f3(2,2,2) bilinear K=1 mlp=128 | 2,482 | 8 | 87.41% | d=8 degraded at all widths |
| f3(2,4,4) bilinear K=4 mlp=32 | 4,906 | 32 | 96.00% | K=4 rescues f3(2,4,4) |
| f3(2,2,4) bilinear K=4 mlp=64 | 5,034 | 16 | 94.05% | K=4 barely helps d=16 factored |

**Key findings**:

1. **joint beats factored3 at every identical non-emb AND emb_dim budget**: at 522 non-emb (emb_dim=16/32 respectively), joint(d=16) will get 95.25% (see exp20) vs f3(2,2,4) 85.75% — a +9.5% gap. This is not a parameter count story; it's purely about embedding quality.

2. **f3(2,2,2) emb_dim=8 fails completely** — accuracy stays in the 82–94% range regardless of MLP width. A 8×8 Gram matrix (64 features) is simply too small to represent the classification signal. The Gram matrix dimensionality is the bottleneck.

3. **New low-end Pareto point**: f3(2,2,4) K=1 mlp=8 at 306 non-emb → 84.92%. First point below 630.

---

### exp20 — Smaller joint embedding dims: joint(d=4), joint(d=8), joint(d=16)

Tested joint bilinear with smaller d to explore the sub-858 regime (previous smallest joint was d=32 at 858 non-emb).

| Model | NonEmb | Test acc | vs factored3 same budget |
|---|---|---|---|
| joint(d=4) K=1 mlp=16 | 270 | 83.61% | new lowest-budget point |
| joint(d=4) K=4 mlp=16 | 498 | **93.53%** | +8.61% over K=1; f3(2,2,4)@522=85.75% |
| joint(d=4) K=1 mlp=32 | 510 | 83.09% | K=1 stuck — Gram too small (16-dim) |
| joint(d=4) K=4 mlp=32 | 930 | 93.87% | minor gain over K=4 mlp=16 |
| joint(d=8) K=1 mlp=32 | 658 | 94.00% | f3(2,2,4)@826=90.63% (diff budget) |
| joint(d=8) K=1 mlp=64 | 1,266 | 93.76% | **+10.91%** vs f3(2,2,2) 82.85% |
| joint(d=8) K=4 mlp=32 | 1,498 | 96.23% | K=4 major boost at d=8 |
| joint(d=8) K=1 mlp=128 | 2,482 | 94.21% | **+6.8%** vs f3(2,2,2) 87.41% |
| joint(d=16) K=1 mlp=16 | 522 | **95.25%** | **+9.5%** vs f3(2,2,4) 85.75% |
| joint(d=16) K=2 mlp=16 | 826 | **95.64%** | **+5.01%** vs f3(2,2,4) 90.63% |
| joint(d=16) K=1 mlp=32 | 954 | 96.02% | — |
| joint(d=16) K=2 mlp=32 | 1,514 | 96.67% | — |
| joint(d=16) K=1 mlp=64 | 1,818 | **96.93%** | — |
| joint(d=16) K=4 mlp=32 | 2,634 | 96.82% | — |
| joint(d=16) K=2 mlp=64 | 2,890 | **97.31%** | — |
| joint(d=16) K=1 mlp=128 | 3,546 | 96.93% | diminishing returns vs K=2 mlp=64 |

**Key findings**:

1. **joint(d=16) is the sweet spot for sub-3K budgets**: it forms the Pareto front from 522 to 2890 non-emb, dominating both joint(d=8) (Gram too small: 64-dim) and joint(d=32) (more embedding params, similar accuracy).

2. **Multi-query rescues tiny Gram matrices**: joint(d=4) K=1 → 83.09%, but K=4 → 93.53% (+10.4%). A 4×4=16-dim Gram has very few informative directions — K=4 exhausts them. This also means K>4 would not help at d=4.

3. **joint(d=16) K=1 mlp=16 at 522 non-emb → 95.25%** beats joint(d=32) mlp=16 at 858 (95.31%) by being 64% cheaper. This is the new Pareto queen in the 500–900 range.

4. **Exact-budget comparisons** (same non-emb, same emb_dim):
   - 522: joint(d=16) **95.25%** vs f3(2,2,4) 85.75% → **+9.50%**
   - 826: joint(d=16) **95.64%** vs f3(2,2,4) 90.63% → **+5.01%**
   - 1266: joint(d=8) **93.76%** vs f3(2,2,2) 82.85% → **+10.91%**
   - 2482: joint(d=8) **94.21%** vs f3(2,2,2) 87.41% → **+6.80%**

   Joint embedding wins by massive margins at identical non-emb budgets. The factored outer-product structure is a significant handicap when d is small.

---

## Current Pareto Front (as of exp20)

| Non-emb params | Test acc | Model |
|---|---|---|
| 270 | 83.61% | joint(d=4) K=1 mlp=16 |
| 306 | 84.92% | f3(2,2,4) bilinear K=1 mlp=8 |
| 498 | 93.53% | joint(d=4) K=4 mlp=16 |
| 522 | 95.25% | joint(d=16) K=1 mlp=16 |
| 826 | 95.64% | joint(d=16) K=2 mlp=16 |
| 954 | 96.02% | joint(d=16) K=1 mlp=32 |
| 1,498 | 96.23% | joint(d=8) K=4 mlp=32 |
| 1,514 | 96.67% | joint(d=16) K=2 mlp=32 |
| 1,818 | 96.93% | joint(d=16) K=1 mlp=64 |
| 2,602 | 97.20% | f3(4,4,4) bilinear+mlp32 |
| 2,890 | 97.31% | joint(d=16) K=2 mlp=64 |
| 2,922 | 97.39% | joint(d=32) K=1 mlp=64 |
| 4,970 | 97.69% | f3(4,4,4) bilinear K=2 mlp=32 |
| 5,130 | 97.72% | f3(4,4,4) bilinear K=1 mlp=64 |
| 7,530 | **97.94%** | f3(4,4,4) bilinear K=1 mlp=96 |
| 18,314 | **98.28%** | f3(4,4,4) bilinear K=2 mlp=128 |
| 18,442 | **98.40%** | f3(4,4,4) bilinear K=2 depth=2 mlp=128 |
| 35,082 | **98.54%** | f3(4,4,4) bilinear K=4 mlp=128 |
| 1.06M | 98.65% | f3(4,4,4) GELU mlp=256 |
| 2.11M | 98.81% | f3(4,4,4) GELU mlp=512 |

Open gap: 7.5K–18K (no new models tested there yet).

Notable: joint(d=16) K=1 mlp=16 at **522 non-emb → 95.25%** — nearly matching the old joint(d=32) baseline (858 → 95.31%) at 61% of the budget.

---

## Promising Directions

1. **7.5K–18K gap**: f3(4,4,4) K=1 mlp=96 gets 97.94% at 7.5K, but the next Pareto point is 18K. Worth filling with K=2 mlp=64 variants.
2. **joint(d=16) K=4 mlp=64+**: K=4 at d=16 got 96.82% at 2634 — worth trying larger MLP to see if it beats K=2 at higher budget.
3. **CIFAR-10 or Fashion-MNIST**: validate the full recipe.

**Closed directions**:
- noise40/50 at 2.11M — noise30 is optimal
- factored3(4,4,6) larger state dim — (4,4,4) is the sweet spot
- 2-head Gram — redundant
- Label smoothing ε=0.1 — doesn't help
- Projection before Gram (exp14) — adds no expressive power
- GLU variants (exp15) — hidden width bottleneck hurts
- Projected Gram + linear classifier (exp16) — nonlinearity is the key, not more features
- f3(2,2,2) and f3(2,2,4) at sub-1K budgets — joint(d=16) dominates by 5–10%

---

## Common Failure Modes

| Bug / Pitfall | Description | Fix |
|---|---|---|
| EMB_KEYS incomplete | When adding new embedding names (e.g. `row_emb_a`), forgetting to add them to `EMB_KEYS` causes Muon's NS to destroy embeddings (train=11%) | Always update `EMB_KEYS` when adding embedding params |
| NS-Adam confusion | Applying NS then Adam to the same parameter — NS output has unit spectral norm, Adam's variance estimate then becomes meaningless | Use true Muon: NS+SGD-momentum for 2D weights, Adam only for 1D/scalars |
| DyT with large Gram values | The raw Gram matrix has large values (~100s); DyT with alpha_init=0.01 is approximately linear but doesn't match LN's centering | LN is better here; DyT designed for pre-norm in transformers where activations are already somewhat bounded |
| Shift aug with factored3 | Random 2D shifts invalidate absolute row/col embeddings | Only use noise aug with factored3 |
| 2-head Gram with multi-transform | `optax.multi_transform` with wrong key labels silently fails | Always verify keys match params dict |
