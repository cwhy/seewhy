# Report 6 — Categorical Latents & Shared Codebook (exp8–12)

## Formulation

Latent construction before the decoder projection (exp9–12, shared codebook):

```python
# Shared codebook
E : (N_CODES, D_EMB)                         # 16 × 16

# Sampling (sparse prior)
k[0]   = 0                                   # slot 0 always absent
k[i]   = 0          with prob P_ZERO         # i = 1..N_VARS-1
k[i]   ~ Uniform(1, N_CODES-1)  otherwise

# Hard lookup — for pixel-space pool decoding (no gradient needed)
z[i]   = E[k[i]]                             # (D_EMB,)
z      = concat(z[0], ..., z[N_VARS-1])      # (LATENT_DIM = N_VARS * D_EMB,)

# Soft lookup — for training step (gradient flows through E)
w[i]   = softmax(one_hot(k[i]) * τ)         # (N_CODES,),  τ=10 → near one-hot
z[i]   = w[i] @ E                            # (D_EMB,)
z      = concat(z[0], ..., z[N_VARS-1])      # (LATENT_DIM,)

# Decoder entry (shared across exp6–12)
h      = GELU(z @ W_mlp1 + b_mlp1)          # (MLP_HIDDEN,)
```

For exp8 (per-variable codebook), replace `E[k[i]]` with `E_i[k[i]]` where
`E_i : (N_CODES, D_EMB)` is a separate table per slot.

---

## Goal

Replace the continuous Gaussian prior z~N(0,I) with structured discrete latents:
a vector of categorical variables, each selecting from a learned codebook.
Test whether prior structure (discrete vs continuous, shared vs per-variable,
sparsity) changes the 87% accuracy ceiling.

---

## Experiments

### exp8 — Per-variable codebook (baseline discrete)

| Hyperparameter | Value |
|----------------|-------|
| Codebook | 16 separate tables, each (16, 8) |
| LATENT_DIM | 16 × 8 = 128 |
| Prior | k_i ~ Uniform(16) for each of 16 slots |
| Prior space | 16^16 ≈ 1.8 × 10^19 |

Gradient flows through a softmax relaxation during training:
```
w = softmax(one_hot(k_i) * TEMP=10)    # near one-hot, but Jacobian reaches all entries
z_i = w @ codebook_i
```

**Result: 87.0%, 495s, 160K params.**

---

### exp9 — Shared codebook + fixed slot 0

| Change from exp8 | Value |
|-----------------|-------|
| Codebook | 1 shared table (16, 16) — D_EMB doubled to compensate |
| Slot 0 | Always code 0 (fixed anchor) |
| Prior space | 16^15 ≈ 1.15 × 10^18 |

The shared codebook forces all slots to draw from the same 16 embeddings.
Code 0 in slot 0 acts as a learned global bias always present in every sample.

**Result: 87.0%, 496s, 167K params.**

Visualisations:
- Samples (hex): https://media.tanh.xyz/seewhy/26-03-23/exp9_samples_hex.png
- Codebook isolation: https://media.tanh.xyz/seewhy/26-03-23/exp9_codebook_viz.png

---

### exp10–12 — Zero-biased prior (sparsity sweep)

Each free slot (1..15) is code 0 with probability P_ZERO, else uniform from 1..15.
This encourages code 0 to learn "absent" and other codes to learn isolated features.

| Exp | P_ZERO | Avg active slots | Bin acc |
|-----|--------|-----------------|---------|
| exp9 | 0 | 15.0 | 87.0% |
| exp10 | 0.5 | 7.5 | 87.0% |
| exp11 | 0.8 | 3.0 | 87.0% |
| exp12 | 0.9 | 1.5 | **86.7%** |

Samples and codebook isolation grids:
- exp10 samples: https://media.tanh.xyz/seewhy/26-03-23/exp10_samples_hex.png
- exp10 codebook: https://media.tanh.xyz/seewhy/26-03-23/exp10_codebook_viz.png
- exp11 samples: https://media.tanh.xyz/seewhy/26-03-23/exp11_samples_hex.png
- exp11 codebook: https://media.tanh.xyz/seewhy/26-03-23/exp11_codebook_viz.png
- exp12 samples: https://media.tanh.xyz/seewhy/26-03-23/exp12_samples_hex.png
- exp12 codebook: https://media.tanh.xyz/seewhy/26-03-23/exp12_codebook_viz.png

---

## Findings

**1. The 87% ceiling is prior-agnostic.**
Continuous Gaussian, discrete uniform categorical (per-variable or shared), and
zero-biased sparse priors all converge to 87.0%. The IMLE matching bottleneck
is entirely in the nearest-neighbour step (N_SAMPLES=1000 vs 60K images), not
in the structure or distribution of the latent space.

**2. Shared codebook works as well as per-variable.**
With doubled D_EMB (8→16), a single (16, 16) codebook matches the accuracy
of 16 separate (16, 8) codebooks at a similar parameter count. The decoder
learns to use the same 16 embedding prototypes differently at each slot.

**3. Moderate sparsity is free; extreme sparsity hurts.**
P_ZERO up to 0.8 (~3 active slots per sample) has zero accuracy cost.
At P_ZERO=0.9 (~1.5 active slots), accuracy drops to 86.7% — the decoder
no longer receives enough signal per sample to reconstruct fine detail.
The sparsity threshold is somewhere between 1.5 and 3 active slots.

**4. Codebook isolation reveals per-slot feature decomposition.**
The codebook vizualisation (all slots=0 except one) shows what each code
contributes in isolation. Sparser priors (exp11, P_ZERO=0.8) produce more
interpretable isolated features since the decoder is trained with fewer
active slots and must make each one count.

**5. exp11 (P_ZERO=0.8) is the recommended setting** for interpretability work:
3 active slots on average, full accuracy, and clear isolated code features
in the codebook grid.
