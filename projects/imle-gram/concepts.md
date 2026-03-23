# Concepts — imle-gram

Model-specific concepts for the gram IMLE project.
For workflow and tooling conventions see [workflow.md](workflow.md).

---

## What is IMLE?

Implicit Maximum Likelihood Estimation (IMLE) is a generative modelling approach
that replaces the encoder of a VAE with a **nearest-neighbor matching** step.

Instead of encoding each real image into a latent code (like a VAE), IMLE:
1. Samples many random latent codes from the prior z ~ N(0, I)
2. Decodes them to get candidate images
3. Assigns each real image to its nearest candidate (in pixel L2)
4. Trains the decoder to move the matched candidate closer to the real image

There is **no encoder, no KL loss, no posterior**. The model is purely a decoder
(generator) trained to cover the data distribution without mode collapse.

### Why IMLE doesn't collapse

Standard MLE on implicit models (GANs) can collapse modes because the gradient
vanishes when model and data distributions don't overlap. IMLE avoids this:
- Each real image is always assigned a nearest candidate — even if it's far away
- The gradient is proportional to (x_real − x̃_matched), which is zero only
  when they coincide (never in practice)
- Coverage is enforced: every real image must have at least one nearby sample

---

## Algorithm

```
for each epoch:
    1. Sample N_SAMPLES codes:  Z = {z_j}  z_j ~ N(0, I)
    2. Decode (no grad):        X̃ = {G_θ(z_j)}  as soft-mean pixel images
    3. for each mini-batch x_i in X_train:
         σ(i) = argmin_j  ||x_i − x̃_j||²         # nearest neighbor
         loss  = −(1/B) Σ_i  log p_θ(x_i | z_{σ(i)})  # NLL with matched z
         θ ← θ − η ∇_θ loss                        # treat σ as fixed
```

**Matching** is done in pixel space (L2 on soft-mean decoded images).
**Training loss** is cross-entropy NLL on the bin distribution, using the
matched latent code. This gives richer gradients than L2 on soft mean.

Gradient flows only through the re-decoding in step 3, not through the
matching step. σ is treated as a fixed assignment.

---

## Decoder Architecture

Identical to vae-gram/exp19. See [vae-gram/concepts.md](../vae-gram/concepts.md)
for full details. Summary:

```
z ∈ R^LATENT_DIM
  → GELU MLP → (A, B) ∈ R^{DEC_RANK × D}
  → pseudo-gram = A @ Bᵀ ∈ R^{D × D}
  → per-pixel positional query → MLP → logits ∈ R^{N_VAL_BINS}
  → softmax → pixel bin distribution
```

Default config (from vae-gram exp19):
- `LATENT_DIM=64`, `DEC_RANK=16`, `MLP_HIDDEN=256`, `D=144` (D_R=D_C=6, D_V=4)
- `N_VAL_BINS=32`

---

## Key Hyperparameters

| Parameter | Role |
|-----------|------|
| `N_SAMPLES` | Latent codes sampled per epoch for NN matching. More → better coverage, slower matching. Start: 1000. |
| `M_EVAL` | Samples for eval matching (larger pool → more accurate eval). Default: 5000. |
| `LATENT_DIM` | Latent space dimension. Unlike VAE, no FREE_BITS constraint — can go larger freely. |
| `ADAM_LR` | Learning rate. Same cosine decay schedule as vae-gram. |

### N_SAMPLES trade-off

| N_SAMPLES | Coverage | Matching cost | Epoch time |
|-----------|----------|---------------|------------|
| 500       | Poor     | Fast          | Fast       |
| 1000      | OK       | Moderate      | Moderate   |
| 5000      | Good     | Slow          | Slow       |
| 10000+    | Best     | Very slow     | Very slow  |

The paper (Adaptive IMLE) maintains ~10K samples. For MNIST, 1K–5K is practical.

---

## Differences from vae-gram

| Aspect | vae-gram (VAE) | imle-gram (IMLE) |
|--------|----------------|------------------|
| Encoder | Yes (gram-based) | None |
| Latent code source | Posterior q(z\|x) | Prior N(0, I) |
| Loss | recon NLL + β·KL | NLL of matched codes |
| FREE_BITS | Required (0.5) | Not applicable |
| LATENT_DIM ceiling | 88.7% at 64-dim | No ceiling — larger dim OK |
| Prior samples | Can have prior hole | Always from prior (no hole) |
| Training signal | All pixels every step | Only matched codes |

---

## Metrics

| Metric | What it tells you |
|--------|------------------|
| `match_loss` (L2/pixel) | How close real images are to their nearest sample. Lower = better coverage. |
| `bin_acc` (eval) | Bin accuracy of nearest-neighbor sample vs real image. Comparable to VAE bin_acc. |

Note: `bin_acc` in IMLE is approximated by finding the nearest sample from a
pool of M_EVAL decoded samples, then computing argmax accuracy of its logits.
It depends on M_EVAL — use the same value when comparing across experiments.

---

## Sampling Strategies (from the paper)

**Basic IMLE** (exp1): resample Z every epoch, fixed N_SAMPLES.

**Adaptive IMLE** (future): per-example threshold τᵢ — easy examples
(already close to some sample) get sparser matching, hard examples get more
samples allocated. Introduced to handle varying example difficulty.

**RS-IMLE** (future): rejection sampling — discard candidates closer than ε to
any training point before matching. Corrects train/test distribution mismatch
caused by increasing m pushing selected samples toward the data manifold.
