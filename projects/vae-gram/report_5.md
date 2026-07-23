# Report 5 — Sample Quality: Reconstruction Accuracy vs Generative Quality

## Setup

Generated 40 prior samples (z ~ N(0,1), 5×8 grid) from exp5 (LATENT_DIM=64, 88.7% bin acc)
and exp11 (LATENT_DIM=512, 89.2% bin acc). Both decoded with soft mean (weighted sum of bin
centers, no argmax).

- exp5 samples: https://media.tanh.xyz/seewhy/26-03-23/exp5_samples_5x8_seed0.png
- exp11 samples: https://media.tanh.xyz/seewhy/26-03-23/exp11_samples_5x8_seed0.png

---

## Observations

**Exp5 (64-dim):** Mostly legible digits. Blurry and occasionally ambiguous (merged/distorted
strokes) but clearly digit-shaped, dark backgrounds, reasonable spatial structure. The prior is
usable — sampling from N(0,1)^64 produces meaningful images.

**Exp11 (512-dim):** Mostly broken. Strong checkerboard/grid artifacts dominate most samples.
A handful are accidentally digit-like (those happen to land near the posterior manifold by chance).
The prior is not usable for generation.

---

## Takeaways

### 1. Reconstruction accuracy and generative quality are decoupled

The 0.5% accuracy improvement from exp5 → exp11 came entirely from better reconstruction
(encoder feeds the decoder the right z). It says nothing about whether the prior is well-matched
to the learned posterior. Bin accuracy is the wrong metric for evaluating generative quality.

### 2. Large latent + free bits = prior hole

With FREE_BITS=0.5 and LATENT_DIM=512, the model is forced to use all 512 dims for
reconstruction. The joint posterior across 512 dims develops a complex structure — the decoder
learns co-activation patterns across many dims simultaneously. A random z ~ N(0,1)^512 almost
never matches these patterns. The prior develops a "hole" around the actual data manifold.

With LATENT_DIM=64, the prior is small enough that N(0,1)^64 samples land reasonably close to
the posterior manifold. The model has nowhere else to put the information.

In short: **more latent dims → tighter reconstruction → worse prior coverage**.

### 3. The checkerboard artifacts reveal the decoder's basis functions

The grid pattern visible in most exp11 samples is the Kronecker product structure of the
positional embeddings (row_emb ⊗ col_emb) breaking through. When the decoder receives an
out-of-distribution z, it can't assemble a coherent image and defaults to showing its
structural components — the orthogonal basis of the gram decoder. This is a fingerprint of
the architecture under confusion.

### 4. Exp5 is actually a usable generative model; exp11 is not

Despite lower reconstruction accuracy, exp5 functions as a generator. Exp11 functions only
as a compressor. If the goal were pure generation (sample quality), exp5 would be preferred.
If the goal is reconstruction fidelity (given an input image), exp11 wins.

### 5. Free bits is the root cause of the tradeoff

FREE_BITS=0.5 is essential for reconstruction — it prevents posterior collapse and forces all
dims to encode signal. But it also *guarantees* that the posterior uses every dim, making the
aggregate posterior increasingly unlike the prior as LATENT_DIM grows. There is no free lunch:
the same mechanism that improves reconstruction damages sample quality.

A possible fix would be a learned prior (e.g. a normalising flow or VampPrior trained to
match the aggregate posterior), but that's a significant architectural change.

---

## Summary

| | Exp5 (latent=64) | Exp11 (latent=512) |
|---|---|---|
| Bin accuracy | 88.7% | 89.2% |
| Prior samples | Recognisable digits | Checkerboard artifacts |
| Prior usable? | Yes | No |
| Good for | Generation + reconstruction | Reconstruction only |

The pursuit of bin accuracy via latent scaling quietly broke the generative model. Future work
should track sample quality (e.g. FID or visual inspection) alongside reconstruction accuracy,
and treat them as separate objectives.
