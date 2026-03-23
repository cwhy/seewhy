# Report 2 — Basic IMLE Ceiling (exp1–4)

## Goal

Establish the performance ceiling of basic IMLE with the gram decoder,
and identify which hyperparameters (sample count, latent dim, decoder size)
limit accuracy.

---

## Results

| Exp | Change | Bin acc | Match L2 (eval) | Time | Params |
|-----|--------|---------|-----------------|------|--------|
| 1 | baseline — N_SAMPLES=1000, LATENT_DIM=64, MLP=256, rank=16 | 87.1% | 1702 | 330s | 1.25M |
| 2 | N_SAMPLES=5000 | 87.2% | 1632 | 550s | 1.25M |
| 3 | LATENT_DIM=256 | 87.1% | 1735 | 314s | 1.25M |
| 4 | MLP=512, rank=32 | 87.1% | 1688 | 323s | 4.7M |

---

## Finding: 87.1% is a hard ceiling for basic IMLE

Every experiment landed at 87.1–87.2% regardless of what was changed:

- **More samples (exp2, N_SAMPLES=5000)**: +0.1% at 67% more time. Negligible
  gain — better coverage per epoch does not translate to better accuracy.

- **Larger latent (exp3, LATENT_DIM=256)**: no effect, slightly worse match L2.
  More dimensions add noise to the nearest-neighbor search without improving
  expressivity of the matched samples.

- **Bigger decoder (exp4, MLP=512, rank=32)**: no effect. The decoder is not
  the bottleneck — the same 87.1% with 4× more parameters.

---

## Why 87.1%?

The ceiling is in the **matching mechanism**, not the decoder.

With N_SAMPLES random codes covering a high-dimensional latent space, each
real image is assigned its nearest neighbor from a sparse random set. The
quality of that assignment is bounded by how well N_SAMPLES points can cover
the region of latent space that decodes to a given digit. At N_SAMPLES=1000–5000,
this coverage is mediocre — most real images are matched to an imperfect sample
regardless of how expressive the decoder is.

Concretely: with N_SAMPLES=1000 and 60K training images, there are on average
~60 real images competing for each sample. The decoder learns to satisfy all of
them simultaneously, which means satisfying none of them precisely.

This is exactly the problem **Adaptive IMLE** was designed to solve: instead of
a fixed random pool, it maintains per-example adaptive thresholds and focuses
samples where coverage is poor.

---

## Visualizations

### exp2 (N_SAMPLES=5000)
- Learning curves: https://media.tanh.xyz/seewhy/26-03-23/exp2_learning_curves.svg
- Nearest-neighbor recon: https://media.tanh.xyz/seewhy/26-03-23/exp2_recon_final.png
- Prior samples: https://media.tanh.xyz/seewhy/26-03-23/exp2_samples_5x8_seed0.png

### exp3 (LATENT_DIM=256)
- Learning curves: https://media.tanh.xyz/seewhy/26-03-23/exp3_learning_curves.svg
- Nearest-neighbor recon: https://media.tanh.xyz/seewhy/26-03-23/exp3_recon_final.png
- Prior samples: https://media.tanh.xyz/seewhy/26-03-23/exp3_samples_5x8_seed0.png

### exp4 (MLP=512, rank=32)
- Learning curves: https://media.tanh.xyz/seewhy/26-03-23/exp4_learning_curves.svg
- Nearest-neighbor recon: https://media.tanh.xyz/seewhy/26-03-23/exp4_recon_final.png
- Prior samples: https://media.tanh.xyz/seewhy/26-03-23/exp4_samples_5x8_seed0.png

---

## Next Step: Adaptive IMLE

Implement the adaptive threshold strategy from Aghabozorgi et al. (ICML 2023):

- Maintain a persistent pool of N_SAMPLES decoded samples throughout training
  (rather than resampling every epoch from scratch).
- Per-example threshold τᵢ: a real image only triggers a gradient update when
  its nearest sample is farther than τᵢ. Easy examples (already well-covered)
  are skipped; hard examples get more updates.
- Threshold adapts downward as coverage improves.

This should allow the model to focus capacity on under-covered regions of the
data manifold, breaking the 87.1% plateau.
