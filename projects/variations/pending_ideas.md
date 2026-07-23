# Pending Ideas — Variations Project

Ideas discussed but not yet run (or partially explored). Ordered roughly by expected
time-efficiency — cheap experiments first.

---

## 1. N_CAT Scaling (code length)

**Status**: not started  
**Cost**: nearly free (tiny embedding table growth)

Increase N_CAT from 4 to 8 or 16. Gives more independent conditioning axes:
- N_CAT=8, VOCAB=8: 8⁸ = 16M codes, COND_DIM=128
- N_CAT=8, VOCAB=32: 32⁸ = 10¹² codes, COND_DIM=128

Each axis can independently capture a different style dimension (color, texture,
lighting, structure). More orthogonal than just a bigger VOCAB.

**Caution**: even sparser than VOCAB=32 per training step. Needs more epochs (400+).
Could combine with continuous conditioning (idea 2) to avoid sparsity entirely.

---

## 2. Continuous Gaussian Noise Conditioning

**Status**: DONE as exp51 — **ruled out** (recall=0.00787, worse than VOCAB=8 baseline)  
**Cost**: identical to exp45 (~43s/epoch)

Replace discrete cat embeddings with eps ~ N(0, I_COND_DIM). Infinite code space,
no sparsity problem, smooth variation landscape. Phase 1 saves winning eps vectors
instead of indices. Reconstruction anchored at eps=0.

**Result**: plateaued at 0.00787 by ep250. Discrete codes are better — embedding table
provides anchor structure that makes variation landscape easier to learn.

---

## 3. z Perturbation (noise on latent directly)

**Status**: running as exp53 (GPU 1, 300ep) — eps ~ N(0, I_2048) added to z  
**Cost**: nearly free (no architecture change, decoder is smaller than exp51)

Instead of separate conditioning, add noise directly to z:
`z_var = z + eps` where `eps ~ N(0, I_2048)`.

Simplest possible change. Variation space is grounded in the 2048-dim latent geometry.
No extra parameters. Recon = eps=0 → z_var = z exactly.

**Variant**: learnable sigma per dimension (heteroscedastic noise).

---

## 4. VOCAB Scaling Beyond 32

**Status**: running as exp52 (GPU 0, 400ep)  
**Cost**: free compute, needs more epochs

Try VOCAB=64 (64⁴ = 16M codes) or VOCAB=128 (128⁴ = 268M codes) with 400+ epochs
and a fresh cosine LR schedule. The VOCAB=32 result showed larger vocab converges
better given enough epochs — worth pushing further.

**Suggested experiment**: VOCAB=64, MAX_EPOCHS=400.

---

## 5. N_VAR Scaling (256 samples)

**Status**: not started  
**Cost**: ~1.6× slower per epoch (4 chunks of 64 vs 2)

N_VAR=128 → 256. Better argmin search per step: each kNN target has 2× more candidates
to match against. Could give faster convergence per epoch, partially offsetting the
compute cost. Net efficiency gain is unclear — depends on whether better matching per
step translates to fewer total epochs needed.

---

## 6. Wider Decoder

**Status**: not started  
**Cost**: ~2× slower per epoch

DC_C1 256 → 512 (and proportionally DC_C2 128→256, DC_C3 64→128). More decoder
capacity to generate pixel-diverse images — possibly the real bottleneck, since all
conditioning improvements had limited early-training benefit.

**Concern**: the architecture might be capacity-limited regardless of conditioning.
This would show up as a significantly lower recall floor even with limited epochs.

---

## 7. Perceptual Loss for IMLE

**Status**: not started  
**Cost**: moderate (need to run a feature extractor per step)

Replace pixel MSE in L_imle with a perceptual loss (e.g., VGG features or a
pretrained CIFAR CNN). Pixel L2 may be a poor signal for CIFAR-10 — two semantically
similar images can be far in pixel space. A perceptual metric might make the kNN
targets and matching more meaningful.

**Implementation**: extract features from a frozen pretrained encoder, compute MSE
in feature space. Could dramatically change what "similar" means for the model.

---

## 8. More Epochs for exp50 (exp35 replica from scratch)

**Status**: exp50 running (GPU 0, 400ep)  
**Pending question**: will it match exp35's 0.00666 at ep400?

This is a validation experiment. If exp50 reaches 0.00666, it confirms the warm-start
was the only special thing about exp35 and the architecture is fine.
If it falls short, there's something about training dynamics that the warm-start helped.

---

## Ruled Out

| Idea | Tried | Why not |
|------|-------|---------|
| K_NN=32 training | exp48 | ep5 recall same as baseline (+25% slower) |
| recon_emb bias | exp47 | final recall 0.00725 — worse than exp45 (0.00703) |
| COND_DIM=256 | exp46 | same as COND_DIM=64, no benefit |
| Precomputed kNN | exp35 style | GEMM is <5% of epoch time, negligible speedup |
| lax.scan | exp35 style | 3.4× slower per epoch, same algorithm |
