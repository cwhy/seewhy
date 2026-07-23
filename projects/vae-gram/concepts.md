# Concepts — vae-gram

Model-specific architecture, loss formulation, key findings, and
hyperparameter guide for the gram VAE project.

For workflow and tooling conventions see [workflow.md](workflow.md).

---

## Architecture

### Encoder — gram-style

1. Quantize pixel values → bin indices (32 bins)
2. Per-pixel token: `(row_emb[r] ⊗ col_emb[c]) ⊗ val_emb[bin]` → shape `(D,)`
3. Gram matrix: `M = Σ_i token_i ⊗ token_i` → shape `(D, D)` — order-invariant summary of all pixels
4. Query M with learned vectors `h_enc` → layer norm → linear → `mu`, `log_var` → `z`

The gram is the key structural choice: it is a **second-order statistic** of
the image tokens, computed as an outer-product sum. It is permutation-invariant
over pixels (order doesn't matter) and quadratic in D.

### Decoder — position-query on pseudo-gram

1. `z → MLP → A, B` via linear projections → `pseudo_gram = A @ B.T` shape `(D, D)`
2. Per pixel: positional query `q = W_pos @ (row_emb[r] ⊗ col_emb[c])` → `f = pseudo_gram @ q`
3. `f` through a small MLP → logits over `N_VAL_BINS` bins → softmax → pixel distribution
4. Loss: cross-entropy NLL (nats/pixel)

### vmap pattern

Both encoder and decoder use nested vmap — keep this pattern consistent:

```python
# Encoder: outer vmap over batch, inner over 784 pixels
def embed_image(bins_i):
    return jax.vmap(embed_pixel, in_axes=(0, 0, 0))(ROWS, COLS, bins_i)
emb = jax.vmap(embed_image)(bins)        # (B, 784, D)

# Decoder: outer vmap over batch grams, inner over 784 positions
def decode_image(g):
    return jax.vmap(decode_pixel, in_axes=(0, 0, None))(ROWS, COLS, g)
logits = jax.vmap(decode_image)(gram)    # (B, 784, N_VAL_BINS)
```

`ROWS` and `COLS` are precomputed module-level constants — always keep them there.

---

## Loss & KL Annealing

```
total_loss = recon_NLL + β × KL
```

- **recon_NLL**: cross-entropy averaged over pixels and batch (nats/pixel).
  Starts near `log(N_VAL_BINS) ≈ 3.47` at random init.
- **KL**: averaged over batch and latent dims. Total KL per image ≈ `KL × LATENT_DIM`.
- **β annealing**: ramp β from 0 → `KL_WEIGHT` over `KL_WARMUP_EPOCHS`. Prevents
  the encoder from collapsing to the prior before the decoder has learned to use z.

**FREE_BITS** — `kl_loss = max(kl_per_dim, FREE_BITS).mean()`:
- Forces every latent dimension to encode at least `FREE_BITS` nats of signal.
- Prevents posterior collapse (KL → 0) in low-information dimensions.
- **Must stay at 0.5.** Lowering it risks partial collapse (seen in exp6 at 0.3).

Signs of posterior collapse: KL → 0, reconstructions look identical regardless of input.

---

## Key Hyperparameters

| Parameter | Role | Exp5 value |
|-----------|------|-----------|
| `D_R, D_C, D_V` | Embedding dims per axis | 6, 6, 4 |
| `D = D_R×D_C×D_V` | Gram space dimension | 144 |
| `N_ENC_Q` | Query vectors into the gram | 8 |
| `LATENT_DIM` | Bottleneck size | 64 |
| `DEC_RANK` | Rank of pseudo-gram `A@B.T` | 16 |
| `MLP_HIDDEN` | Decoder hidden dim | 256 |
| `N_VAL_BINS` | Pixel quantisation bins | 32 |
| `FREE_BITS` | Min KL per latent dim | 0.5 (locked) |

**Dominant params**: `W_A` and `W_B` each have shape `(MLP_HIDDEN, DEC_RANK × D)`.
These dominate total param count. Reduce `DEC_RANK` or `MLP_HIDDEN` to shrink the model.

---

## Param Efficiency (exp19 baseline)

The model is massively over-parameterised at exp5's scale (5M params).
Reducing to MLP_HIDDEN=256 and DEC_RANK=16 gives:

| Exp | Params | Acc | Time |
|-----|--------|-----|------|
| 5   | 5.0M   | 88.7% | 262s |
| 19  | 1.4M   | 88.6% | 243s |
| 20  | 436K   | 88.5% | 241s |

Use **exp19** (MLP_HIDDEN=256, DEC_RANK=16, LATENT_DIM=64) as the default
baseline for new experiments. It is 3.5× smaller with negligible accuracy loss.

---

## Information-Theoretic Ceiling

**LATENT_DIM × FREE_BITS = minimum nats per image.**

At LATENT_DIM=64 and FREE_BITS=0.5: the model encodes exactly 32 nats/image.
The encoder already saturates this budget from the first attempt (exp5).
No encoder change — more queries, larger gram, non-linear projection — can push
past 88.7% because the bottleneck is the bottleneck itself, not the encoder.

| Config | Bin accuracy | Prior samples |
|--------|-------------|---------------|
| LATENT_DIM=64  (exp5/19) | 88.7% ceiling | Usable digits |
| LATENT_DIM=512 (exp11)   | 89.2%          | Broken (checkerboard) |

---

## Prior Hole Problem

With large LATENT_DIM (512) and FREE_BITS=0.5, all 512 dimensions must encode
signal. The aggregate posterior becomes a complex 512-dimensional distribution
that a simple N(0,I) prior cannot approximate. Samples z~N(0,1)^512 land
off-manifold and produce checkerboard artifacts (Kronecker product basis
functions bleeding through the decoder).

LATENT_DIM=64 avoids this — the aggregate posterior in 64D is close enough to
the prior that samples look like digits.

**Rule**: larger LATENT_DIM → better reconstruction, worse prior samples.
There is no architectural fix within a single-stage VAE.

---

## Metrics to Collect

| Metric | What it tells you |
|--------|------------------|
| Bin accuracy | Fraction of pixels where argmax(logits) == true bin. Primary metric. |
| recon NLL (nats/pixel) | Reconstruction quality — lower is better |
| KL (mean over dims) | Latent utilisation — should be ≥ FREE_BITS |
| Total loss | Actual training objective |

Store `recon_curve` and `kl_curve` as full per-epoch lists in `results.jsonl`
so plots can be regenerated without retraining.

---

## Visualizations to Generate

Always generate these at the end of every experiment (via `lib/viz.py`):

| Function | What it shows |
|----------|--------------|
| `save_learning_curves` | Recon NLL, KL, bin accuracy over epochs |
| `save_reconstruction_grid` | Original vs reconstructed images |
| `save_sample_grid` | Prior samples z~N(0,1) decoded with soft mean — **never argmax** |
| `save_pixel_entropy_grid` | Per-pixel softmax entropy — reveals model uncertainty |
| `save_latent_pca` | 2D PCA of posterior means coloured by class |

**Never sharpen sample images** (no argmax on logits). Use soft mean:
`(softmax(logits) × bin_centers).sum(-1)`. Sharpening hides blur and makes
broken prior samples look better than they are.
