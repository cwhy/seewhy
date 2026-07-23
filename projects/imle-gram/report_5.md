# Report 5 — Tensor Product Decoder (exp6, exp7)

## Goal

Replace the gram decoder with a gram-free `r⊗c⊗h` tensor product and scale
up to match the 87.1% gram ceiling.

## Architecture

```
z (64) → h = GELU(z @ W_mlp1)          # (MLP_HIDDEN,) shared

per pixel (r, c):
f = (row_emb[r][:,None,None] * col_emb[c][None,:,None] * h[None,None,:]).reshape(-1)
h1 = GELU(W_val1 @ f)
logits = W_val2 @ h1                    # (32,)
```

## Results

| Exp | MLP_HIDDEN | f_dim | Params | Bin acc | Time |
|-----|-----------|-------|--------|---------|------|
| exp6 | 64 | 2,304 | 154K | 86.9% | 490s |
| exp7 | 128 | 4,608 | 603K | **87.0%** | 808s |
| exp1 (gram baseline) | 256 | — | 1.25M | 87.1% | 330s |

## Findings

- **Gram decoder was unnecessary.** The tensor product reaches the same ceiling at
  half the parameters (603K vs 1.25M).
- **The ceiling is the ceiling.** Every architecture variation — gram, tensor product,
  64K–4.7M params — converges to 87.0–87.2%. The bottleneck is in the IMLE
  matching, not the decoder.
- **Speed regressed.** Larger f_dim (4608) means heavier per-pixel matmuls inside
  vmap — 808s vs 330s for the gram baseline.

## Visualizations

- exp6 recon: https://media.tanh.xyz/seewhy/26-03-23/exp6_recon_epoch100.png
- exp6 samples: https://media.tanh.xyz/seewhy/26-03-23/exp6_samples_5x8_seed0.png
- exp7 recon: https://media.tanh.xyz/seewhy/26-03-23/exp7_recon_epoch100.png
- exp7 samples: https://media.tanh.xyz/seewhy/26-03-23/exp7_samples_5x8_seed0.png

## Next

Try different latent variable distributions (non-Gaussian z) to test whether
the IMLE matching ceiling can be broken by changing the prior.
