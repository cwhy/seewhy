# Recall-GAN — Concepts

An in-context GAN. The discriminator reads a context of M images into one KDA
state and must say whether a query image is one of them. The real images of
the GAN are the context images; the fake images are the generator's. The
discriminator is the product: the goal is that its final KDA state becomes an
estimator of "is this image in my context?" and that it wins the game.

This file defines the current method and metrics. What was run, and why the
method looks like this, is in [experiment_log.md](experiment_log.md).

## Task / data

MNIST, pixels in [0,1], flattened to 784. Training episodes come from the 60k
train split; every evaluation uses the 10k test split, so every evaluated
context is made of images the networks never trained on.

One episode: M = 16 context images, drawn at random from the pool, and 8
queries:

| query | source | label |
|---|---|---|
| 4 × real | copies of random context images | 1 ("in context") |
| 4 × fake | generator outputs | 0 ("not in context") |

Balanced 4/4, so chance accuracy is 0.5. Nothing else is queried in training:
unseen real images never appear as queries.

## Model

**Discriminator.** Shared MLP encoder `e(x)` (784 → 512 → 256, GELU). The
context is written by one KDA (delta-rule linear attention) layer, 4 heads,
dk = 64, so the state is 4 × 64 × 64 = 16,384 numbers:

```
S <- S . Diag(alpha_t)                alpha = sigmoid(W_a e + b_a), init horizon 8M
S <- S + beta_t (v_t - S k_t) k_t^T   k = l2norm(W_k e), v = W_v e, beta = sigmoid(W_b e)
o  = S q / sqrt(dk)                   q = l2norm(W_q e(query))
logit = w^T gelu(W_h [o ; e(query)])
```

The final state `S` is the only path from the context to the logit.

**Generator** (two versions). *Context-aware*: its own encoder and KDA state
over the same context, read with `q = W_q z`; `[z ; o]` → MLP (512, 512) →
sigmoid(784). *Blind*: `z` → the same MLP, no state. `z` is 64-d Gaussian.

## Loss and recipe

- D: binary cross-entropy over the 8 queries, plus a gradient penalty on the
  real queries: `(γ/2) E ||∂D/∂x||²`, γ = 10 (R1, Mescheder et al. 2018).
- G: non-saturating, `-log sigmoid(D(ctx, G(ctx, z)))`.
- One D step then one G step per iteration; 64 episodes per step; 20k steps.
- Adam, learning rate 2e-4 for both, **β1 = 0, β2 = 0.99** (StyleGAN's
  setting). This is what made training reliable (0 failures in 12 runs).

Code: `lib/model.py`, `lib/train.py` (`Cell(arm="ctx", q_mem=4, q_non=0,
q_gen=4, r1=10.0, beta1=0.0, adam_b2=0.99)`), e.g. `experiments15.py`.

## Metrics

All evaluation-only, on held-out episodes.

- **Class probe** (`class_acc`) — the main indicator. One context per digit,
  each of 16 test images of that digit; each of 1,000 other test images is
  assigned the digit whose context scores it highest. Chance 0.10. Never
  trained on: no labels, no single-digit contexts in training.
- **Swap test** (`auroc_gen_swap` vs `auroc_gen`) — AUROC of context images vs
  generated images, scored with the episode's own context and with another
  episode's. Equal = D ignores its KDA state; lower with the wrong context = D
  relies on it.
- **Context vs unseen real** (`auroc_real`) — AUROC of context images vs real
  test images not in the context. Chance 0.5.
- **Generator context use** (`g_ctx_dep`) — output change when the generator's
  context is swapped, divided by the change when its noise is redrawn. 0 =
  ignores the context.
- **Generator variety** (`gen_std`) — pixel std across samples; < 0.06 =
  collapsed.
- **Failure** (fixed before exp14): class probe ends < 0.5, or variety ends
  < 0.06.

**No pixel L2 as a reference for similarity** (decision 2026-10-08). L2 is one
arbitrary measure, not the truth. Pixel nearest-neighbour and pixel RBF
density may appear as *baselines*, never as the yardstick. Metrics built on L2
(`closer_gen`, `closer_real`, `d_*_gen`) are still logged by the code but are
superseded.
