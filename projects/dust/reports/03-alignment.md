# DUST's gradient aligns with backprop's; EGGROLL's stays near zero

On the 100M-token checkpoint, DUST's gradient estimate at 16,384 draws has cosine **0.989** with
backprop's on the head, **0.728** on attention outputs, **0.625** on the MLPs and
**0.412** on values. EGGROLL at 16,384 forward passes: **0.024** on the head and at most
**0.014** everywhere else. A random-direction estimate on this network should land there, and does. This is
the mechanism behind the ladder in the next report: the two methods feed the optimiser gradients of very different quality.

## What is measured

Take one batch of eight held-out sequences and one set of weights. Backprop gives the gradient. Each zeroth-order method
gives an estimate of it from forward passes. The **cosine** between the two, layer by layer, says how much of the estimate
points along the truth: 1.0 is backprop itself, 0 is no information. Population is draws (DUST) or forward passes (EGGROLL).

**What EGGROLL's cosine must be.** One population member is one number, and that number has to steer every parameter.
Average `n` antithetic pairs along random directions and the error in layer `l` is isotropic noise of energy
`d_l·|g|²/n`, where `d_l` is the layer's size and `|g|` the norm of the *whole* gradient, because every pair's score is
a sum over all layers. With `s_l = |g_l|/|g|` the layer's share of the gradient norm,

    cos_l(n) = s_l / sqrt( s_l² + d_l / n )  ≈  s_l · sqrt(n / d_l)

![EGGROLL on one weight matrix](https://media.tanh.xyz/seewhy/26-10-08/dust_eggroll_update.svg)

The shares are measured, not assumed (`scripts/eggroll_theory.py`). At initialisation the head holds
99.7% of the squared gradient norm, because the head's weights start near zero; at this checkpoint it holds
14%. Every other layer is a small share of a gradient that EGGROLL must find
by random search, so its cosine there is tiny whatever the optimiser does with it.

## DUST: the cosine climbs with population

![DUST cosine to backprop, 100M-token checkpoint](https://media.tanh.xyz/seewhy/26-10-08/cosine_dust_bp100M_step6100_b.svg)

Cosine to backprop, 100M-token checkpoint, mean over noise draws. Last column: the paper's fitted law
`c_max / sqrt(1 + c/K)` (their Table 8, same checkpoint size) at K = 16,384.

| layer type | K = 64 | K = 256 | K = 1,024 | K = 4,096 | K = 16,384 | paper's law, K = 16,384 |
|---|---:|---:|---:|---:|---:|---:|
| LM head | 0.385 | 0.643 | 0.859 | 0.959 | 0.989 | 0.978 |
| attn out | 0.102 | 0.166 | 0.347 | 0.532 | 0.728 | 0.573 |
| MLP (in, out mean) | 0.064 | 0.093 | 0.233 | 0.408 | 0.625 | 0.472 |
| token emb | 0.062 | 0.082 | 0.203 | 0.368 | 0.552 | 0.494 |
| V | 0.005 | 0.013 | 0.050 | 0.173 | 0.412 | 0.258 |
| Q | 0.010 | 0.020 | 0.105 | 0.294 | 0.574 | 0.419 |
| K | 0.004 | 0.004 | 0.024 | 0.082 | 0.212 | 0.161 |
| value gate | -0.029 | -0.027 | 0.043 | 0.174 | 0.410 | 0.336 |
| value emb | -0.003 | 0.004 | 0.011 | 0.058 | 0.188 | 0.179 |

The curves have the shape of the paper's law. The head is near its ceiling by 4,096 draws (0.959).
Attention outputs (0.728), MLPs (0.625) and queries
(0.574) are still climbing at 16,384. Keys (0.212) and value embeddings
(0.188) are lowest. At K = 16,384 our cosines sit above the paper's law for every type, including
the head, by 0.01 to
0.16.
The law is fitted to their checkpoint and their per-population estimator constants, so the offset is not a defect.
Fitting `c_max / sqrt(1 + c/K)` to our five populations and setting it beside the paper's (`c_max, c`):

| layer type | ours | paper's Table 8 |
|---|---|---|
| LM head | 1.00, 365 | 1.00, 759 |
| attn out | 0.84, 5,485 | 0.87, 21,627 |
| MLP (in, out mean) | 0.85, 13,591 | 0.87, 40,020 |
| token emb | 0.74, 13,041 | 0.73, 19,498 |
| V | 1.00, 100,666 | 0.88, 173,780 |
| Q | 1.00, 41,155 | 1.00, 76,736 |
| K | 1.00, 421,791 | 1.00, 616,595 |

Five points cannot pin the ceiling of a curve that has not flattened, so only the half-saturation scale `c` is
informative for the slow types.

## EGGROLL: where random directions put it

![DUST and EGGROLL, 100M-token checkpoint](https://media.tanh.xyz/seewhy/26-10-08/cosine_vs_eggroll_bp100M_step6100_b.svg)

Measured cosine at each population, with the prediction `cos_l(n)` above in brackets (n = passes ÷ 2 pairs):

| layer type | 64 passes | 256 passes | 1,024 passes | 4,096 passes | 16,384 passes |
|---|---:|---:|---:|---:|---:|
| LM head | 0.002 (pred. 0.001) | 0.003 (pred. 0.003) | 0.006 (pred. 0.006) | 0.011 (pred. 0.012) | 0.024 (pred. 0.023) |
| attn out | 0.000 (pred. 0.001) | 0.001 (pred. 0.001) | 0.003 (pred. 0.003) | 0.005 (pred. 0.005) | 0.010 (pred. 0.010) |
| MLP (in, out mean) | 0.001 (pred. 0.001) | 0.002 (pred. 0.001) | 0.003 (pred. 0.003) | 0.006 (pred. 0.006) | 0.011 (pred. 0.012) |
| token emb | 0.000 (pred. 0.000) | -0.000 (pred. 0.000) | 0.000 (pred. 0.000) | 0.000 (pred. 0.000) | -0.000 (pred. 0.000) |
| V | 0.001 (pred. 0.001) | 0.002 (pred. 0.002) | 0.004 (pred. 0.003) | 0.007 (pred. 0.007) | 0.014 (pred. 0.013) |
| Q | 0.001 (pred. 0.000) | 0.001 (pred. 0.001) | 0.001 (pred. 0.001) | 0.003 (pred. 0.003) | 0.005 (pred. 0.006) |
| K | 0.000 (pred. 0.000) | 0.002 (pred. 0.001) | 0.002 (pred. 0.002) | 0.004 (pred. 0.004) | 0.005 (pred. 0.007) |

## Alignment at a fixed population falls as the network trains

Cosine at K = 16,384 draws, DUST, on the initial weights and on the two checkpoints. It is lower after training than at
initialisation for 6 of these 7 layer types.

| layer type | initialisation | 10M tokens | 100M tokens |
|---|---:|---:|---:|
| LM head | 0.999 | 0.995 | 0.989 |
| attn out | 0.826 | 0.690 | 0.728 |
| MLP (in, out mean) | 0.796 | 0.630 | 0.625 |
| token emb | 0.850 | 0.887 | 0.552 |
| V | 0.648 | 0.421 | 0.412 |
| Q | 0.694 | 0.587 | 0.574 |
| K | 0.142 | 0.202 | 0.212 |

## What this changes

- The ladder gap has a cause we can measure. At 16,384 the two methods hand the optimiser gradients whose cosine differs
  by a factor of 29 to 126, depending on the layer.
- EGGROLL needs no explanation beyond arithmetic: its cosine is `s·sqrt(n/d)`, and `d` is 3.8×10⁷.
- DUST's weakest layers (keys and value embeddings) are credited through the *estimated* attention-output error rather than
  the loss, so their error compounds the attention-output estimate's own. They improve slowest with population. If the
  ladder flattens before backprop, look there first.
- One batch of eight sequences, a few noise draws per cell, and our own 100M-token checkpoint (trained with the paper's
  10M-token optimiser settings): the shapes are firm, the third decimal is not.

Code: `lib/align.py`, `scripts/eggroll_theory.py`, `experiments6.py`.
