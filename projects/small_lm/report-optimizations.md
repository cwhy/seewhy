# Optimization experiments — Report

Comparing three architectural/optimizer changes against the baseline:
**Muon optimizer**, **RoPE positional encoding**, and **separate (untied) LM head**.

**Dataset:** 60K synthetic Guppy-fish conversations (57K train / 3K eval)  
**Training:** 10K steps, cosine LR + warmup, batch_size=32, RTX 4090

---

## Results summary

| Experiment | Params | PPL | Time (s) | vs baseline |
|---|---|---|---|---|
| exp_baseline | 8.73M | **1.4614** | 128 | — |
| exp_muon | 8.73M | **1.5162** | 214 | PPL +0.055, time +86s |
| exp_rope | 8.68M | **1.4620** | 138 | PPL +0.001, time +10s |
| exp_no_tie | 10.30M | **1.4617** | 147 | PPL +0.000, time +19s |

![Comparison curves](https://media.tanh.xyz/seewhy/26-04-07/small_lm_optim_comparison.png)

---

## Experiment details

### exp_baseline

Direct GuppyLM port. AdamW optimizer, learned absolute position embeddings, weight-tied LM head. 8.73M parameters.

- **Parameters:** 8,726,016 (8.73M)
- **Final eval PPL:** 1.4614
- **Final eval loss:** 0.3794
- **Training time:** 128s
- **PPL trajectory:** 2.5 → 1.4614 (×2 reduction)

### exp_muon

Muon optimizer: Newton-Schulz orthogonalisation applied to all 2D weight matrix gradients before the update step; AdamW handles embeddings, biases, and LayerNorm. Same architecture as baseline.

- **Parameters:** 8,726,016 (8.73M)
- **Final eval PPL:** 1.5162
- **Final eval loss:** 0.4162
- **Training time:** 214s
- **PPL trajectory:** 4.8 → 1.5162 (×3 reduction)

### exp_rope

RoPE (Rotary Position Embedding) replaces learned positional embeddings. Q and K vectors are rotated by sinusoidal frequencies before the attention dot-product, encoding relative positions. Saves 49K params (no pos_embed).

- **Parameters:** 8,676,864 (8.68M)
- **Final eval PPL:** 1.4620
- **Final eval loss:** 0.3798
- **Training time:** 138s
- **PPL trajectory:** 2.3 → 1.4620 (×2 reduction)

### exp_no_tie

Separate LM head — the output projection `lm_head` is independent of `token_embed`, giving the model ~1.57M extra parameters (~10.3M total). Hypothesis: untied output space allows better calibration.

- **Parameters:** 10,298,880 (10.30M)
- **Final eval PPL:** 1.4617
- **Final eval loss:** 0.3796
- **Training time:** 147s
- **PPL trajectory:** 2.3 → 1.4617 (×2 reduction)

---

## Analysis

**Best PPL:** exp_baseline (1.4614)

**Muon:** did not improve PPL by 0.0548 vs baseline (slower by 86s). Newton-Schulz orthogonalisation constrains weight updates to the Stiefel manifold, effectively acting as a second-order approximation for matrix parameters.

**RoPE:** did not improve PPL by 0.0006 vs baseline. At context=128 and vocab=2375 tokens, absolute position embeddings are unlikely to be a bottleneck — both approaches encode similar positional information for short sequences.

**No-tie:** did not improve PPL by 0.0003 vs baseline (+1.57M parameters). Decoupling input embeddings from the output projection gives the model more freedom but also increases the number of parameters to optimise.

---

## Pareto front: training time vs perplexity

![Pareto front](https://media.tanh.xyz/seewhy/26-04-07/small_lm_pareto.png)

★ = Pareto-optimal experiments (not dominated in both dimensions).