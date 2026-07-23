# exp52 — Local IMLE, VOCAB=64, N_VAR=128, 400ep

**HTML report**: https://media.tanh.xyz/seewhy/26-04-26/variations_exp52_report.html

## Result

| Metric | Value |
|--------|-------|
| Final recall MSE | **0.00610** (new best) |
| Final recon MSE | 0.00082 |
| Training time | ~285 min |
| Epochs | 400 |

Beats exp49 (0.00658, VOCAB=32). VOCAB scaling trend continues.

## Key change from exp49

`VOCAB_SIZE 32→64`: gives 64⁴ ≈ 16M conditioning codes vs 32⁴ ≈ 1M.
Near-zero compute cost (embedding table: 4×64×16 = 4096 params).
400 epochs needed (vs 300 for VOCAB=32) — sparser embedding gradients early.

## VOCAB Scaling Trend

| Exp | VOCAB | Code space | Epochs | Recall MSE |
|-----|-------|-----------|--------|------------|
| exp45 | 8 | 8⁴ = 4,096 | 150ep | 0.00703 |
| exp49 | 32 | 32⁴ ≈ 1M | 300ep | 0.00658 |
| **exp52** | **64** | **64⁴ ≈ 16M** | **400ep** | **0.00610** |

Consistent improvement with each VOCAB doubling. Still slowly improving at ep399 —
VOCAB=128 with 500+ epochs likely pushes further.
