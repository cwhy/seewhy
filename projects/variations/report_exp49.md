# exp49 — Local IMLE, VOCAB=32, N_VAR=128, 300ep

**HTML report**: https://media.tanh.xyz/seewhy/26-04-26/variations_exp49_report.html

## Result

| Metric | Value |
|--------|-------|
| Final recall MSE | **0.00658** (new best) |
| Final recon MSE | 0.00090 |
| Training time | ~215 min |
| Epochs | 300 (full schedule, no early stopping) |

Beats exp35 (0.00666, warm-start from 200 pre-trained epochs).

## Key change from exp45

`VOCAB_SIZE 8 → 32`: gives 32⁴ ≈ 1M conditioning codes vs 8⁴ = 4096.
Near-zero compute cost (embedding table: 4×32×16 = 2048 params).

## Recall trajectory (last 10 evals)

ep249: 0.00662 → ep254: 0.00661 → ep259: 0.00660 → ep264: 0.00660
→ ep269: 0.00659 → ep274: 0.00659 → ep279: 0.00658 → ep284: 0.00658
→ ep289: 0.00658 → ep294: 0.00658 → **ep299: 0.00658**

Still very slowly improving at ep299 (~0.00001–0.00002 per eval).
More epochs would give marginal gains; LR is at floor (1.5e-5).

## Findings

- **VOCAB=32 starts slower** (ep5: 0.00949 vs exp45's 0.00929) — sparser embedding gradients early
- **Converges better** given enough epochs — richer code space enables finer conditioning
- VOCAB scaling is free compute-wise; the benefit is purely from expressiveness
- Next: VOCAB=64 or 128 (may need proportionally more epochs)
