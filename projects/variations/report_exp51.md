# exp51 — Continuous Gaussian Noise Conditioning, COND_DIM=64, 300ep

**HTML report**: https://media.tanh.xyz/seewhy/26-04-26/variations_exp51_report.html

## Result

| Metric | Value |
|--------|-------|
| Final recall MSE | **0.00787** |
| Best recall MSE | 0.00783 |
| Final recon MSE | 0.00084 |
| Training time | ~215 min |
| Epochs | 300 |

Significantly worse than discrete VOCAB=32 (exp49: 0.00658). Plateaued around ep250 at ~0.00784.

## Key Change

Replace discrete `cat_emb` + categorical sampling with `eps ~ N(0, I_64)`:
- Phase 1 saves winning eps vectors (not indices) under stop_gradient
- Phase 2 re-decodes saved eps with full gradient
- Recon anchored at eps=0

No cat_emb parameters — 4096 fewer params than exp49.

## Recall Trajectory

ep5: 0.00956 → ep25: 0.00893 → ep100: 0.00836 → ep150: 0.00813
→ ep200: 0.00793 → ep250: 0.00788 → ep299: **0.00787**

Plateau visible from ~ep250. Improvement rate dropped to near zero.

## Verdict

**Ruled out.** Continuous conditioning is worse than discrete at this scale.
The decoder struggles to learn which eps dimensions matter without the anchor
structure of an embedding table. Discrete codes force well-separated conditioning
vectors that are easier to learn from. VOCAB scaling is the better direction.

## Comparison

| Exp | Conditioning | Code space | Epochs | Recall MSE |
|-----|-------------|-----------|--------|------------|
| exp45 | discrete VOCAB=8 | 8⁴ = 4096 | 150ep | 0.00703 |
| exp49 | discrete VOCAB=32 | 32⁴ ≈ 1M | 300ep | **0.00658** |
| **exp51** | **continuous N(0,I)** | **∞** | **300ep** | **0.00787** |
