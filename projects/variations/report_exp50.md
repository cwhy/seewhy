# exp50 — exp35 Replica from Scratch, COND_DIM=16, VOCAB=8, 400ep

**HTML report**: https://media.tanh.xyz/seewhy/26-04-26/variations_exp50_report.html

## Result

| Metric | Value |
|--------|-------|
| Final recall MSE | **0.00678** |
| Final recon MSE | 0.00084 |
| Training time | ~285 min |
| Epochs | 400 |

Matches exp35's warm-start result (0.00666) within noise, and both are beaten by exp49 (0.00658).

## Purpose

Validation experiment. exp35 was a warm-start from exp33 (200 pre-trained epochs), making
its 200-epoch run effectively 400 total epochs. exp50 replicates the same hyperparameters
(LATENT=2048, COND_DIM=16, VOCAB=8, K=16, N_VAR=128) from random init with a per-step JIT
loop (~3.4× faster per epoch than exp35's lax.scan).

## Key Finding

exp50 (from scratch) ≈ exp35 (warm-start) → the warm-start gave no architectural advantage.
The gap we spent time investigating was purely epoch count. exp49 (VOCAB=32, COND_DIM=64)
reaching 0.00658 is a genuine improvement, not a warm-start artifact.

## Recall trajectory (last 10 evals)

ep349: 0.00681 → ep354: 0.00681 → ep359: 0.00680 → ep364: 0.00680
→ ep369: 0.00679 → ep374: 0.00679 → ep379: 0.00679 → ep384: 0.00679
→ ep389: 0.00678 → ep394: 0.00678 → **ep399: 0.00678**

Still very slowly improving; likely would reach ~0.00675 with more epochs but not competitive
with exp49's VOCAB=32 architecture.

## Comparison

| Exp | VOCAB | COND_DIM | Init | Epochs | Recall MSE |
|-----|-------|----------|------|--------|------------|
| exp45 | 8 | 64 | random | 150ep | 0.00703 |
| exp35 | 8 | 16 | warm-start | 200ep | 0.00666 |
| **exp50** | **8** | **16** | **random** | **400ep** | **0.00678** |
| exp49 | 32 | 64 | random | 300ep | 0.00658 |
