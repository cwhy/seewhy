# exp21 — DG-ES v2 Report

**Date:** 2026-03-21  **Epochs:** 100  **n_pop:** 128  **σ:** 0.001  **rank schedule:** 8→2 at ep12

## Fix over exp20

exp20 used perturbation norm as surprisal (chi-squared noise, no semantic content).
v2 uses the model's own cross-entropy: `ℓ_i = (CE_+ + CE_-)/2`.
Delight: `χ_i = -fitness_i × CE_avg_i` (sign: positive when pair helped AND model was uncertain).

## Results

| Variant | Config | Train | Test | Time |
|---|---|---|---|---|
| A-standard | rank=8→2 ep12 (control) | 93.96% | 93.89% | 1285s |
| B-DG-v2  η=0.01 | DG-v2 η=0.01, rank=8→2 | 91.18% | 91.09% | 1282s |
| C-DG-v2  η=0.001 | DG-v2 η=0.001, rank=8→2 | 83.03% | 84.08% | 1283s |

## Accuracy curves

![accuracy](https://media.tanh.xyz/seewhy/26-03-21/exp21_accuracy.png)

## Diagnostics

![diagnostics](https://media.tanh.xyz/seewhy/26-03-21/exp21_diagnostics.png)