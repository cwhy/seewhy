# exp22 — Single-sided ES Report

**Date:** 2026-03-21  **Epochs:** 100  **σ:** 0.001  **rank schedule:** 8→2 at ep12

## Hypothesis

Antithetic pairs destroy individual CE values needed for DG.
Single-sided ES with n_pop=256 has same compute as antithetic n_pop=128.
With single-sided, ℓ_i = CE_i is directly available as action surprisal.

## Results

| Variant | Config | n_pop (evals/batch) | Train | Test | Time |
|---|---|---|---|---|---|
| A-antithetic     n=128 (256 evals) | antithetic | 128 (256) | 94.14% | 93.89% | 1288s |
| B-single-sided   n=256 (256 evals) | single-sided | 256 (256) | 95.68% | 95.39% | 1284s |
| C-single+DG η=0.01  n=256 | single-sided DG=0.01 | 256 (256) | 93.50% | 93.63% | 1281s |
| D-single+DG η=auto  n=256 | single-sided DG=auto | 256 (256) | 93.28% | 93.48% | 1280s |

## Accuracy curves

![accuracy](https://media.tanh.xyz/seewhy/26-03-21/exp22_accuracy.png)

## Diagnostics

![diagnostics](https://media.tanh.xyz/seewhy/26-03-21/exp22_diagnostics.png)