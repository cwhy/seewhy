# exp18 — Scale-fix Initialization Report

**Date:** 2026-03-21  **Epochs:** 25  **n_pop:** 128  **σ:** 0.001  **rank:** 8

## Hypothesis

Initialize emb at scale `784^(-1/6) ≈ 0.330` so `M = emb@emb.T` has `O(1)` entries from epoch 0, eliminating the plateau caused by near-zero gradient signal.

Scale derivation: `M[i,i] = Σ_{784} emb² ≈ 784·s⁶ = 1  →  s = 784^(-1/6)`

## Results

| Variant | Train | Test | Time | Plateau escape |
|---|---|---|---|---|
| Baseline (scale=0.02)           | 88.94% | 89.79% | 327s | ep 7 |
| Scale fix (scale≈0.329)    | 77.60%  | 78.06%  | 322s  | ep 1 |

## Accuracy curves

![accuracy](https://media.tanh.xyz/seewhy/26-03-21/exp18_accuracy.png)

## |fitness| over time

![fitness](https://media.tanh.xyz/seewhy/26-03-21/exp18_fitness.png)