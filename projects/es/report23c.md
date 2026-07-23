# exp23c — Single-sided ES Sweep Report

**Date:** 2026-03-22  **Epochs:** 100  **n_pop:** 256

Baseline: sw=12, σ=0.001 → **95.40%** (exp23-A)

## Results

| Variant | σ | rank_sw | Train | Test | Time |
|---|---|---|---|---|---|
| A sw=12 σ=.001 | 0.001 | 12 | 95.47% | 95.41% | 1283s |
| B sw=8  σ=.001 | 0.001 | 8 | 95.45% | 95.37% | 1275s |
| C sw=8  σ=.0007 | 0.0007 | 8 | 95.17% | 94.84% | 1272s |
| D sw=8  σ=.0005 | 0.0005 | 8 | 93.87% | 93.76% | 1276s |
| E sw=12 σ=.0007 | 0.0007 | 12 | 95.12% | 94.98% | 1275s |
| F sw=6  σ=.001 | 0.001 | 6 | 95.21% | 94.87% | 1275s |

## Accuracy curves

![accuracy](https://media.tanh.xyz/seewhy/26-03-22/exp23c_accuracy.png)