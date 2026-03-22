# exp20 — Delightful ES Report

**Date:** 2026-03-21  **Epochs:** 100  **n_pop:** 128  **σ:** 0.001  **rank schedule:** 8→2 at ep12

## Hypothesis

Apply Delightful Policy Gradient (arxiv:2603.14608) gating to ES.
Gate = σ(fitness_i × centered_surprise_i / η) where surprise = -log p(eps).
Suppresses blunders (rare large eps with negative fitness).

## Results

| Variant | Config | Train | Test | Time |
|---|---|---|---|---|
| A-standard  rank=8→2 | rank=8→2 ep12 | 93.93% | 93.97% | 1127s |
| B-DG-ES  η=1.0 | DG η=1.0, rank=8→2 | 91.55% | 91.77% | 1124s |
| C-DG-ES  η=0.1 | DG η=0.1, rank=8→2 | 91.05% | 91.40% | 1123s |

## Accuracy curves

![accuracy](https://media.tanh.xyz/seewhy/26-03-21/exp20_accuracy.png)

## |fitness| and gate diagnostics

![diagnostics](https://media.tanh.xyz/seewhy/26-03-21/exp20_diagnostics.png)