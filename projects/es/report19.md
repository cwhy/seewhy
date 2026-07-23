# exp19 — Domain-invariant Initialization Report

**Date:** 2026-03-21  **Epochs:** 50  **n_pop:** 128  **σ:** 0.001  **rank:** 8

## Hypothesis

exp18 showed scale-fix eliminates the plateau but converges slower.
Cause: large *random* M has strong signal but no structure.
Here we test inits that give M *structured* signal from ep0.

## Results

| Variant | Init | Train | Test | Time | Plateau escape |
|---|---|---|---|---|---|
| A-baseline | scale=0.02 (random) | 91.83% | 92.15% | 568s | ep6 |
| B-sinusoidal | sin/cos PE (row+col+val) | 88.84% | 89.65% | 562s | ep1 |
| C-ortho-frame | orthogonal frame (row+col+val) | 89.69% | 89.90% | 561s | ep0 |
| D-sinusoidal+ortho-val | sin/cos PE + ortho val | 90.90% | 91.39% | 561s | ep0 |

## Accuracy curves

![accuracy](https://media.tanh.xyz/seewhy/26-03-21/exp19_accuracy.png)

## |fitness| over time

![fitness](https://media.tanh.xyz/seewhy/26-03-21/exp19_fitness.png)