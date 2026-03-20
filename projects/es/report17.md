# exp17 — Meta-ES-EA Report

**Date:** 2026-03-20  **Epochs:** 100  **n_pop:** 128

## Results

| Variant | Train | Test | Time |
|---|---|---|---|
| Standard ES (σ=0.001, rank=8, fixed) | 94.41% | 94.42% | 1280s |
| Meta-ES-EA (σ+rank population, EA)   | 75.16% | 76.87% | 1282s |

Final gene pool — σ median: `0.05000` (ref: 0.001)  rank median: `14.1` (ref: 8→2)

## Accuracy curves

![accuracy](https://media.tanh.xyz/seewhy/26-03-20/exp17_accuracy.png)

## |fitness| over time

![fitness](https://media.tanh.xyz/seewhy/26-03-20/exp17_fitness.png)

## Sigma population vs reference σ=0.001

Shaded band = IQR (25–75%). Dashed blue = best known fixed sigma.

![sigma population](https://media.tanh.xyz/seewhy/26-03-20/exp17_sigma_pop.png)

## Rank population vs reference ranks

Shaded band = IQR (25–75%). Dashed blue = rank=8 (warm-up ref), dotted teal = rank=2 (convergence ref).

![rank population](https://media.tanh.xyz/seewhy/26-03-20/exp17_rank_pop.png)