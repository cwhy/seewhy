# exp17 — Meta-ES-EA Report

**Date:** 2026-03-20  **Epochs:** 100  **n_pop:** 128

## Results

| Variant | Train | Test | Time |
|---|---|---|---|
| Standard ES (σ=0.001, rank=8, fixed) | 94.23% | 94.19% | 1126s |
| Meta-ES-EA (σ+rank population, EA)   | 90.24% | 90.45% | 1124s |

Final gene pool — σ median: `0.00053` (ref: 0.001)  rank median: `1.0` (ref: 8→2)

## Diagnosis

**What worked:** once-per-epoch EA update (vs per-batch) stabilised sigma evolution. The model escaped the plateau at ep7 (67.8%) — one epoch earlier than standard ES (ep8) — due to the initial population including high-sigma/rank candidates. Learning continued smoothly to 90.45%.

**Root failure — plateau collapse:** During eps 0–6, all groups had |fitness| ≈ 0 (random init → M matrix is noise → no gradient signal). With near-zero signal, the normalised fitness `|fit| / (σ√rank)` is dominated by noise, making selection essentially random. By chance, the population collapsed toward σ≈5e-4 rank≈1 before learning began. Once diversity was lost, the EA could not recover — mutation is too slow (1.2× per epoch) to explore back to σ=0.001 rank=8 in the remaining epochs.

**Core insight:** EA selection for ES hyperparameters is fundamentally hard during the plateau phase because the selection criterion (gradient quality) is indistinguishable from noise when the gradient is near zero. The rank schedule (exp11/exp15) solves this by explicit design: use rank=8 to force escape, then switch to rank=2. An EA cannot rediscover this schedule without an external signal for when the plateau ends.

**What would fix it:** protect one candidate as a fixed reference (σ=0.001 rank=8) that never mutates, ensuring the population always contains a known-good genome. Selection would then converge toward the reference after escape rather than collapsing away from it.

## Accuracy curves

![accuracy](https://media.tanh.xyz/seewhy/26-03-20/exp17b_accuracy.png)

## |fitness| over time

![fitness](https://media.tanh.xyz/seewhy/26-03-20/exp17b_fitness.png)

## Sigma population vs reference σ=0.001

Shaded band = IQR (25–75%). Dashed blue = best known fixed sigma.

![sigma population](https://media.tanh.xyz/seewhy/26-03-20/exp17b_sigma_pop.png)

## Rank population vs reference ranks

Shaded band = IQR (25–75%). Dashed blue = rank=8 (warm-up ref), dotted teal = rank=2 (convergence ref).

![rank population](https://media.tanh.xyz/seewhy/26-03-20/exp17b_rank_pop.png)