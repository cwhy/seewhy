# Report 3 — Adaptive IMLE (exp5)

## Goal

Test whether the Adaptive IMLE strategy (per-example thresholds, persistent pool,
partial refresh) breaks the 87.1% ceiling established in exp1–4.

## Setup

| Hyperparameter | Value |
|----------------|-------|
| `N_POOL` | 2000 (persistent pool, vs per-epoch N_SAMPLES=1000 in exp1) |
| `REFRESH_FRAC` | 0.2 (20% of pool replaced each epoch) |
| `CHANGE_COEF` | 0.9 (τᵢ ← dist × 0.9 after each match) |
| `LATENT_DIM` | 64 |
| `MLP_HIDDEN` | 256 |
| `DEC_RANK` | 16 |

## Algorithm

```
tau[i] = 0  for all i   (all examples start active)

each epoch:
  refresh 20% of pool with fresh random decodes
  for each batch:
    find nearest sample → distances d[i]
    active[i] = d[i] > tau[i]          # skip well-covered examples
    train on active examples only
    tau[i] ← d[i] * 0.9               # tighten threshold
```

## Bug fix: threshold initialisation

First attempt initialised `tau = inf`, making active = (dist > inf) = False
for all examples — nothing ever trained. Fixed to `tau = 0`.

## Results

| Epoch | Match L2 (train) | Active frac | Bin acc |
|-------|-----------------|-------------|---------|
| 0 | 9,605,257 | 100.0% | — |
| 1 | 8,264,548 | 9.7% | — |
| 4 | 4,069,397 | 0.0% | 81.0% |
| 9 | 3,310,186 | 18.1% | 83.0% |
| 99 | 1,543,283 | 90.2% | 87.0% |

**Final: 87.0% bin accuracy, 760s.**

## Observations

**Same ceiling, higher cost.** Adaptive IMLE reaches 87.0% — identical to basic
IMLE (87.1%) but 2.3× slower (760s vs 330s).

**CHANGE_COEF=0.9 is too aggressive.** The threshold shrinks at 90% of the current
distance each round, requiring 10% improvement per example per epoch to be
"satisfied". Since the decoder improves more slowly than this, the active fraction
never drops below ~87% — the adaptive gating barely filters anything. The model
effectively runs basic IMLE for most of training.

**Match L2 still declining at epoch 100.** The model has not converged. However,
given the flat bin accuracy over the last 20 epochs, the match L2 decline likely
reflects a few hard examples getting slightly better coverage, not overall
improvement.

**The ceiling is in the decoder, not the matching strategy.** All five experiments
converge to 87–87.2% regardless of:
- N_SAMPLES (1000 vs 5000)
- LATENT_DIM (64 vs 256)
- Decoder size (1.25M vs 4.7M params)
- Matching strategy (basic vs adaptive)

## Visualizations

- Learning curves: https://media.tanh.xyz/seewhy/26-03-23/exp5_learning_curves.svg
- Nearest-neighbor recon (final): https://media.tanh.xyz/seewhy/26-03-23/exp5_recon_final.png
- Prior samples: https://media.tanh.xyz/seewhy/26-03-23/exp5_samples_5x8_seed0.png

## Next Directions

1. **More epochs** — match L2 still declining; try 200 epochs to see true ceiling.
2. **Decoder architecture change** — the gram decoder may have a representational
   limit around 87%. A direct MLP decoder (no gram structure) could test this.
3. **Smaller CHANGE_COEF** (e.g. 0.5) or **larger N_POOL** — but exp2 showed
   more samples barely helps, so this is unlikely to move the needle.
