# Better optimisers cut DUST's loss, but backprop's more

At 1M tokens, Muon takes DUST with 1,024 draws from **5.939** under SGD to **5.466** (one seed), a gain of
0.47 nats. Backprop gains more, from **5.937** to **5.358** (0.58), so DUST goes from level with backprop
(+0.002) to 0.09 nats behind it (Muon against Muon). At 256 draws, three seeds, DUST goes 6.074 →
5.860 under Adam and 5.617 under Muon, backprop 5.937 → 5.358 and 5.372.
At 4,096 draws (one seed) DUST under Muon scores 5.408, 0.036 nats behind backprop under Muon, where under SGD it was 0.047 ahead. Muon is clearly the better optimiser for DUST; for backprop Adam and Muon are level.
EGGROLL, which was already run with Adam, is not helped by Muon. It is slightly worse, by a few seed standard deviations (0.01 to 0.015) but far less than the others gain: 64 passes +0.052, 256 passes +0.029 nats against its Adam arm.

## The two optimisers, in short

*SGD with momentum* steps along a running average of the gradient, with one learning rate per group of parameters.
*Adam* divides each coordinate's step by the typical size of that coordinate's gradient, so every weight moves by about the learning
rate whatever its gradient's scale. *Muon* works on a weight *matrix* as a whole: it takes the momentum matrix, replaces it by the
nearest matrix whose singular values are all equal (five Newton–Schulz steps), and steps along that. The update has no dominant direction by
construction. Embeddings, the head and the scalars keep Adam. Both optimisers are scale-free, so the unknown constant in a
zeroth-order estimate does not matter to them.

Why try Muon on these estimates: an EGGROLL estimate of a matrix is a sum of rank-one terms, so its spectrum is very uneven, and Muon flattens spectra.

## What was run

Paper's Table 7 learning-rate groups for Adam: DUST matrices and head at the rate in the grid, token and value embeddings at 300× that, residual scalars at 100×,
β₁ = 0.7; backprop embeddings 0.3 and scalars 0.2, β₁ = 0.8; EGGROLL 0.003 throughout, β₁ = 0.5; β₂ = 0.95 for all. Under Adam the paper also re-weights DUST's
draws toward the token embedding, at noise 0.4. Its per-block split is not given, so the SGD recipe's tilt is kept and rescaled; the draws
per update are 1.375 times the nominal population. Muon: Moonlight scaling, so its learning rate is comparable to Adam's. Grids on seed 42, then the winner on three seeds (one seed at 1,024 and 4,096).

| method | population | optimiser | matrix learning rate → test loss (seed 42) |
|---|---:|---|---|
| bp | – | adam | 0.001 → 5.513, 0.002 → 5.342, 0.004 → 6.703 |
| bp | – | muon | 0.001 → 5.513, 0.003 → 5.381, 0.01 → 5.576 |
| dust | 64 | adam | 0.001 → 6.329, 0.00175 → 6.035, 0.003 → 7.021 |
| dust | 64 | muon | 0.001 → 6.138, 0.003 → 5.850, 0.01 → 6.269 |
| dust | 256 | adam | 0.001 → 5.981, 0.0015 → 5.895, 0.003 → 7.102 |
| dust | 256 | muon | 0.001 → 5.883, 0.003 → 5.613, 0.01 → 6.018 |
| eggroll | 64 | muon | 0.001 → 7.865, 0.003 → 7.827, 0.01 → 7.870 |
| eggroll | 256 | muon | 0.001 → 7.484, 0.003 → 7.427, 0.01 → 7.460 |

The same cell run twice (seed 42, 256 draws, Adam at 0.0015) scored 5.895 and 5.978: identical for the first four updates, then apart. That is GPU floating-point nondeterminism amplified over training, so a single-seed grid under Adam has about 0.08 nats of noise before seeds differ at all.

Rate check at 1,024 draws under Muon (seed 42): 0.002 → 5.488, 0.003 → 5.466, 0.005 → 5.547: the carried-over rate was the best of the three by validation loss.

## Results

![DUST and backprop under SGD, Adam and Muon](https://media.tanh.xyz/seewhy/26-10-08/optimiser_zoom_h.svg)

DUST at each population under the three optimisers (violet, light to dark) and backprop under the same three as flat lines (red, light to dark; Adam and Muon, 5.358 and 5.372, nearly coincide).

![All three methods under each optimiser](https://media.tanh.xyz/seewhy/26-10-08/optimiser_ladder_h.svg)

The same, with EGGROLL (grey): its loss is 1.30 to 1.95 nats above DUST under the same optimiser's, which is why the figure above leaves it out.

| | SGD (report 4) | Adam | Muon |
|---|---:|---:|---:|
| backprop | 5.937 ± 0.027 | 5.358 ± 0.014 | 5.372 ± 0.009 |
| DUST, 64 draws | 6.315 ± 0.043 | 6.036 ± 0.006 | 5.864 ± 0.020 |
| DUST, 256 draws | 6.074 ± 0.020 | 5.860 ± 0.103 | 5.617 ± 0.011 |
| DUST, 1,024 draws | 5.939 ± 0.010 | 5.836 (n=1) | 5.466 (n=1) |
| DUST, 4,096 draws | 5.890 (n=1) | – | 5.408 (n=1) |
| EGGROLL, 64 passes | – | 7.764 ± 0.011 | 7.816 ± 0.015 |
| EGGROLL, 256 passes | – | 7.409 ± 0.012 | 7.438 ± 0.011 |
| EGGROLL, 1,024 passes | – | 7.133 ± 0.004 | 7.148 (n=1) |

Three seeds except 1,024 and 4,096 draws and 1,024 passes (one seed, at the 256-population winner's rate). SGD rows are report 4's, EGGROLL's Adam row is the ladder of report 4.

DUST minus backprop, under each optimiser (negative means DUST is ahead):

| population | SGD | Adam | Muon |
|---:|---:|---:|---:|
| 64 | +0.378 | +0.678 | +0.493 |
| 256 | +0.137 | +0.502 | +0.245 |
| 1,024 | +0.002 | +0.478 | +0.095 |
| 4,096 | -0.047 | – | +0.036 |

![DUST minus backprop under the same optimiser](https://media.tanh.xyz/seewhy/26-10-08/optimiser_gap_h.svg)

![Validation loss over training, backprop and DUST at 256 draws](https://media.tanh.xyz/seewhy/26-10-08/optimiser_curves_h.svg)

Validation loss over the 61 updates, mean over seeds: backprop (red) and DUST at 256 draws (violet) under SGD, Adam and Muon, light to dark.

## What this changes

- Both optimisers lower DUST at every population, and Muon lowers it more: 6.32 → 5.86 at 64 draws (0.45 nats), 5.94 → 5.47 at 1,024.
  The paper's claim that DUST works under Adam holds in that sense. Adam, in this port, is the weaker of the two for DUST, by 0.17 to 0.37 nats.
- Backprop gains more (0.58 nats against 0.45 to 0.47 for DUST under Muon at 64 to 1,024 draws), so report 4's headline, DUST level with backprop at 1,024 draws, is a statement
  about SGD. Under Muon the same population leaves DUST 0.09 nats behind (one seed); under Adam it leaves it 0.48 behind, but that cell is
  one seed at a rate taken from a 256-draw grid with 0.08 nats of noise, so it is weak evidence.
- At 4,096 draws, DUST under Muon scores 5.408 (one seed), 0.036 nats behind backprop under Muon. Under SGD the same population was 0.047 ahead. The distance under Muon at 64, 256, 1,024 and 4,096 draws is +0.493, +0.245, +0.095, +0.036: it shrinks by more than half with each quadrupling of the population and has not reached zero. Four points, the last on one seed, do not say whether it will.
- The paper's own Adam section (its Section 3) fits a power law to DUST's Adam ladder and reports a limit of 5.248, below its backprop's 5.361.
  Our backprop under Adam, 5.358, agrees with theirs. Our DUST under Adam is 0.48 nats behind it at 1,024 draws, and three populations do not support a fitted limit.
- EGGROLL is not helped by Muon (slightly worse at all three populations, 1,024 on one seed). A reading that fits the alignment result of report 3, offered as a hypothesis and not tested here: an estimate whose cosine to
  the gradient is about 0.01 has no spectrum worth flattening, and normalising noise does not produce signal.
- Not tested: EGGROLL's own tuning at the new settings (its Muon rate was reused across populations), a rate search at 4,096 draws (1,024 got a check, above), DUST under Adam at 4,096, and the paper's exact draw
  split under Adam (its per-block numbers are not given; the draws here are 1.375 times the nominal population).

Code: `lib/optim.py`, `experiments19.py` to `experiments28.py`, `scripts/gpu_queue.py`.
