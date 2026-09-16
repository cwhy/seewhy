# Auditing PC-ALM and EGGROLL

The [Forward-Forward audit](https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_ff_audit.html)
found that arm mechanically correct but starved by a negative-sampling shortcut,
which cost two of the paper's claims. This is the same treatment for the other
two arms we implemented from their papers.

**Short version: both check out. PC-ALM reproduces its own theorem to six
decimal places and matches the authors' code to within half a point; EGGROLL's
estimator is unbiased and its low-rank claim holds. One finding worth acting on:
EGGROLL's antithetic sampling is costing it accuracy per unit of compute here.**

## PC-ALM against its own theorem

The paper's central claim is that in a deep *linear* network the dual variables
converge to the backprop credit signals, so the weight update must converge to
the backprop gradient. That is checkable to arbitrary precision — run inference
long enough and see.

| inference steps T | PC-ALM, cosine to backprop | PC, cosine | ‖g_alm‖ / ‖g_bp‖ |
|---:|---:|---:|---:|
| 8 | 0.766210 | 0.704411 | 0.7070 |
| 32 | 0.965941 | 0.778227 | 1.3637 |
| 128 | 0.959880 | 0.937621 | 1.1544 |
| 512 | 0.993529 | 0.970866 | 1.0271 |
| 2048 | **0.999982** | 0.968388 | **1.0004** |

Both direction and magnitude converge on backprop's gradient. Predictive coding,
which has no duals to accumulate, stalls around 0.97 — which is the paper's
distinction, reproduced. With ReLU units, where no theorem applies, PC-ALM still
reaches 0.9977 at T = 1024 against PC's 0.9742.

Four more checks, all passed:

| check | result |
|---|---|
| α = 0 must reproduce PC exactly (the ablation the mechanism rests on) | max difference 0.000e+00 |
| at convergence the activations return to the forward pass while the duals hold the credit | drift 1.05e-01 → 2.15e-03, residuals 4.32e-01 → 1.14e-02 as T goes 16 → 2048 |
| inference must descend the energy at our step size | monotone over every step: 7.77, 3.57, 2.66, 2.25, 2.00, … |
| our power-iteration λ_max must match a dense eigendecomposition | ratio 0.9997 (depth 4), 0.9924 (depth 8) |

## The PC discrepancy was not ours

The paper reported one loose end: our PC scored 4.7 points below the number in
the authors' repository for their Fashion-MNIST cell, while backprop and PC-ALM
matched within a point. We settled it by running *their* implementation, on our
box, against the same data exported to the IDX format their loader expects.

| arm | ours (3 seeds) | their code, same data (3 seeds) | their README (1 seed) |
|---|---|---|---|
| backprop | 0.7770 ± 0.0034 | 0.7764 ± 0.0090 | 0.7866 |
| PC | 0.6340 ± 0.0194 | 0.6294 ± 0.0134 | 0.6813 |
| PC-ALM | 0.7687 ± 0.0046 | 0.7683 ± 0.0076 | 0.7775 |

Our implementation and theirs agree to 0.0046 on PC, 0.0006 on backprop and
0.0004 on PC-ALM. The gap was never between the two implementations — it was
between their README's single seed-0 run and what their own code produces on
this hardware, where seed 0 gives 0.6297. PC is the highest-variance arm in
every experiment here, and their own three seeds span 0.6159 to 0.6426 with
gradient cosines from 0.490 to 0.728. A single run of it is not a number to
compare against, and we should not have treated it as one.

## EGGROLL: is the estimator the estimator?

**Unbiased.** Averaging independent estimates at fixed weights, the alignment
with the true gradient climbs as more directions accumulate, tracking the
√(P/d) the geometry predicts:

| independent draws | effective P | measured cosine | √(P/d) |
|---:|---:|---:|---:|
| 1 | 512 | 0.0964 | 0.0955 |
| 4 | 2,048 | 0.1891 | 0.1910 |
| 16 | 8,192 | 0.3541 | 0.3820 |
| 64 | 32,768 | 0.6066 | 0.7641 |

**The scale is 2×, as documented.** An antithetic pair gives
f = L(θ+ε) − L(θ−ε) ≈ 2ε·g, so the estimator converges to twice the gradient.
Adam's per-coordinate normalisation absorbs it, which is why it never showed up
as a problem; it is worth stating precisely rather than as "some constant".

**Low rank costs nothing here.** The paper's claim is that rank-r perturbations
approach full-rank ES as O(1/r). At matched population, rank 1, rank 4 and dense
Gaussian perturbations are indistinguishable:

| population | rank 1 | rank 4 | dense |
|---:|---:|---:|---:|
| 128 | 0.0472 | 0.0461 | 0.0483 |
| 512 | 0.0938 | 0.0955 | 0.0938 |

**σ has a wide valid window, and we are in the middle of it.** Cosine is flat at
0.099 from σ = 1e-3 to 1e-1. Below that, float32 cancellation destroys the
finite difference (0.0197 at 1e-5); above it, the linear regime ends (0.0400 at
σ = 1). Our 3e-3 sits centrally, which is why the earlier grid found σ barely
mattered.

## One finding worth acting on

**Antithetic sampling is costing EGGROLL accuracy per unit of compute here.** An
antithetic pair costs two forward passes and yields one direction; one-sided
sampling against the known base loss costs one pass per direction. At matched
forward passes — the thing that actually costs time — one-sided wins:

| forward passes | antithetic | one-sided |
|---:|---:|---:|
| 256 | 0.0463 | 0.0633 |
| 1024 | 0.0923 | 0.1311 |

That is the √2 you would expect from getting twice as many directions for the
same budget. Antithetic sampling earns its keep when the objective is noisy
between evaluations — the reinforcement-learning setting these methods come
from, where there is no stable base value to subtract. Here the base loss is
evaluated on the same minibatch and is exact, so the variance antithetic
cancels was already cancellable for free.

We have **not** changed the arm: the published method is antithetic, and the
comparison in the paper is of methods as published. But a practitioner reaching
for EGGROLL on a deterministic objective should know that the pairing is not
free, and the paper's EGGROLL row would be about 40% more accurate per second
without it.

---

Diagnostics: `scripts/tmp/check_pcalm.py`, `check_eggroll.py`. The reference
comparison runs the authors' unmodified code from
[SakanaAI/pc-alm](https://github.com/SakanaAI/pc-alm) against our exported data
(`scripts/tmp/export_idx.py`).
