# Our EGGROLL baseline trails the paper's, by 0.34 nats at 64 passes and 0.09 at 1,024

We tuned EGGROLL on the paper's own task and landed at **7.775** with 64 forward passes on the
tuning seed (paper: 7.428) and **7.399** with 256 (paper: 7.200). The final three-seed
ladder, at those settings, reads 7.764, 7.409 and 7.133 at 64, 256 and 1,024 passes
(paper: 7.428, 7.200, 7.046). So our baseline is *weaker* than
theirs by 0.34, 0.21 and 0.09 nats, a gap that closes as population grows. That matters because any weakness here would
flatter DUST. DUST at 256 draws scores 6.074, so the DUST–EGGROLL gap at that population is 1.33 nats here
against 1.13 in the paper. The ladder's gap is 6 times the baseline's, so a baseline this much weaker cannot
explain it. We could not close the 0.20 at 256 passes either.

## EGGROLL, in one picture

![EGGROLL on one weight matrix](https://media.tanh.xyz/seewhy/26-10-08/dust_eggroll_update.svg)

EGGROLL (Sarkar et al.) is evolution strategies made cheap. For every weight matrix, a population member adds a
*rank-one* perturbation `σ·a·bᵀ`. It costs `m + n` random numbers, not `m × n`, and the forward pass applies it as a
side branch, so a member is not a copy of the network. A member is scored as an *antithetic pair*, two forward
passes with `+` and `−` the same perturbation, whose loss difference is one number. The gradient estimate is the
sum of the perturbations weighted by those numbers, and goes to Adam. This is exactly the estimator in
`local-learning/lib/es.py`, applied to the transformer. Population is counted in forward passes, two per pair.

## What we tuned

Seed 42, validation loss picks the winner, 1M tokens. Adam rate and noise scale σ on a grid; then the paper's Adam
constants (β₁ = 0.5, β₂ = 0.95, from its Table 7); then σ extended downward, because the best σ sat on the grid's edge.

![EGGROLL tuning grid](https://media.tanh.xyz/seewhy/26-10-08/eggroll_tuning_c.svg)

| passes | σ | Adam (β₁, β₂) | variant | lr | validation | test |
|---:|---:|---|---|---:|---:|---:|
| 64 | 0.0003 | paper (0.5, 0.95) | raw, bf16 | 0.003 | 7.796 | 7.790 |
| 64 | 0.001 | paper (0.5, 0.95) | raw, bf16 | 0.003 | 7.789 | 7.783 |
| 64 | 0.003 | paper (0.5, 0.95) | f32 | 0.003 | 7.779 | 7.773 |
| 64 | 0.003 | paper (0.5, 0.95) | rank-shaped | 0.003 | 7.761 | 7.754 |
| 64 | 0.003 | paper (0.5, 0.95) | rank-shaped | 0.01 | 8.200 | 8.196 |
| 64 | 0.003 | paper (0.5, 0.95) | raw, bf16 | 0.001 | 8.067 | 8.063 |
| 64 | 0.003 | paper (0.5, 0.95) | raw, bf16 | 0.003 | 7.782 | 7.775 **←** |
| 64 | 0.003 | default (0.9, 0.999) | raw, bf16 | 0.0003 | 8.245 | 8.244 |
| 64 | 0.003 | default (0.9, 0.999) | raw, bf16 | 0.001 | 8.067 | 8.064 |
| 64 | 0.003 | default (0.9, 0.999) | raw, bf16 | 0.003 | 7.823 | 7.815 |
| 64 | 0.003 | default (0.9, 0.999) | raw, bf16 | 0.01 | 8.319 | 8.319 |
| 64 | 0.01 | default (0.9, 0.999) | raw, bf16 | 0.0003 | 8.261 | 8.260 |
| 64 | 0.01 | default (0.9, 0.999) | raw, bf16 | 0.001 | 8.126 | 8.123 |
| 64 | 0.01 | default (0.9, 0.999) | raw, bf16 | 0.003 | 7.932 | 7.925 |
| 64 | 0.01 | default (0.9, 0.999) | raw, bf16 | 0.01 | 8.319 | 8.319 |
| 256 | 0.001 | paper (0.5, 0.95) | raw, bf16 | 0.003 | 7.410 | 7.401 |
| 256 | 0.003 | paper (0.5, 0.95) | raw, bf16 | 0.001 | 7.760 | 7.752 |
| 256 | 0.003 | paper (0.5, 0.95) | raw, bf16 | 0.003 | 7.407 | 7.399 **←** |
| 256 | 0.003 | default (0.9, 0.999) | raw, bf16 | 0.001 | 7.766 | 7.759 |
| 256 | 0.003 | default (0.9, 0.999) | raw, bf16 | 0.003 | 7.467 | 7.456 |
| 256 | 0.003 | default (0.9, 0.999) | raw, bf16 | 0.01 | 7.804 | 7.794 |

## What moved the loss, and what did not

- **The Adam rate** is the one that matters: at 64 passes and σ = 3e-3, the cells run
  8.244, 8.064, 7.815, 8.319 for rates 0.0003, 0.001, 0.003, 0.01. At the largest rate the loss stays at its starting value.
- **σ** hardly does. At the paper's Adam constants and lr 3e-3, σ of 0.0003, 0.001, 0.003 gives
  7.790, 7.783, 7.775: a spread of 0.014 across a decade.
- **The paper's Adam constants** help a little: 7.815 → 7.775 at 64 passes, and
  7.456 → 7.399 at 256. The paper reports that EGGROLL gains under 0.01 from switching to Adam at all.

**Two more knobs, at 64 passes.** The paper lists fitness shaping among EGGROLL's tuned constants, and bfloat16 rounding could be eating the loss differences. Against the best cell (7.775): rank-shaped, lr 0.003: 7.754; f32, lr 0.003: 7.773; rank-shaped, lr 0.01: 8.196. The best of them is 0.021 nats below the best raw cell, one seed and well inside seed-to-seed spread. The ladder keeps the raw loss differences of `local-learning`.

## What this changes

- The ladder in the next report uses these settings: the best-validation cell at each population, and the 256-pass
  winner at 1,024 passes, which was too expensive to tune.
- Our EGGROLL is a faithful instance of the method, not a replica of the paper's. The remaining 0.09 to
  0.34 nats are not explained, and could be any detail of theirs we cannot see: their code is not released.
  One thing is visible: ours improves faster with population, 0.158 nats per doubling against the paper's 0.095.
- The conclusion this puts in doubt is not the ordering, which the gap in the ladder dwarfs, but the *slope* of
  EGGROLL's ladder, which sets the extrapolation of how much population it would need to match DUST.

Code: `lib/eggroll.py`, `experiments5.py`, `experiments8.py`, `experiments14.py`, `experiments16.py`.
