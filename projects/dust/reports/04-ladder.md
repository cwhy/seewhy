# 64 DUST draws beat 1,024 EGGROLL passes: 0.8 nats, at 15× less compute

At 1M training tokens, DUST with **64 draws** reaches a test loss of **6.315** at 2.2 s per update.
EGGROLL with **1,024 forward passes** reaches **7.133** at 32.4 s per update. For reference, backprop
scores 5.937, and a model that knows only token frequencies scores 6.994. EGGROLL, at 1,024 passes, is above that
floor. DUST at 256 draws: 6.074, DUST at 1,024 draws: 5.939.
All of this is in our own implementation, with EGGROLL set up as in `local-learning`. Three seeds per cell.

## The ladder

![Test loss against population, 1M tokens](https://media.tanh.xyz/seewhy/26-10-08/ladder_1M_c.svg)

Test loss at the best-validation checkpoint. Population is draws for DUST and forward passes of the batch for EGGROLL. Dashed
lines: backprop (ours and the paper's) and the unigram floor.

| population | DUST (ours) | DUST (paper) | EGGROLL (ours) | EGGROLL (paper) | EGGROLL − DUST (ours) | (paper) |
|---:|---|---:|---|---:|---:|---:|
| 64 | 6.315 ± 0.043 | 6.290 | 7.764 ± 0.011 | 7.428 | 1.45 | 1.14 |
| 256 | 6.074 ± 0.020 | 6.067 | 7.409 ± 0.012 | 7.200 | 1.33 | 1.13 |
| 1,024 | 5.939 ± 0.010 | 5.934 | 7.133 ± 0.004 | 7.046 | 1.19 | 1.11 |

Every DUST cell is within 0.025 of the paper's. Our EGGROLL is weaker than the paper's by
0.09 to
0.34 nats (report 2), so every gap in the last two columns is wider here than in the paper.

## DUST against backprop

| population | DUST − backprop | backprop sd | DUST sd |
|---:|---:|---:|---:|
| 64 | +0.378 | 0.027 | 0.043 |
| 256 | +0.137 | 0.027 | 0.020 |
| 1,024 | +0.002 | 0.027 | 0.010 |

DUST closes on backprop as population grows, and at 1,024 draws it is level with it. The paper has it slightly *below*.

## What it costs

![Test loss against measured seconds per update](https://media.tanh.xyz/seewhy/26-10-08/cost_1M_c.svg)

| population | DUST s/update | EGGROLL s/update | EGGROLL ÷ DUST |
|---:|---:|---:|---:|
| 64 | 2.2 | 2.1 | 0.95 |
| 256 | 7.6 | 8.2 | 1.08 |
| 1,024 | 29.5 | 32.4 | 1.10 |
| backprop | 0.13 | | |

A DUST draw and an EGGROLL forward pass cost about the same here, so at equal population the comparison is also equal-compute.
The cost is wall-clock on one RTX 4090, from code written for clarity (see the paper's limitations).

![Validation loss over training](https://media.tanh.xyz/seewhy/26-10-08/curves_1M_c.svg)

Validation loss over the 61 updates, mean over seeds.

## How much population would EGGROLL need?

The paper extrapolates EGGROLL's ladder to the population at which it would reach DUST's loss at 64 draws (6.315 here), and reports it as a multiple of 64. Same fits, on our ladder (64, 256, 1,024 passes). The paper's Table 2 numbers for 1M tokens are in brackets.

| fit | multiple of 64 (ours) | paper |
|---|---:|---:|
| straight line in log population, top three | 547× | 8,200× |
| straight line, top two | 973× | 6,000× |

Our multiples are 6× to 15× smaller than the paper's. The reason is the slope. Our EGGROLL gains 0.158 nats per doubling of population (64 → 1,024); the paper's gains 0.095 over the same range. We start weaker and improve faster, so the line reaches DUST's loss sooner. Both are far beyond anything measured here: 34,982 to 62,251 forward passes on our ladder. A line through three points is an interpolation that has been extended. Read these as an order of magnitude, not a number.

## What this changes

- The ordering reproduces on a baseline we implemented ourselves: DUST at 64 draws beats EGGROLL at 1,024 passes, by 0.82 nats.
- The *size* of the efficiency advantage does not transfer. Extrapolated, it is hundreds of times the population, not the
  paper's thousands, and it depends on the slope of EGGROLL's ladder: three or four points, one baseline implementation, and
  an EGGROLL that is weaker than the paper's but steeper.
- Not tested: populations above 1,024, 10M tokens and above at more than one seed (report 5), and whether DUST
  keeps passing backprop as populations grow, which the paper reports and we have not reached.

Code: `experiments3.py`, `experiments9.py`, `experiments10.py`.
