# Local learning vs backprop — replicating a "quick MNIST test"

Replicates [@mayfer's table](https://x.com/mayfer/status/2099541904241385615)
(2026-09-14): backprop, PC-ALM and Forward-Forward on MNIST at ~56K parameters,
one epoch. Five seeds per arm, one RTX 4090, JAX.

**The parameter counts identify the architectures, and they differ.** 56,128 is
exactly the PC-ALM paper's headline cell — a residual MLP of width 32, depth 32.
56,232 is exactly a plain 784-66-66 MLP, which is what Forward-Forward needs
(goodness replaces the output layer). Two rows are 32 layers deep; the third is
two layers wide.

## 1. It replicates

| arm | params | ours (5 seeds) | posted | train s | posted | 10k infer s | posted |
|---|---:|---|---:|---:|---:|---:|---:|
| backprop 32x32 res | 56128 | 0.8704 ± 0.0100 | 0.8739 | 0.199 | 0.324 | 0.00025 | 0.0229 |
| PC-ALM 32x32 res | 56128 | 0.8533 ± 0.0085 | 0.8628 | 8.123 | 13.485 | 0.00025 | 0.0241 |
| Forward-Forward 2x66 | 56232 | 0.8869 ± 0.0085 | 0.8883 | 0.042 | 0.549 | 0.00129 | 0.0781 |
| PC (control) | 56128 | 0.6125 ± 0.0403 | — | 8.133 | — | 0.00023 | — |
| backprop 2x66 (control) | 56902 | 0.9116 ± 0.0058 | — | 0.037 | — | 0.00016 | — |
| Forward-Forward 31x32 (control) | 56800 | 0.5751 ± 0.0744 | — | 0.324 | — | 0.00177 | — |

All three posted accuracies fall inside our seed spread. PC-ALM/backprop
training ratio 40.8x against the posted 41.6x. Chance is 0.10.

![accuracy against the posted numbers](https://media.tanh.xyz/seewhy/26-09-15/accuracy_vs_tweet.svg)

## 2. Forward-Forward's win is the architecture

Backprop on FF's own 784-66-66 stack: **0.9116 ± 0.0058**, i.e. 2.5 points above FF
itself. FF moved onto the deep-narrow stack: **0.5751 ± 0.0744**.

![FF readouts](https://media.tanh.xyz/seewhy/26-09-15/ff_readouts.svg)

The deep collapse is the readout, not the features: a linear probe on the
31-layer net's own activations gets 0.8919 where the goodness sweep gets
0.5751. Not a learning-rate artifact — the best of four rates on the deep
stack gives 0.5808.

## 3. One shared learning rate is doing a lot of the work

![learning-rate grid](https://media.tanh.xyz/seewhy/26-09-15/learning_rate_grid.svg)

Best rate per arm differs by 30x. 1e-3 — the value the PC-ALM rule gives at this
cell, and the one the posted comparison used throughout — is best for exactly
one arm: Forward-Forward's. Re-run at each arm's own best rate, 5 seeds:

| arm | rate | tuned | vs default |
|---|---:|---|---:|
| backprop 32x32 res | 0.03 | 0.9374 ± 0.0067 | +0.0670 |
| PC-ALM 32x32 res | 0.01 | 0.9277 ± 0.0023 | +0.0745 |
| backprop 2x66 | 0.003 | 0.9171 ± 0.0025 | +0.0056 |
| Forward-Forward 2x66 | 0.001 | 0.8869 ± 0.0085 | +0.0000 |
| PC 32x32 res | 0.003 | 0.6778 ± 0.0258 | +0.0653 |
| Forward-Forward 31x32 | 0.003 | 0.6106 ± 0.0469 | +0.0354 |

The posted order was FF > backprop > PC-ALM. Tuned, it is backprop > PC-ALM > FF
— reversed among exactly the three arms compared.

![tuned vs default](https://media.tanh.xyz/seewhy/26-09-15/tuned_vs_default.svg)

## 4. At equal wall-clock

![accuracy vs training seconds](https://media.tanh.xyz/seewhy/26-09-15/accuracy_vs_walltime.svg)

Backprop passes PC-ALM's one-epoch accuracy inside its second epoch (0.908 at
0.40 s vs 0.855 at 7.68 s) and reaches 0.946 in the time PC-ALM needs for one
epoch. Forward-Forward is the only arm that gets *worse* with training: 0.901 at
two epochs, 0.860 at ten, and its linear probe falls further (0.9102 → 0.7956).

## 5. The source paper's claim holds

PC and PC-ALM differ here by one scalar (alpha = 0 or 1), same T, same eta_h,
same seeds.

![accuracy vs depth](https://media.tanh.xyz/seewhy/26-09-15/accuracy_vs_depth.svg)

| layers | backprop | PC | PC-ALM | cos(PC) | cos(PC-ALM) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.9054 | 0.9055 | 0.9049 | 0.999 | 0.999 |
| 4 | 0.9012 | 0.8963 | 0.8943 | 0.766 | 0.965 |
| 8 | 0.8900 | 0.8856 | 0.8839 | 0.702 | 0.928 |
| 16 | 0.8830 | 0.8043 | 0.8628 | 0.724 | 0.903 |
| 32 | 0.8711 | 0.6137 | 0.8554 | 0.806 | 0.889 |
| 64 | 0.8518 | 0.5547 | 0.8207 | 0.775 | 0.888 |

## 6. Is our reimplementation faithful?

Written from the paper's equations, then checked against the authors' published
reference run on their Fashion-MNIST cell (their numbers are single runs):

| | ours (3 seeds) | theirs | our cosine | their cosine |
|---|---|---:|---:|---:|
| backprop | 0.7770 ± 0.0034 | 0.7866 | 1.000 | 1.000 |
| PC | 0.6340 ± 0.0194 | 0.6813 | 0.786 | 0.604 |
| PC-ALM | 0.7687 ± 0.0046 | 0.7775 | 0.901 | 0.909 |

Backprop and PC-ALM land within a point; PC is 4.7 low and is the one arm that
does not match. Swapping our measured activity step for their tabulated one
changes nothing (0.6330 vs 0.6340).

---

Full write-up: the paper. Rows: `results.jsonl`, 236 cells.
