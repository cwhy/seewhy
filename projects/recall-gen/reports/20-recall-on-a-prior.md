# Recall on a synthetic prior stops at 0.73, and the memory is not why

A network trained on the synthetic prior is shown sixteen items from one world
and then a seventeenth that is a copy of one of them, with half its coordinates
erased. Asked which of the sixteen it is, it answers correctly
**0.732** of the time. Chance is 0.063 and the ceiling is 1.000 — feeding
the true answer in always identifies it.

These are worlds the network trained on, with the answer literally present. It
should be near-perfect and it is not.

## It is not failing to rebuild the item

Split those episodes into quartiles by how far the target sits from its nearest
rival in the same context — a property of the episode, measured without any
network.

![Identification against candidate separation](https://media.tanh.xyz/seewhy/26-09-02/recall-gen_r20_margin_v1.svg)

Identification is **1.000** in the quartile where rivals are far
apart and **0.234** where they are close. That alone would be
unremarkable. What matters is the second curve: in the failing quartile the
network's reconstruction error is **0.028**, its *best* of the four,
while in the quartile it gets right it is **0.090**, its worst.

So the network reconstructs the hard-to-name items better than the easy ones and
still cannot name them. Recall is not unlearned.

The arithmetic explains it. To pick the nearer of two items that sit
0.02 apart, the output has to land within about
0.02 of the right one. The network lands within
0.028. Close, and not close enough — and since an exact
reconstruction would score 1.000, this is a real shortfall in precision rather
than a broken metric.

Why are the candidates so close? Because they come from one world. The prior
draws each world's items from a latent subspace whose dimension is sampled as
low as 1, and sixteen draws from a one-dimensional world lie almost on a line.
The median separation across all episodes is 0.36.

## The memory is not the bottleneck

The obvious suspect was the memory. Recall here works by writing sixteen items
into a fixed-size matrix with the delta rule, `S += e k^T`, and two similar keys
write along nearly the same direction, so the second write partly erases the
first. That predicts exactly the observed pattern: failures concentrated where
items are similar.

It is testable without touching the prior. `d_model = heads x dk`, and `Wq`,
`Wk`, `Wv` and `Wo` are all `d_model x d_model`, so trading heads against dk
leaves the parameter count untouched and changes only how much state there is.

![Parameters against memory size](https://media.tanh.xyz/seewhy/26-09-02/recall-gen_r20_capacity_v1.svg)

| | heads x dk | state floats | parameters | identification |
|---|---|---|---|---|
| d=256, 4x64 | 4 x 64 | 16,384 | 4.06M | 0.614 |
| d=512, 8x64 | 8 x 64 | 32,768 | 14.95M | 0.759 |
| d=512, 4x128 | 4 x 128 | 65,536 | 14.94M | 0.789 |

Doubling the memory at identical parameters buys
**+0.029**. Across three independent draws of
512 episodes the spread on this metric is about 0.015, so that is real and
barely — roughly twice the noise.

Adding parameters buys **+0.146**, five times
as much.

If interference in a bounded memory were what caps recall, doubling that memory
should not be worth a fifth of what parameters are worth. The hypothesis was
wrong, and the plainer reading survives: this is a precision problem, and
precision tracks capacity.

One caveat keeps it from being airtight. Because `d_model = heads x dk`, the
same trade that doubles the state also halves the number of heads — eight
independent memories become four larger ones. A 2 x 256 point would have
separated those, and it asks for 23.65 GiB on a 24 GB card, because the state is
`(batch, heads, dk, dk)` and the scan keeps a carry per token.

## What the network is worth against a ceiling

With the answer absent rather than present, the same episodes have a
model-free ceiling worth stating. A per-episode least-squares fit from the
visible coordinates to the hidden ones, solved in the dual on the sixteen context
items, is the strongest estimator the context allows — and is precisely the
computation linear attention performs.

| | worlds seen | worlds never seen |
|---|---|---|
| network, d=512 8x64 | 0.411 | 0.575 |
| network, d=512 4x128 | 0.415 | 0.550 |
| in-context least squares | **0.324** | **0.303** |
| soft look-up | 0.367 | 0.348 |
| ridge, ignores the context | 0.545 | 0.581 |

On worlds it has never seen the network is
1.9 times worse
than a sixteen-by-sixteen linear solve that has seen no world at all. The gap is
the headroom, and it is large.

## What this changes

Recall on this prior is limited by reconstruction precision, not by the memory
architecture, and precision has tracked capacity at every point measured. The
untested next step is width, not a different memory.

The in-context least-squares baseline outlives the hypothesis it was built to
test. Until now the strongest model-free reference here was a similarity-weighted
average of the context, which is the right ceiling when items are drawn
independently and the wrong one when they share a generative structure. Every
future run on a structured prior should be quoted against it.

## What this does not establish

Where the ceiling is. The largest model tried is the best on every measure and
the trend has not turned over.

That the memory never matters. It was measured at one width, on one prior, with
heads and dk traded against each other rather than varied independently, and the
one point that would have separated them did not fit in memory.

## Sources

`results.jsonl` rows `exp45`, `exp49` and `exp54` — recall training on the
synthetic prior at d_model 256/512 and two head/dk splits — and `baselines_synth_M16_r4_split_class` for the
model-free references, which are computed on single-world episodes to match the
context type these networks were trained on. The margin breakdown is recomputed
from the `exp49` checkpoint by this script. Figures generated by
`scripts/gen_report_20.py`. No training was run for this report.
