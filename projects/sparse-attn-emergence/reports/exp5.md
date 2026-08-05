# exp5 — cellular automata, in context

Everything so far used the linear map, where one secret matrix `A` is fixed for the whole run
and can be memorised into the weights. This is the paper's second synthetic task, and it is
structurally different in a way worth stating plainly.

A pool of **`N = 256` rules** is drawn per run, and — per the paper's appendix — *"one rule is
sampled per training example"*. Each sequence therefore uses a **different** rule, so the map
cannot be stored in the weights. The model has to identify the active rule **from the sequence
itself** and then apply it. exp1–exp4 test emergence of an in-weights circuit; exp5 tests
emergence of an **in-context** one.

**Setup.** `S = 16` cells (the paper does not state this — ours), `C = 4` colours, window
`W = 3`, `T = 16` states flattened to a 256-token sequence, 4 layers, `D = 128`, `H = 8`,
10,000 steps, **8 seeds** (4 layers × 256 tokens is ~8× the cost of the linear map runs).
Composing the rule `k` times per transition widens the required attention span to `2k+1`.

Plateau is `ln 4 ≈ 1.386`.

## Metrics have to change

The first state is uniform noise, and the early states are *genuinely ambiguous* — with 256
possible rules, no model can predict state 2 well from state 1 alone. Reporting one loss over
the whole sequence would mix "impossible" with "not yet learned". So the headline is loss on
the **final state**, where all in-context evidence is available, and the per-state profile is
reported as the in-context learning curve.

![cellular automata](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_exp5_ca.svg)

## Result

| `k` | required span | solves /8 | final-state loss | per-state loss, first → last |
|---|---|---|---|---|
| 1 | 3 | **4** | **0.110** | 1.298 → 0.130 |
| 2 | 5 | 0 | 0.837 | 1.375 → 0.831 |
| 3 | 7 | 0 | 1.258 | 1.378 → 1.197 |

**The in-context circuit does emerge, and difficulty tracks the required span.** At `k=1` half
the seeds solve it, and the per-state curve is the signature of in-context learning: near the
`ln 4` plateau on the first predicted state, falling to 0.130 by the last as evidence about
which rule is active accumulates. At `k=2` the model gets partway (0.837, clearly below
plateau) but no seed finishes; at `k=3` it barely moves.

This matters for the paper's argument in a specific way: the sparsity/context effect is **not
an artifact of the linear map's algebra**. A different task family, a different vocabulary, a
rule that changes every sequence — and widening the required attention pattern from 3 to 5
positions still takes the model from half-solved to never.

## Caveats

- **8 seeds, not 16** — the cost per run is ~8× the linear map's. A 4/8 solve rate carries a
  wide interval; treat it as "roughly half", not 50.0%.
- **`S = 16` is ours.** The paper does not state the cell count, and it plausibly affects
  difficulty the way `S` does for the linear map.
- **Boundaries wrap.** Also unstated in the paper; we use circular neighbourhoods.
- **Only `k=1` is learnable at this budget**, so this sweep shows a monotone wall rather than
  the richer shape [exp2](sparse_attn_emergence_exp2.html) found. A `k=1` run with a larger
  `S`, or a longer budget at `k=2`, would say more.
- No attention-pattern IoU is reported here: with 4 layers and a rule that changes per
  sequence, the "correct" pattern is a composition across layers rather than a single row
  support, so the exp4-style metric does not transfer directly.
