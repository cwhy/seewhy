# exp2 — the sparsity × context-length difficulty window

**H2.** The paper reports that `S=8` is solvable at every sparsity, while medium sparsity
becomes *unlearnable* at `S=16` and `S=32`. This sweeps all 24 cells: `S ∈ {8,16,32}` ×
`s`, 16 seeds each, 10,000 steps, one `results.jsonl` row per cell.

**Each seed draws its own `A`** (exp1 fixed it). A cell should describe the `(S, s)` regime,
not one lucky matrix. See [Methods](sparse_attn_emergence_methods.html).

## The difficulty surface

![sweep panels](https://media.tanh.xyz/seewhy/26-08-04/sparse_attn_emergence_exp2_sweep.svg)

Solve rate (fraction of 16 seeds reaching `acc2 > 0.95` within budget):

| `s` → | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| **S=8** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | | | | |
| **S=16** | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | 0.00 | 0.00 | 1.00\* | | |
| **S=32** | 1.00 | 1.00 | 0.31 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00\* |

\* degenerate — see below. **This is the paper's claim: `S=8` learns every sparsity, and an
unlearnable band opens at longer context.** The band widens sharply: it starts at `s=6` for
`S=16` and at `s=4` for `S=32`, where it then covers everything up to `s=24`.

## What actually sets the difficulty

Not `s`, and not `S` — the **number of candidate supports per row**, `C(S,s)`, which is the
size of the space the model has to search:

![search space](https://media.tanh.xyz/seewhy/26-08-04/sparse_attn_emergence_exp2_search_space.svg)

| `C(S,s)` | ≲ 500 | ~1,800 – 5,000 | ≳ 8,000 |
|---|---|---|---|
| outcome | always solves | 31–50% of seeds | never |

Cells from different context lengths land together when matched on `C`, which is the
interesting part:

| cell | `C(S,s)` | solve | median `t*` |
|---|---|---|---|
| S=16, s=3 | 560 | 1.00 | 815 |
| S=32, s=2 | 496 | 1.00 | 510 |
| S=16, s=4 | 1,820 | 0.50 | 6,718 |
| S=32, s=3 | 4,960 | 0.31 | 9,170 |
| S=16, s=6 | 8,008 | 0.00 | — |
| S=32, s=4 | 35,960 | 0.00 | — |

So "context length makes sparse patterns harder to find" is more precisely: **longer context
inflates the number of wrong patterns**, and difficulty follows that count. `C(32,16) ≈
6×10⁸` against `C(16,8) = 12,870` is why the `S=32` band is so much wider.

### But `C` is not the whole story

`C(16,4)` and `C(16,12)` are **both 1,820**. The sparse one solves half the time; the dense
one never does. Equal search space, opposite outcomes — so there is a second, smaller cost
that grows with `s` itself, plausibly the arity of the XOR the MLP must compute once the
right positions are attended to. Difficulty is mostly search-space size, with a dense-side
penalty on top; the inverted U is **not** symmetric.

## The `s = S` column is an artifact, not a recovery

Both `s=S` cells look like triumphant recoveries — 16/16 in 34 steps at `S=16`, 32 steps at
`S=32`, faster than anything else in their rows. They are neither recoveries nor solves.

At `s = S` every row of `A` is all-ones, so all second-half tokens equal `parity(x₀)`: **one
value, repeated `S` times**. A model that computes nothing can emit position `S` at chance
and copy it for the remaining `S−1`, scoring `1 − 0.5/S` and leaving exactly `ln 2 / S` loss.

The observed final losses were **0.0433** at `S=16` and **0.0217** at `S=32`. `ln 2/16 =
0.04332`; `ln 2/32 = 0.02166`. Verified directly with
`scripts/check_dense_shortcut.py` — per-position accuracy at `S=16`:

| position | 16 | 17 | 18 | … | 31 |
|---|---|---|---|---|---|
| accuracy | **0.488** | 1.000 | 1.000 | … | 1.000 |

Chance at the first second-half token, perfect at every later one. It is copying.

This also resolves an anomaly: `S=8, s=8` was the *slowest* cell in its row (1,247 steps)
despite being another `C=1` case. Copying at `S=8` yields only 0.9375 — below the 0.95
threshold — so that cell had to genuinely learn 8-bit parity, and it did, reaching
`loss2 = 0.0000` rather than stalling at `ln 2/S`.

**Consequence for the claim.** The dense end of the inverted U carries no evidence about
finding attention patterns, and `acc2 > 0.95` is too weak a bar there for any `S ≥ 16`. The
paper uses the same construction, so the same caveat applies to its `s = S` results.

## Verdict: H2 partially replicated

**Replicated:** `S=8` learns every sparsity; an unlearnable band exists at `S=16` and
`S=32`; the band widens with context length. Adding to the paper: the boundary is
quantitative and lives in `C(S,s)`, roughly the same threshold at both context lengths.

**Not replicated as framed:** "both extremes are easy" holds only trivially. The dense
extreme is a degenerate task where copying passes the threshold without the map being
learned, and the one dense cell where copying *cannot* pass (`S=8, s=8`) was the slowest in
its row rather than the fastest.

## Reproducibility footnote

Ten cells were re-run after a sync accident destroyed their rows. Same seeds, same
matrices, same code — and `S=16, s=3` came back 815 steps against 833, `s=4` 6,718 against
6,207 (solve rate identical at 0.50). GPU reductions are not bit-deterministic, so
individual `t*` values carry roughly ±8% run-to-run noise. Nothing here rests on
differences that small: the effects are factors of 5 to 100, or the difference between
16/16 and 0/16.
