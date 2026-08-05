# exp3 — heads versus head dimension

**H4.** The paper reports that more attention heads consistently lower final loss — even 128
heads of dimension 1 — while increasing head dimension gives diminishing returns past a
minimum capacity.

Their sweep holds total width `D = 128` fixed and splits it into more heads, which moves two
things at once: the number of independent attempts at the pattern search, and the capacity of
each attempt. So this runs **two legs**:

- **heads** — `D = 128` fixed, `H ∈ {1 … 64}`, `d_head = 128/H`. The paper's sweep.
- **headdim** — `H = 8` fixed, `d_head ∈ {2 … 64}`. Capacity alone; search width constant.

Config comes from [exp2](sparse_attn_emergence_exp2.html): `S=16, s=4`, the cell where the
default model solves ~50% of seeds, so there is room to move in both directions.

## The metric matters more than the result

`acc2 > 0.95` is too lax here. At `S=16`, a model that learns 15 of the 16 rows of `A` scores
`15/16 + 0.5/16 = 0.969` and counts as solved — and its residual loss is exactly `ln 2/16 =
0.0433`, a value that turns up repeatedly in these rows. So alongside `solve_rate` this page
reports **exact rate**: the fraction of seeds with final `loss2 < 0.01`, meaning every row
learned.

The two metrics tell different stories, and the strict one is the paper's:

![heads vs head dim](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_exp3_heads.svg)

## More heads: monotone

| `H` (`d_head`) | 1 (128) | 2 (64) | 4 (32) | 8 (16) | 16 (8) | 32 (4) | 64 (2) |
|---|---|---|---|---|---|---|---|
| solve rate | 0.00 | 0.00 | 0.12 | 0.44 | **0.81** | 0.62 | 0.75 |
| **exact rate** | 0.00 | 0.00 | 0.00 | 0.12 | 0.38 | 0.50 | **0.56** |
| median `t*` | — | — | 5710 | 5885 | 7244 | 6706 | 6219 |

On the lax metric this looks noisy and saturating. On the strict metric it is **monotone
across every doubling from 8 to 64** — which is the paper's claim, and I would have missed it
reading `solve_rate` alone. A single head with all 128 dimensions cannot do the task at all.

## Bigger heads: saturating

| `d_head` (`H=8`) | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| solve rate | 0.12 | 0.31 | 0.56 | 0.62 | **0.81** | **0.81** |
| **exact rate** | 0.00 | 0.25 | 0.38 | 0.38 | **0.62** | 0.44 |
| median `t*` | 9560 | 6957 | 6122 | 5818 | 5742 | 5239 |

Capacity buys real improvement up to `d_head = 32` and then stops — 64 is no better than 32
and slightly worse on the strict metric. **Diminishing returns past a minimum capacity**, as
claimed.

### Heads are the cheaper axis

The two legs let you price the same gain two ways:

| configuration | attention width | exact rate |
|---|---|---|
| `H=16, d_head=8` | 128 | 0.38 |
| `H=8, d_head=64` | **512** | 0.44 |
| `H=64, d_head=2` | 128 | **0.56** |

Quadrupling attention width at fixed head count roughly matches what you get by doubling the
head count at constant width. Adding heads is the more efficient axis, which is what you
would expect if each head is another attempt at the same search.

## Noise estimate, for free

`H=8, d_head=16` is the *same configuration* in both legs, run twice with different data
order: 7/16 and 10/16 solved. So seed-level noise on a 16-seed solve rate is roughly ±3/16
here, and single-step differences in the tables above should not be read as effects.

## Verdict: H4 replicated

Head count helps monotonically on the strict metric; head dimension saturates. Both halves of
the paper's claim hold.

## Not measured

Both `d_head = 1` configurations — `H=128, d_head=1` (the paper's most striking data point)
and `H=8, d_head=1` — **failed to run**. Three attempts: an OOM instantiating a CUDA graph,
then a 2h20m wedge at 1.7% CPU with command buffers disabled, then the same wedge again with
halved client preallocation. It reproduces only at `d_head = 1`, at both head counts, which
points to an XLA compilation pathology with unit trailing dimensions rather than genuine
memory pressure.

So the specific claim that **128 heads of dimension 1 still solve the task is unverified
here** — not contradicted, unmeasured. The trend across `H = 8 … 64` is consistent with it.
