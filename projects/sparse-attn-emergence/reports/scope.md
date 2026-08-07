# exp11 & exp12 — what the paper's tasks do and don't cover

The paper explains emergent capabilities like indirect object identification and induction. Its
synthetic tasks are meant to isolate the mechanism behind them. Two questions about that
bridge, each answered by one experiment:

1. **Is the synthetic setting even in-context learning, or just memorisation?** (exp12)
2. **Does anything transfer to a pattern keyed on *content* rather than position?** (exp11)

The answers differ: the first says the paper is on firmer ground than it looks, the second
finds a limit its conclusions do not cover.

## Two axes, not one

| | what varies per sequence | what the head must key on |
|---|---|---|
| linear map | nothing — `A` is fixed for the run | **position** — a fixed set of slots |
| cellular automata | the rule | **position** — a fixed local window |
| IOI / induction / copying | the content | **content** — match a token, copy its successor |

Both synthetic tasks reuse *the same routing every sequence*. An induction head recomputes
where to look from the tokens themselves. That difference is what exp11 tests.

## exp12 — the CA task is genuine in-context learning

The paper draws `N = 256` rules once per run and samples one per example, so a model could in
principle memorise all 256 tables (4 KB) and infer only *which* is active. That would be
in-context *selection*, not learning. One parameter separates the two: sweep the pool size,
including a **fresh table per sequence** drawn from `4⁶⁴`, where memorisation is impossible.

![pool size](https://media.tanh.xyz/seewhy/26-08-06/sparse_attn_emergence_exp12_pool.svg)

| `N` | 1 | 16 | **256** (paper) | 4096 | **fresh** |
|---|---|---|---|---|---|
| memorisable | 16 B | 256 B | 4 KB | 64 KB | **impossible** |
| solves | 8/8 | 8/8 | 8/8 | 7/8 | 8/8 |
| median `t*` | **448** | 1,064 | 4,918 | 4,540 | **5,776** |
| **in-context gain** | **0.000** | 0.235 | **1.193** | 1.155 | **1.187** |

The *in-context gain* is how far per-state loss falls **within** a single sequence — how much
the model infers as evidence arrives, rather than knowing in advance.

**Removing memorisability costs almost nothing.** `fresh` matches `N=256` on solve rate (8/8),
timing (5,776 vs 4,918) and inference (1.187 vs 1.193). The model has learned to identify an
unseen rule from the sequence, not to index a stored table.

`N=1` is what makes this readable rather than assumed: with a single fixed rule the model
solves it **10× faster** and the in-context gain is **exactly zero** — nothing is inferred,
because nothing needs to be. That is what memorisation looks like on this instrument, and the
paper's setting looks nothing like it. `N=16` sits partway (gain 0.235), and by `N=256` the
curve has already saturated at the unmemorisable limit.

**So the CA task is real in-context learning, and this page's earlier scepticism was wrong.**
Pool size controls *how much* inference is required, and the paper's choice is already at the
ceiling.

## exp11 — content-keyed patterns behave differently

Associative recall: a sequence of pairs `a, f(a)` with **a fresh permutation `f` per
sequence**. A repeated key can only be answered by matching its earlier occurrence and copying
the token after it. The position of the answer moves every sequence, so no fixed pattern works.
Two layers for every arm — a one-layer model cannot express the circuit.

At 30,000 steps (10,000 was not enough for any arm to finish):

| arm | recall accuracy | solves | median `t*` | recall loss (plateau 3.466) |
|---|---|---|---|---|
| **KDA** | **1.000** | **14/16** | **15,889** | **0.0005** |
| transformer | 0.905 | 7/16 | 26,929 | 0.295 |
| static mixer | 0.125 | **0/16** | — | 3.135 |

**The mixer never solves it, at any learning rate or budget.** On the linear map the same
architecture *beat* attention across four sparsity levels. Here it cannot do the task at all,
which is what the taxonomy predicts: a static matrix expresses a fixed positional pattern
exactly and a content-matched one not at all.

**And KDA beats the transformer** — perfect recall, twice the seeds, emerging at half the step
count. A delta-rule memory is key→value storage retrieved by key match, which is this task's
native operation.

### The ranking inverts between the two task families

| | linear map (position-keyed) | induction (content-keyed) |
|---|---|---|
| static mixer | **best** — 0.69 / 0.62 / 0.31 past `s=4` | **cannot do it at all** |
| transformer | middle | middle — 7/16, 0.905 |
| KDA | **worst** — fails from `s=4` | **best** — 14/16, 1.000 |

The ordering is exactly reversed, and the axis that explains it is **whether mixing is
conditioned on content**. The mixer's weights are constants: unbeatable when the correct
routing is fixed, useless when it must be computed per sequence. KDA conditions entirely on
content matching: excellent at recall, poor at finding an arbitrary position subset. Attention
does both adequately and neither best.

So "an MLP-Mixer beats a transformer" is not a fact about mixers. It is a fact about
*positional* tasks — the only kind the paper tested.

Its accuracy does creep above chance (0.032 → 0.125), and that has a mundane explanation worth
stating rather than mistaking for partial induction: the answer to a recallable pair is always
a value **already seen** in the sequence, so a model that simply spreads mass over previously
seen values beats 1/32 without matching anything.

**H1's phenomenology does transfer, though.** The transformer's emergence is late (median
`t*` ≈ 26,900 of 30,000 steps), abrupt, and seed-random — 7 of 16 seeds, with the rest still at
chance when training stopped. At 10k steps exactly one seed had made it. That is the paper's
central claim reproducing in the content-keyed regime it never tested.

## What this means for the paper

- **H1 (abrupt, seed-random emergence)** — holds for positional *and* content-keyed circuits.
  The strongest and most portable of the paper's claims.
- **The CA task is genuine ICL** — the memorisation objection does not survive contact with
  the pool sweep.
- **H5 (architecture)** — does **not** transfer, and inverts. The mixer's advantage belongs to
  tasks whose correct routing is fixed and positional; on content-keyed routing it is not
  merely slower but incapable, while KDA goes from worst to best. Any ranking of architectures
  from the paper's synthetic tasks describes positional routing only.
- **Untested by both the paper and us**: whether the sparsity and context-length *laws* from
  [exp2](sparse_attn_emergence_exp2.html) — difficulty tracking `C(S,s)` — have an analogue
  when the pattern is content-keyed. The candidate space there is a set of matching rules
  rather than a set of position subsets, and nothing here shows the two scale alike.
