# Findings

Everything the replication established, in one place. Seven experiments, ~130 training runs of
16 seeds each, all on one RTX 4090 pair. Start with [the paper in plain
terms](sparse_attn_emergence_paper.html) if the setup is unfamiliar; the definitions behind
every number are in [Methods](sparse_attn_emergence_methods.html).

## What the task is

![the linear map task](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_diag_task.svg)

## What emergence looks like

![what emergence looks like](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_diag_emergence.svg)

## Verdict by claim

| | Claim | Verdict | Evidence |
|---|---|---|---|
| **H1** | Emergence is abrupt and seed-random | **holds** (timing), **softened** (abruptness) | [exp1](sparse_attn_emergence_exp1.html), [exp4](sparse_attn_emergence_exp4.html) |
| **H2** | Difficulty is non-monotone in sparsity, grows with context | **holds**, with a degenerate column | [exp2](sparse_attn_emergence_exp2.html) |
| **H3** | The loss jump *is* the pattern being found | **holds**, causally | [exp4](sparse_attn_emergence_exp4.html) |
| **H4** | More heads help; head dim saturates | **holds** on the strict metric | [exp3](sparse_attn_emergence_exp3.html) |
| **H5** | A non-attention mixer learns it faster | **holds past `s=4`**, reversed below it | [exp6/7/8](sparse_attn_emergence_exp67.html) |
| — | Not linear-map-specific | **holds** — same wall on cellular automata | [exp5](sparse_attn_emergence_exp5.html) |
| — | The CA task is in-context learning, not memorisation | **holds** — unmemorisable rules cost nothing | [exp12](sparse_attn_emergence_scope.html) |
| — | Does any of it transfer to *content*-keyed patterns? | **H1 yes, H5 no** — the mixer cannot do induction at all | [exp11](sparse_attn_emergence_scope.html) |

## H1 — timing is random, abruptness is oversold

Two independent 16-seed samples of the same configuration, differing only in initialisation
and data order:

| | median `t*` | range | spread | jump width (`loss2` 0.6→0.05) |
|---|---|---|---|---|
| exp1 | 885 | 469 – 2521 | **5.4×** | median 354 steps ≈ 0.42 × `t*` |
| exp4 | 923 | 500 – 1984 | **4.0×** | — |

The medians agree within 5% and both show a 4–5× spread, so the stochasticity is not one
unlucky draw. **This is the paper's central claim and it is solid.** It also argues against
the paper's own 3 seeds: `[469, 563, 566]` and `[1196, 2187, 2521]` are both plausible 3-seed
draws from our 16 and they support opposite conclusions.

Where we soften it: the drop takes a median 354 steps against a median `t*` of 885, and up to
2173 steps for one seed. Fast relative to when it starts — not the cliff the figures imply.

## H2 — difficulty is quantitative

Learnability tracks **`C(S,s)`**, the number of candidate supports per row — not `s`, not `S`:

| `C(S,s)` | ≲ 500 | 1,800 – 5,000 | ≳ 8,000 |
|---|---|---|---|
| outcome | always solves | 31–50% of seeds | never |

| cell | `C(S,s)` | solves | median `t*` |
|---|---|---|---|
| `S=16, s=3` | 560 | 16/16 | 815 |
| `S=32, s=2` | 496 | 16/16 | 510 |
| `S=16, s=4` | 1,820 | 8/16 | 6,718 |
| `S=32, s=3` | 4,960 | 5/16 | 9,170 |
| `S=16, s=6` | 8,008 | 0/16 | — |
| `S=32, s=4` | 35,960 | 0/16 | — |

Cells from different context lengths land together when matched on `C`. So "longer context
makes sparse patterns harder to find" resolves into something sharper: **longer context
inflates the number of wrong patterns**, and difficulty follows that count. `C(32,16) ≈ 6×10⁸`
against `C(16,8) = 12,870` is why the `S=32` unlearnable band is so much wider.

`C` is not the whole story: `C(16,4)` and `C(16,12)` are both 1,820, yet the sparse one solves
half the time and the dense one never does. A smaller second cost grows with `s` itself.

### And one of the task's columns is degenerate

![the degenerate dense column](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_diag_artifact.svg)

At `s = S` the "recovery" (16/16 in ~30 steps) is **copying**, verified per position: 0.488
accuracy on the first output token, 1.000 on all others, final loss exactly `ln 2/S`. The
paper shares this construction, so the caveat applies to its `s=S` results too.

## Would more examples help? No — more context makes it harder

A natural question the paper only half answers. It sweeps trajectory length `T` on the
cellular automata task (longer trajectories → longer plateaus, non-monotonically) but fixes
`T = 2` for the linear map, so on that task the axis is untested. exp10 tests it: same matrix
`A`, but `T ∈ {2, 4, 8}` states per sequence, i.e. **1, 3 or 7 worked examples of the same map
inside every sequence**.

![trajectory length](https://media.tanh.xyz/seewhy/26-08-06/sparse_attn_emergence_exp10_traj.svg)

| | `T=2` | `T=4` | `T=8` |
|---|---|---|---|
| supervised targets per step | 4,096 | 6,144 | **7,168** |
| `s=3`, every row exact | **1.00** | 0.56 | **0.12** |
| `s=3`, median `t*` | **820** | 3,617 | **6,525** |
| `s=4`, solved | **7/16** | **0/16** | **0/16** |

**More examples per sequence makes it strictly worse** — 8× slower at `s=3`, and it turns a
half-solvable cell (`s=4`) into an unsolvable one. Note the confound points the other way:
tokens per step are held fixed, so larger `T` actually delivers *more* supervised targets per
step, because a smaller fraction of each sequence is the unpredictable first state.

**Why, most likely: with absolute position embeddings, more examples means more patterns, not
more evidence for one pattern.** Predicting `x₁[i]` requires attending to row `i`'s support
inside `x₀`; predicting `x₂[i]` requires the same support one state later, at *different
absolute positions*. Nothing in the architecture ties those together, so `T=8` asks the model
to find seven separate absolute patterns, each with the same `C(S,s)` search space, instead of
one. That predicts relative position encoding (RoPE) should flip the result — untested here,
and the obvious next experiment.

**So: the failure is not sample-limited.** Training samples are unlimited already (fresh
sequences every step, ~2.5M per run), the paper's own attention-biasing experiment converges
"almost instantly" once the pattern is supplied, and adding worked examples *inside* the
context makes learning slower. What helps is being given the pattern; what does not help is
more data. In-context examples do help *inference* — exp5's per-state loss falls 1.298 → 0.130
within a CA sequence as the active rule becomes identifiable — but that is using a circuit,
not forming one.

## H3 — the mechanism, causally

| condition | second-half loss |
|---|---|
| intact | **0.0000** |
| best-aligned head removed | **4.2264** |
| worst-aligned head removed | 0.0803 |
| `ln 2` — knowing nothing | 0.6931 |

Removing the head whose attention matches `A` costs five orders of magnitude; removing the
least-aligned head costs almost nothing. Note that 4.23 is **six times** the plateau: ablation
does not restore ignorance, it corrupts a computation that depended on that head, leaving the
model confidently wrong (~1.4% on the true token).

Honest limit: alignment saturates at 0.84, not 1.0, while loss reaches 7×10⁻⁶. Soft attention
need not match the support exactly to support an exact computation.

## H4 — heads are the efficient axis

Strict metric (`loss2 < 0.01`, every row learned) at `S=16, s=4`:

| more heads, `D=128` fixed | H=8 | H=16 | H=32 | H=64 |
|---|---|---|---|---|
| exact rate | 0.12 | 0.38 | 0.50 | **0.56** |

| bigger heads, `H=8` fixed | dh=4 | dh=8 | dh=16 | dh=32 | dh=64 |
|---|---|---|---|---|---|
| exact rate | 0.25 | 0.38 | 0.38 | **0.62** | 0.44 |

Monotone in head count; saturating in head dimension past ~32. Quadrupling attention width at
fixed head count buys about what doubling the head count buys at constant width — heads are
cheaper. The paper's `H=128, d_head=1` point is **unmeasured** here (XLA compile pathology at
`d_head=1`).

## H5 — holds in its regime, with a crossover at `s = 4`

Solve rate, both arms, best learning rate per cell (16 seeds each):

| `s` | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| transformer | **1.00** @732 | 0.50 | 0.06 | 0.00 | 0.00 | 0.00 |
| causal mixer | **1.00** @3693 | **0.69** | **0.62** | **0.31** | **0.31** | **0.19** |
| KDA linear attention | **1.00** @1817 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Attention is better while the search is easy — both solve `s=3`, the transformer 5× faster —
and mixing takes over from `s=4`, becoming the only architecture that solves anything from
`s=5` on. **The paper's claim holds in the regime it was made about**; the unqualified reading
("learns the linear map faster") does not.

**A third architecture localises the cause.** KDA linear attention computes its mixing from
content like attention but without any softmax, and it tracks *attention*, not the mixer —
solving `s=3` (in 1817 steps) and then failing from `s=4`, one cell earlier than the
transformer. So removing the softmax is not what buys the mixer its range. What does is that
the mixer's position weights are free parameters optimised directly, rather than computed from
content. The paper's "sparse attention patterns are hard to learn" is more precisely
**content-dependent position selection is hard to learn**.

Two qualifications. Read strictly (every row learned, not 15 of 16), both architectures
collapse past `s=4` and the mixer's advantage lives mostly in partial solutions. And an
**unmasked** mixer solves everything in ~390 steps at *chance-level* alignment (0.31 against
chance 0.28) — it reads the token it is predicting, so any mixer comparison without causal
masking is void. The paper does not state whether its mixer is masked.

## Beyond the paper

Five things this replication adds rather than merely confirms:

1. **A spread estimate.** 16 seeds instead of 3 turns "stochastic" into 4–5×, measured twice.
2. **A quantitative difficulty law.** `C(S,s)` collapses the `(S, s)` surface onto one axis
   with a threshold that holds across context lengths.
3. **A degenerate column in the shared task design**, caught by an exact arithmetic match
   (`ln 2/S`) and confirmed per position.
4. **Metric sensitivity that changes conclusions.** `acc2 > 0.95` admits pure copying at
   `s=S`, and admits 15-of-16 rows anywhere; H4 looks noisy under it and monotone under the
   strict criterion.
5. **Masking sensitivity in the architecture comparison** — an unstated implementation choice
   that flips the headline result.
6. **A sharper statement of the mechanism.** Three architectures separate softmax from
   content-dependence, and the difficulty follows content-dependence: KDA has no softmax and
   still fails like attention, while the mixer's directly-optimised weights are what extend its
   range.

## Limits

One layer and `D=128` for the linear map; `S ≤ 32`; 10,000 steps; the real-LM half (Pythia,
IOI) out of scope entirely; one alternative architecture rather than seven; three learning
rates; 8 seeds on the CA task; `t*` carries ~8% run-to-run noise because GPU reductions are
not bit-deterministic.

## How it was built

![how the experiments fit together](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_diag_map.svg)

Every seed of a configuration trains **simultaneously** under one `jax.vmap` over a leading
parameter axis — 16 seeds cost about what one costs, which is what makes seed-distribution
claims affordable. A full 16-seed, 10,000-step run of the base config takes **167 seconds**.

Code, one file per experiment, and a `results.jsonl` row per configuration carrying
hyperparameters and per-seed curves: `projects/sparse-attn-emergence/` in
[seewhy](https://github.com/cwhy/seewhy). These pages are generated from committed markdown.

The errors made along the way, and how each surfaced, are on
[Mistakes](sparse_attn_emergence_mistakes.html) — including one wrong published claim that was
caught only because a reader pushed back.
