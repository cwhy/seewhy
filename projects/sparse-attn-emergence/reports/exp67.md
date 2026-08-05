# exp6 & exp7 — mixer versus transformer, and what masking is worth

**H5.** The paper's sharpest claim: an MLP-Mixer — which learns position-mixing weights
directly instead of computing them through a softmax competition — learns the linear map
*faster* than a transformer. If sparse patterns are hard to **find**, an architecture that
doesn't have to search shouldn't suffer the plateau.

This page covers a failed first attempt and the experiment that replaced it, because the
failure is the more useful half.

## exp6 asked the wrong question

exp6 auto-selected its comparison cell as *the cell where the transformer does worst* — and
found the mixer no better: 0/16 for both at `S=32, s=4`, plus the mixer losing badly at `s=3`
and `s=4`. I reported that as H5 not replicating.

That was wrong, for a reason that only appeared on re-reading the paper: **their mixer claim is
at `S=16, s=7`.** In our [difficulty surface](sparse_attn_emergence_exp2.html) `s=7` sits
inside the unlearnable band, and the claim is precisely that the mixer wins *where attention
fails*. exp6 compared at cells where the transformer is comfortable, so it never tested the
claim. Two further differences also mattered: the paper publishes **no hyperparameters** for
these runs, and it never says whether its mixer is **causally masked**.

## exp7: the paper's config, three arms, swept LR

| arm | position mixing |
|---|---|
| transformer | softmax attention, 8 heads |
| causal mixer | one static learned matrix, **masked lower-triangular** |
| unmasked mixer | the same matrix, **no mask** — unsound for next-token prediction, included as a diagnostic |

The unmasked arm exists because an unmasked mixing matrix **leaks the target**: position
`S+i−1` mixes in position `S+i`, the very token being predicted. The paper describes its mixer
only as "a static learned matrix that mixes information across sequence positions", so this arm
measures what that unstated choice is worth.

Learning rates `{3e-4, 1e-3, 3e-3}`, 16 seeds, 10,000 steps, best LR shown per arm.

![architectures](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_exp7_arch.svg)

| | transformer | causal mixer | unmasked mixer | chance |
|---|---|---|---|---|
| **`s=7`** (paper's cell) solves | **0/16** | **5/16** | 16/16 | |
| median `t*` | — | 5844 | **392** | |
| support IoU, all seeds | 0.47 | 0.35 | 0.31 | 0.28 |
| support IoU, **solved seeds only** | — (none) | **0.48** | 0.31 | 0.28 |
| **`s=3`** (easy cell) solves | **16/16** | 4/16 | 16/16 | |
| median `t*` | 820 | 7986 | **386** | |
| support IoU, all seeds | **0.80** | 0.63 | 0.12 | 0.10 |
| support IoU, **solved seeds only** | **0.80** | **0.73** | 0.12 | 0.10 |

## Three findings

**1. The direction of the paper's claim replicates — at its own config.** At `s=7`, where our
transformer fails at every learning rate tried, the causal mixer solves 5/16. Attention-free
mixing does succeed where attention cannot. That is the substance of H5 and it holds.

**2. The easy-cell result was a learning-rate artifact — see exp8 below.** exp7 reported the
mixer at 4/16 on `s=3`, but ran only one learning rate there while sweeping three at `s=7`.
At `lr=1e-3` the mixer solves `s=3` **16/16**. The claim that it "loses badly where the search
is easy" was wrong.

**3. Without causal masking the comparison is void, and the IoU proves it.** The unmasked mixer
reaches *exactly zero* loss in ~390 steps in both cells — while its support IoU is **0.31 and
0.12**, at or below the untrained baseline. It solves the task perfectly having learned nothing
about the pattern, because it reads the answer from the token it is predicting.

That third point speaks to a specific paper claim. They report the mixer "outperforms a
transformer by an order of magnitude in learning the ground-truth attention pattern" — but a
leaking model has *chance-level* alignment by construction, so their pattern-learning result
cannot come from an unmasked model. Their mixer was therefore very likely masked.

**Correction.** An earlier version of this page said our causal mixer's alignment sat *below*
the transformer's (0.35 against 0.47), so we did not reproduce a mixer that finds the pattern
better. That comparison was wrong: it averaged the mixer's IoU over eleven seeds that never
learned anything and compared it against a transformer that never solved the cell at all.
Conditioned on seeds that actually solve, the mixer reaches **0.48** at `s=7` — the only arm
that both solves the cell and sits meaningfully above chance on it. The right statement is
that we cannot compare pattern-learning between an arm that succeeds and an arm that never
does; the paper's magnitude claim remains untested here rather than contradicted.

The chance column is why this matters. Random top-`s` selection scores 0.28 at `s=7` and 0.10
at `s=3`, so the unmasked mixer's 0.31 and 0.12 are **exactly chance** — a rigorous statement
of "learned nothing about the pattern", not an eyeballed one.

## exp8 — the crossover, swept properly

Both arms across `s ∈ {3…8}`, two learning rates each, 16 seeds, best setting per cell:

![crossover](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_exp8_crossover.svg)

| `s` | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| **transformer** solved | **1.00** @732 | 0.50 | 0.06 | 0.00 | 0.00 | 0.00 |
| **mixer** solved | **1.00** @3693 | **0.69** | **0.62** | **0.31** | **0.31** | **0.19** |
| transformer, every row exact | **1.00** | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 |
| mixer, every row exact | 0.81 | 0.19 | 0.06 | 0.00 | 0.06 | 0.00 |
| chance IoU | 0.10 | 0.14 | 0.19 | 0.23 | 0.28 | 0.33 |

**The crossover is at `s = 4`.** Below it attention is better — both architectures solve `s=3`
completely, but the transformer gets there in 732 steps against the mixer's 3693, five times
faster. From `s=4` on the mixer leads, and from `s=5` it is the only architecture that solves
anything at all: 10/16 where attention manages 1/16, then 5/16, 5/16 and 3/16 across cells
where attention never once succeeds.

**So the paper's claim holds, in the regime it was made about.** Sparse patterns are hard to
*find* by softmax competition, and an architecture that learns its mixing weights directly
keeps working past the point where attention stops. What does not hold is the unqualified
"learns the linear map faster": at low sparsity the transformer is five times quicker, and the
advantage only appears once the search gets hard.

**One caveat the strict metric exposes.** The mixer's wins are mostly *partial* solutions. At
`s=5` it "solves" 10/16 by the `acc2 > 0.95` bar but learns every row in only 1/16 — at
`S=16` that bar tolerates one unlearned row. Read strictly, both architectures collapse to
near-zero past `s=4`; the mixer's advantage is real but sits largely in the band between
"learns most rows" and "learns all of them".

**And when the mixer solves, it has found the pattern.** Alignment among solving seeds runs
0.77 / 0.57 / 0.49 / 0.48 at `s=3,4,5,7` against chance levels of 0.10 / 0.14 / 0.19 / 0.28 —
comfortably above chance everywhere, and nothing like the unmasked arm's chance-level 0.31.

## The learning rate was not a detail

The causal mixer went **0/16 → 5/16 → 3/16** across `3e-4 → 1e-3 → 3e-3`. exp6 gave both arms
`3e-4` in the name of fairness, which is exactly backwards: identical hyperparameters are only
fair when the optimum is shared. At exp6's LR the mixer looked hopeless at the paper's cell;
tuned, it beats the transformer there.

## Caveats

- **Our causal mixer is weaker than a standard Mixer by construction.** A Mixer's token-mixing
  is a two-layer MLP over positions, which cannot be made causal — its hidden units see every
  position. The causal analogue is a single masked matrix: 1,024 mixing parameters against the
  transformer's ~65k of QKVO. If the paper used the two-layer form, it either leaked or used a
  masking scheme it does not describe.
- **Three learning rates, one architecture family.** The paper compares seven architectures
  (Mamba, RWKV, xLSTM, Gated DeltaNet, linear RNN); we test one.
- 16 seeds, `A` drawn per seed, `S=16` only.
