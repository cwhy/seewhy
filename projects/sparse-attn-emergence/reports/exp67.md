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

| | transformer | causal mixer | unmasked mixer |
|---|---|---|---|
| **`s=7`** (paper's cell) solves | **0/16** | **5/16** | 16/16 |
| median `t*` | — | 5844 | **392** |
| support IoU | 0.47 | 0.35 | 0.31 |
| **`s=3`** (easy cell) solves | **16/16** | 4/16 | 16/16 |
| median `t*` | 820 | 7986 | **386** |
| support IoU | **0.80** | 0.63 | 0.12 |

## Three findings

**1. The direction of the paper's claim replicates — at its own config.** At `s=7`, where our
transformer fails at every learning rate tried, the causal mixer solves 5/16. Attention-free
mixing does succeed where attention cannot. That is the substance of H5 and it holds.

**2. It is not a general speed advantage.** At `s=3` the transformer solves every seed in 820
steps while the causal mixer manages 4/16 in ~8000. So "the mixer learns the linear map
faster" is too broad: it wins only in the regime where the search is hard enough to defeat
attention, and loses badly where the search is easy.

**3. Without causal masking the comparison is void, and the IoU proves it.** The unmasked mixer
reaches *exactly zero* loss in ~390 steps in both cells — while its support IoU is **0.31 and
0.12**, at or below the untrained baseline. It solves the task perfectly having learned nothing
about the pattern, because it reads the answer from the token it is predicting.

That third point speaks to a specific paper claim. They report the mixer "outperforms a
transformer by an order of magnitude in learning the ground-truth attention pattern" — but a
leaking model has *poor* alignment by construction, so their pattern-learning result cannot
come from an unmasked model. Their mixer was therefore very likely masked, and the
disagreement is about magnitude: our causal mixer's alignment (0.35) sits **below** the
transformer's (0.47), so we do not reproduce a mixer that finds the pattern better.

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
