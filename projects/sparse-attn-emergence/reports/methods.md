# Methods

Everything here is fixed across experiments unless a page says otherwise. Deviations from
the paper are listed at the bottom rather than buried.

## The linear map task

A matrix `A ∈ {0,1}^{S×S}` is sampled with **exactly `s` ones per row**. The transition is

```
f(x) = A x  mod 2
```

Each training sequence is `x₀ ~ Uniform{0,1}^S` followed by `x₁ = f(x₀)`, flattened to
`S·T` tokens with `T = 2` and vocabulary `C = 2`. The model does ordinary autoregressive
next-token prediction. Samples are drawn fresh every step — there are no epochs, so
training loss is an unbiased estimate of test loss.

The construction is the point:

> Predicting token `S+i` requires attending to exactly the `s` positions where row `i` of
> `A` is 1. **The correct attention pattern is known in advance**, for every query.

That is what makes "did it find the pattern" measurable instead of interpretive.

Two consequences worth stating:

- **The first half of every sequence is unpredictable.** `x₀` is i.i.d. uniform, so its
  cross-entropy is exactly `ln 2` no matter how good the model is. All headline metrics
  here use **second-half tokens only**; a full-sequence loss would be half noise floor.
- **`s = 1` and `s = S` are both easy.** One position to find, or "attend to everything",
  which near-uniform attention already approximates. The interesting regime is in between
  — that non-monotonicity is H2.

Batch size is set from a fixed token budget, `BATCH_TOKENS / (S·T)`, so tokens-per-step is
constant as `S` varies and the sweep compares like with like (the paper's protocol).

## The cellular automata task

For exp5, not yet run. A lookup table `R : {0..C−1}^W → {0..C−1}` with `W = 3`, `C = 4`,
composed `k` times per transition so the required attention span is `2k+1` wide;
`T = 16` states flattened into one sequence. Plateau is `ln 4 ≈ 1.386`.

`N = 256` is a **pool of rules**, per the paper's appendix: *"one rule is sampled per
training example"*. The pool is drawn once per run; each sequence then uses one rule.

That difference is structural, not cosmetic. The linear map has a single `A` per run, so
the pattern can be memorised into the weights. The CA task hands the model a different
rule every sequence, so it must infer the active rule **from the sequence itself** — exp5
tests whether a sparse-attention circuit emerges *in context*, where exp1–exp4 test
in-weights learning. State size is unstated in the paper; we use `S = 16`.

## Model

Paper defaults: **1 layer, `D = 128`, MLP 512, `H = 8` heads** (`d_head = 16`) for the
linear map; 4 layers for the CA task. Pre-LN blocks, learned positional embeddings, causal
mask, untied output head. 202,626 parameters per seed.

AdamW, `lr = 3e-4`, 200-step linear warmup then constant, weight decay 0.01, 10,000 steps.

Parameters are held in a flat dict of arrays so a leading seed axis vmaps cleanly — all
seeds of a configuration train in one process, one XLA program.

## Metrics

| Name | Meaning |
|---|---|
| `loss2` | second-half cross-entropy, nats. **`ln 2 = 0.693` means total failure** (uniform prediction); `→ 0` means solved |
| `acc2` | second-half exact-token accuracy |
| `t*` | **time-to-emergence**: first step whose trailing-mean (window 10) `acc2` exceeds a threshold. Reported at 0.90 / 0.95 / 0.99; **0.95 is the headline**. A run that never crosses is **censored**, never silently dropped |
| `solve_rate` | fraction of seeds emerging within budget — the observable for H1 and H2. A mean loss curve cannot show this |
| `ent_min` | attention entropy `−Σ sᵢⱼ log sᵢⱼ` of the most-peaked head, averaged over second-half query rows |
| `iou_head` | pick one head, average its top-`s` overlap with the true row support across rows, take the best head |
| `iou_row` | per row take the **best head**, then average over rows |

`iou_head` versus `iou_row` matters more than it looks. exp1 logged only `iou_head` and it
saturated at 0.49–0.97 while loss was already ~0 — because different heads specialise on
different rows of `A`, and picking a single head then averaging understates the model.
`iou_row` is the honest aggregation and is used from exp2 onward. Both are reported so the
two are never confused.

For candidate keys, both IoU variants rank only positions in `[0, S)` — where the support
lives by construction. Attention to second-half positions is captured through entropy
instead.

## Deviations from the paper

Stated up front, because they change how the numbers should be read:

1. **16 seeds, not 3.** The paper averages 3 seeds; H1/H2 are distributional claims, so we
   trade a little breadth for a real spread estimate.
2. **Learning rate and warmup are ours** (`3e-4`, 200 steps) — the paper does not state
   them. A constant post-warmup schedule keeps the plateau readable.
3. **exp1 fixes `A` across seeds**; exp2 draws a fresh `A` per seed. Fixed `A` isolates
   search noise at constant difficulty; per-seed `A` makes a sweep cell describe the
   `(S, s)` regime rather than one lucky matrix.
4. **Two architectures in exp6**, not seven — MLP-Mixer versus transformer. Mamba, RWKV,
   xLSTM, Gated DeltaNet and the linear RNN are out of scope at this scale.
5. **The CA rule-count reading is unresolved.** The paper lists `N = 256` beside
   `C = 4, T = 16, k = 1, W = 3`, which does not sit cleanly with a single lookup table of
   `C^W = 64` entries per run. To be resolved before exp5; if it stays ambiguous, one fixed
   `R` per run, stated as a deviation.
6. **Emergence threshold is a judgement call.** `acc2 > 0.95` is the headline; `t*` is
   also recorded at 0.90 and 0.99 so nothing hinges on the choice.
