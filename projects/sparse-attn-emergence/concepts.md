# Sparse-Attention Emergence — Concepts

Replication of arXiv:2606.25010. See [proposal.md](proposal.md) for the staged plan
and the claims table (H1–H5).

## Task / data

**Linear map.** `A ∈ {0,1}^{S×S}` with exactly `s` ones per row, fixed per run.
`x₀ ~ U{0,1}^S`, `x₁ = A x₀ mod 2`, sequence `= concat(x₀, x₁)` of `S·T` tokens
(`T = 2`), vocab `C = 2`. Autoregressive next-token prediction, fresh samples every
step (no epochs — train loss is an unbiased test loss).

Predicting token `S+i` requires attending to exactly the `s` positions where row `i`
of `A` is 1. **The ground-truth attention support is known by construction** — that
is the whole point of the task.

Batch size comes from a fixed token budget (`BATCH_TOKENS / (S·T)`) so tokens-per-step
is constant across `S`.

**Cellular automata** (exp5, not yet built). Lookup table `R : {0..C−1}^W → {0..C−1}`,
`W = 3`, `C = 4`, composed `k` times per transition so the required span is `2k+1`;
`T = 16` states flattened.

`N` resolved from the paper (Appendix B.1): **"N: Number of rules; one rule is sampled per
training example"** — `N = 256` tables are sampled once per *run*, then each example draws
one of them and iterates `x_{t+1} = r^k(x)` from a random initial state.

This makes exp5 structurally different from exp1–exp4, and it is worth being explicit
about: the linear map has **one** `A` per run, learned into the weights, whereas the CA
task presents a *different* rule per sequence, so the model must infer which rule is
active **from the sequence itself**. exp5 therefore tests sparse-attention emergence in an
in-context setting, not an in-weights one. State size is not stated in the paper; we use
`S = 16` and record it as ours.

## Model & loss

Paper defaults: linear map = 1 layer, `D = 128`, MLP 512, `H = 8` heads
(`d_head = 16`); CA = 4 layers. Pre-LN blocks, learned positional embeddings, causal
mask, untied output head. AdamW, `lr = 3e-4`, 200-step linear warmup then constant,
`wd = 0.01`, 10,000 steps. **LR and warmup are ours** — the paper doesn't state them.

Params are a flat dict so a leading seed axis vmaps: every seed of a config trains
simultaneously in one process (`lib/models.py`).

## Metrics

The first half of each sequence is i.i.d. uniform — CE exactly `ln 2`, no signal. **All
headline metrics use second-half tokens only.**

| Metric | Definition |
|---|---|
| `loss2` | second-half CE in nats. `ln 2 = 0.693` = total failure, `→ 0` = solved |
| `acc2` | second-half exact-token accuracy |
| `t*` | first step whose trailing-mean (window 10) `acc2` exceeds a threshold; reported at 0.90 / 0.95 / **0.95 main** / 0.99. Never reached ⇒ **censored**, not dropped |
| `solve_rate` | fraction of seeds emerging within budget — the H1/H2 observable |

**The `s = S` column is degenerate — do not read it as a solve.** With `s = S` every row of
`A` is all-ones, so all second-half tokens equal `parity(x₀)`: one value repeated. A model
that computes nothing can guess position `S` at chance and **copy** it for the remaining
`S−1`, scoring `1 − 0.5/S` accuracy and leaving exactly `ln 2 / S` loss. Verified directly
by `scripts/check_dense_shortcut.py`: at `S=16`, position 16 accuracy is 0.488 while
positions 17–31 are all 1.000, overall 0.9680 vs the predicted 0.9688, loss 0.0437 vs
`ln 2/16 = 0.0433`.

Consequences: (a) `acc2 > 0.95` is passed by a pure copier once `S ≥ 16`, so `t*` at `s=S`
measures copying latency, not emergence; (b) `S=8, s=8` is the *only* dense cell that had
to learn the map, because copying there yields 0.9375, below threshold — which is why it
took 1247 steps and reached `loss2 = 0.0000` while `S=16/32` "solved" in ~30 steps and
stalled at `ln 2/S`.
| `ent_min` | attention entropy `−Σ s_ij log s_ij` of the most-peaked head, averaged over second-half query rows |
| `iou_max` | best head's IoU between its top-`s` attended keys **among the first-half positions** and the true support of row `i`, averaged over rows. Restricting to `[0, S)` is deliberate — that is where the support lives; second-half attention is scored only through entropy. Scalarises the paper's before/after attention maps |

## Findings

### exp1 — H1 partially replicated (S=16, s=3, 16 seeds, fixed A, 167 s total)

- **Plateau is where theory says.** Mean pre-jump `loss2` = 0.6894 vs `ln 2` = 0.6931.
- **Timing is strongly seed-random.** `t*` = 469 … 2521 steps, median 885, a **5.4×
  spread** with `A`, hyperparameters and token budget all identical — only init and
  data order differ. This is the H1 claim and it holds clearly.
- **Abrupt, but not a cliff at this config.** `loss2` 0.6 → 0.05 takes a median **354
  steps ≈ 0.42 × t\***, ranging 78 (seed 2) to 2173 (seed 14). The paper's figures
  suggest sharper transitions; at S=16, s=3 the drop is fast relative to when it
  starts, not instantaneous.
- **This config is inside the learnable regime**: 16/16 solved, final `loss2` ≤ 3e-5.
  The "unlearnable medium-sparsity window" is an exp2 question, not visible here.
- **Metric caveat found.** `iou_max` picks one head then averages rows. Final values
  span 0.49–0.97 while loss is ~0, so the circuit **spreads rows across heads** and
  single-best-head aggregation understates alignment. exp4 should use per-row
  best-head IoU; exp1's stored `diag_iou_max` should be read with that in mind.
