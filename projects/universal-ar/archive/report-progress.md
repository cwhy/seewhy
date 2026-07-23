# Universal AR — Progress Report

*Updated 2026-07-24. Covers exp2–exp5 (exp1 was a smaller pilot, superseded by
exp2's matrix-completion framing). All numbers are pulled directly from
`projects/universal-ar/results.jsonl` and `logs/exp2.log`–`logs/exp5.log`.*

## 1. Framing

The core idea ([concepts.md §1](concepts.md#1-the-core-reframing--dataset-as-a-stream-of-records))
is to dissolve the "sample" as the atomic unit. A datum is a **record**
`(value, pos, ref)`: a value at a position, tagged to a reference/sample id.
The model ([concepts.md §2](concepts.md#2-the-model--auto-associative-memory-with-generic-completion))
is a **symmetric auto-associative memory** built from role⊗filler (HRR)
bindings, `M = Σ p⊗p`, so any field can cue any other — there's no fixed
key→value direction. Every task, including classification, is completion of a
masked field ([concepts.md §3](concepts.md#3-evaluation--the-label-is-just-another-field)):
label prediction is completing the `pos_label` field; pixel inpainting is
completing a pixel-position field.

Concretely this is a **matrix-completion** problem: rows = samples (`ref`),
columns = positions (`pos`, including one label column and, from exp4 on, one
`ref`-self column), entries = values. Each episode observes a subset of
entries per row. Three read-out tests probe different things:

1. **Retrieval (`recall`)** — recall an observed entry (self-read).
2. **Label generalization (`label_attn`/`label_lin`)** — predict a withheld
   label for a row whose label was never given (leave-self-out cross read).
3. **Content generalization (`ink_attn`/`ink_lin`)** — predict a held-out
   *ink* pixel entry for another row (leave-self-out cross read; restricted to
   true bin > 0 so the ~81%-background pixels don't dominate the metric — see
   caveats).

Both readouts from [concepts.md §5a](concepts.md#5a-readout--keep-both-linear-and-attention-as-proposals)
are computed from the same stored patterns: **linear** (`p̂ = M q`, O(1),
truly continual but lossy) and **attention/Hopfield** (`p̂ = Σ softmax(β·pᵢ·q) pᵢ`,
exact recall, the accuracy ceiling). Attention consistently dominates linear
below, so it's the number to read for "how good can this get."

## 2. What exp2–exp5 tested

All four train on MNIST, `D=256`, 32 value bins, Adam lr=1e-3.

| | variant | loss | episode size S | v_refs | steps | params | wall time |
|---|---|---|---|---|---|---|---|
| exp2 | B (cross-completion) | leave-one-out cross-sample recon | 32 | — | 4000 | 211,713 | 4107s |
| exp3 | A (self-recall-only) | pure self-recall, no cross term | 32 | — | 3000 | 211,713 | 3739s |
| exp4 | A + ref self-recall | self-recall + ref-field self-recall | 48 | 128 | 5000 | 244,737 | 113s |
| exp5 | A + ref, long context | same as exp4 | **256** | 384 | 5000 | 310,273 | 149s |

- **exp2 (variant B)** — the loss reconstructs **observed** entries via
  leave-one-out cross-sample completion (masked pixel recon + leave-one-out
  label). Nothing held-out enters the loss; label/content generalization are
  measured to see how well they emerge under an explicit cross-sample
  objective.
- **exp3 (variant A)** — the loss is **pure self-recall**: each row
  reconstructs only its own given data (the label is stored into the row's
  own pattern so there's something to self-recall). No cross-row term in the
  loss; cross-sample completion is measured as a purely emergent property.
  Adds the train/test split and the ink-only content metric (exp2 had
  neither).
- **exp4 (variant A + ref)** — adds a `ref`-self field (the row recalling its
  own `ref` binding) to the self-recall loss, and switches the `val_emb`
  gather to a dense one-hot matmul (see §4 below) — this is what makes exp4
  113s instead of exp2/exp3's ~1 hour.
- **exp5** — same as exp4 but with **8x longer episodes** (S=256 vs S=48),
  to test whether variant A's self-recall training holds up with far more
  in-context distractor samples per episode.

## 3. Loss curves (the headline ask)

![Loss curves for exp2–exp5. exp3/exp4/exp5 (variant A, self-recall) and exp2 (variant B, cross-completion) plotted together; log scale. Absolute loss level is not comparable across variants — exp4/exp5 add a ref self-recall term and exp2's loss composition differs entirely (leave-one-out cross reconstruction vs pure self-recall) — read the SHAPE and convergence point, not the level.](https://media.tanh.xyz/seewhy/26-07-24/universal-ar_loss_curves.svg)

All four runs converge fast and plateau by roughly **step 1000–2000**, then
stay flat (exp4/exp5 run 5000 steps mostly past the point of further loss
improvement; exp2/exp3 similarly flatten early relative to their total step
budgets). None show late-training instability or divergence. Because the
loss terms differ by variant (exp4/exp5 add a ref-recall term; exp2 is a
cross-completion objective, not self-recall), **absolute loss values are not
meaningfully comparable across the dashed (exp2) vs solid (exp3/4/5) curves**
— only within a variant. What *is* comparable and instructive: all four
optimize cleanly and hit their plateau at a similar step count regardless of
episode size (S=32 to S=256) or added ref term, i.e. none of this is an
optimization-difficulty story — the metric differences in §5–6 come from the
generalization side, not from harder/easier optimization.

## 4. Efficiency fix (why exp4/exp5 are ~35x faster than exp2/exp3)

exp2/exp3 took ~3700–4100s (≈1 hour) for 3000–4000 steps. Profiling
(`logs/diag.log`) found the backward pass costing **~7.2s/step**
(`train step 1: 7.209s`, `step 2: 7.269s`, …). Cause: `val_emb[bins]` is a
gather from a `(32, D)` embedding table; real MNIST pixels are ~81%
background (bin 0), so nearly every one of the ~keep~400 observed entries per
row maps to `val_emb[0]`, and the backward pass scatter-adds almost all
gradients onto that single row — XLA serializes this rather than
parallelizing it. Replacing the gather with a **one-hot matmul**
(`onehot @ val_emb`, dense backward, no scatter) removed the contention:
`logs/prof.log` shows the same step at **21ms** post-fix — a **~345x**
speedup (`~7.2s → 21ms`). exp4 and exp5 use this fix and finish 5000 steps in
**113s and 149s** respectively, vs. exp2/exp3's ~1 hour for 3000–4000 steps.

## 5. Results (final checkpoint)

| metric | chance | exp2 (B, S=32) | exp3 (A, S=32) | exp4 (A+ref, S=48) | exp5 (A+ref, S=256) |
|---|---|---|---|---|---|
| recall | — | 0.839 | 0.922 | 0.924 | 0.924 |
| ref_recall | — | n/a (no ref field) | n/a | **1.000** | **1.000** |
| label_attn | 0.10 | **0.741** | 0.404 | 0.324 | 0.154 |
| label_lin | 0.10 | 0.127 | 0.133 | 0.111 | 0.101 |
| ink_attn | 0.031 | not measured (confounded raw ~0.83, see caveats) | 0.131 | 0.116 | 0.095 |
| ink_lin | 0.031 | not measured | 0.054 | 0.055 | 0.039 |

Source: `results.jsonl`, one row per `experiment` (`exp2`..`exp5`). exp4/exp5
train≈test throughout (`tr_*` fields track within ~0.03 of the top-level test
fields at every checkpoint) so only test-set numbers are shown.

### Context-length trend

![label_attn and ink_attn (final checkpoint, attention read) vs episode size S, across exp3 (S=32), exp4 (S=48), exp5 (S=256) — all variant A, self-recall-only training. Both metrics fall monotonically as S grows.](https://media.tanh.xyz/seewhy/26-07-24/universal-ar_context_length_trend.svg)

Longer in-context episodes **hurt** variant A's cross-sample generalization:
`label_attn` goes 0.404 → 0.324 → 0.154 and `ink_attn` goes 0.131 → 0.116 →
0.095 as S goes 32 → 48 → 256. `recall` and `ref_recall` (self-operations,
computed for the row's own data) are unaffected — they stay flat at
~0.92 / 1.00 across all three.

### Representative training curves (exp4)

![exp4 (variant A+ref, S=48) metric curves over training, test set: recall and ref_recall shoot up early and stay near-ceiling; label_attn and ink_attn rise more slowly and plateau well below the recall level, well above their chance baselines.](https://media.tanh.xyz/seewhy/26-07-24/universal-ar_exp4_curves.svg)

## 6. Findings

1. **Efficiency fix: 345x speedup from removing a gather-induced backward
   bottleneck.** `val_emb[bins]` gather's backward scatter-adds ~all
   gradients onto `val_emb[0]` because ~81% of MNIST pixels are background,
   and XLA serializes that scatter. A one-hot-matmul rewrite (dense backward)
   cut the step time from ~7.2s to ~21ms (`logs/diag.log`, `logs/prof.log`).
   This is why exp4/exp5 finish in ~2 minutes versus exp2/exp3's ~1 hour.
2. **Ref self-recall is perfect and free.** `ref_recall` = 1.000 at both
   S=48 (exp4) and S=256 (exp5) — adding a self-recall field for the `ref`
   binding itself is a well-posed, easily-learned addition; it costs nothing
   in recall/ref_recall regardless of context length.
3. **Longer context hurts variant A's cross-sample generalization,
   monotonically.** `label_attn`: 0.404 (S=32) → 0.324 (S=48) → 0.154
   (S=256). `ink_attn`: 0.131 → 0.116 → 0.095. Mechanism: variant A's loss
   trains *only* self-recall, so the cross-sample kernel (what lets one row's
   query match another row's stored pattern) is never directly trained —
   it's a side effect of the self-recall representation. More in-context
   samples per episode means more distractor rows competing in the softmax
   for that untrained kernel, so cross-read accuracy degrades as S grows.
   `recall`/`ref_recall` are self-reads (a row against its own stored
   pattern) and are unaffected by how many other rows are in the episode.
4. **Label generalization only partially emerges from self-recall; the
   explicit cross-completion loss (variant B) roughly doubles it.** exp2's
   `label_attn` (0.741, S=32) vs. exp3's (0.404, S=32, same episode size) —
   nearly 2x. Content/ink generalization is weak throughout (exp3 best case:
   0.131 attn / 0.054 lin, vs 0.031 chance) and clearly needs the cross term
   more than label does.
5. **Attention read dominates linear read for cross-sample metrics.** Across
   all rows, `*_attn` ≫ `*_lin` for label and ink (e.g. exp4: label 0.324 vs
   0.111; ink 0.116 vs 0.055). The linear (truly continual, O(1)) memory is
   close to chance on cross-sample reads throughout — whatever
   generalization exists lives mostly in the exact-match attention read, not
   in the summed linear memory.

## 7. Caveats

- **exp2's content metric was background-confounded and never re-measured
  with the ink-only fix.** ~81% of MNIST pixel entries are background
  (`bg=0.81` in exp3–exp5's logs). exp2's raw content accuracy (~0.83,
  `content_attn`/`content_lin` in `results.jsonl`, not in the table above)
  is barely above "always predict background" (0.81) and was never broken
  out to ink-only pixels — not informative about real content
  generalization. exp3 onward add `ink_attn`/`ink_lin` (restricted to true
  bin > 0); that's the metric to trust, and exp2 has no comparable number.
- **`ref` is not yet anonymized.** Per [concepts.md §4a](concepts.md#4a-random-re-assignment-each-episode--the-anonymization-trick),
  the design calls for re-randomizing the ref↔sample mapping (and ideally
  label values) each episode so the model can't memorize "ref = this
  content" and must read purely from in-context binding. None of exp2–exp5
  do this — labels and positions are fixed across episodes, so what's being
  measured is closer to amortized matrix completion (a learned row/column
  factorization) than to in-context learning in the anonymized-label sense.
  This likely caps label/content generalization and may also interact with
  the context-length-hurts-generalization finding (§6.3) — untested.
  Anonymization was deferred, not because it failed, but because it hasn't
  been implemented yet.
- **Content generalization is weak everywhere.** Best case across all four
  experiments is exp3's ink_attn = 0.131 vs 0.031 chance — a real but small
  effect, the least-solved of the three tests, and it degrades further with
  context length (§6.3).
- **exp2 vs exp3+ aren't a clean apples-to-apples ablation of "cross loss
  on/off."** exp2 never stores the label in the row's own pattern (it's
  decoded from neighbor-row context only); exp3+ store the label
  self-referentially (needed for self-recall). The label numbers are
  directional, not a controlled ablation.

## 8. Next steps

1. **exp6 — variant B at long context (S=256).** Since variant B trains an
   explicit cross-sample kernel (unlike variant A), test whether a *trained*
   kernel avoids the context-length degradation seen in exp3→4→5, or whether
   more distractors hurt it too.
2. **Mixed loss**: combine variant A's self-recall term with variant B's
   cross-completion term in one objective, aiming to get both variant B's
   higher label ceiling (0.74) and variant A's clean train≈test behavior and
   ref-recall trick.
3. **Anonymized/permuted labels** (concepts.md §4a): re-randomize the
   ref↔sample and label-value assignment per episode, forcing genuine
   in-context binding rather than a fixed row/column association — the
   prerequisite for calling this "in-context learning" rather than amortized
   matrix completion, and a likely precondition for context length to ever
   *help* rather than hurt.
