# Universal AR — Proposal (token-level, v1)

*A clean restart. The archived exp1–7 (per-sample HRR associative memory) is kept
under `archive/` for its results and lessons; this proposal adopts the token-level
consensus.*

## Premise
Dissolve the "sample". A dataset is a **flat bag of tokens**, each a
`(pos, value, ref)` triple. Every prediction — classification, inpainting,
denoising — is the **same operation**: complete a masked token's value given its
`(pos, ref)`. In-context, continual, compositional.

## Principles

![anatomy](https://media.tanh.xyz/seewhy/26-07-24/universal-ar_proposal_anatomy.svg)

1. **Token = `(pos, value, ref)`** — the atomic unit; `pos`+`ref` are the address,
   `value` the content.
2. **The label is just a token at `pos_label`** (its value is the class); the value
   vocab is **pixel-bins ∪ classes**. Classification = predict value at
   `(pos_label, ref)` — identical to completing any pixel.
3. **`ref` is a coreference tag**, randomly drawn per episode from a pool: **the
   same tag sits on all of a sample's tokens** (an exact-match key that groups
   them), re-randomized across episodes (blocks memorization). The correct `ref`
   is always present in the context.
4. **Attention is over TOKENS, not samples** — no per-sample aggregation. A small
   **multi-layer** attention over the flat bag (a set; permutation-invariant;
   position is a *field*, not sequence order).
5. **Completion = masked-token prediction**: a query token is `(pos*, _, ref*)`
   with the value replaced by a guess/MASK token; attention predicts the value.
6. **Hold-out principle**: a held-out joint `(pos, value, ref)` has each of `pos`,
   `value`, `ref` present *individually* in the context — only the joint is
   absent. The task is to **bind seen parts into an unseen whole**.
7. **One mechanism, many tasks.**

## Mechanism

![mechanism](https://media.tanh.xyz/seewhy/26-07-24/universal-ar_proposal_mechanism.svg)

A query `(pos*, _, ref*)` attends over the bag. The layers learn to (1) gather
`ref*`'s own tokens (exact ref match) into a content representation, then
(2) match cross-sample tokens at `pos*` from *similar* samples and read their
value. A single random `ref` can't drive step (2), so it is genuinely
**multi-layer**, not one associative read (that was the archived approach's limit).

## Proposed exp1 architecture
- **Token embedding**: `e = pos_emb[pos] + value_emb[value|MASK] + ref_emb[ref]`
  (learned tables; `ref` from a pool of `V`). *(HRR role⊗filler is an option.)*
- **L = 2–4** transformer layers (multi-head self-attention + MLP), no causal mask.
- **Head**: masked/query tokens → softmax over the unified value vocab; CE loss.

## Decisions kept from the archived arc
- Unified value vocab (pixel-bins ∪ classes); **label as `pos_label`**.
- Position = plain **learned** embedding, **no spatial prior**.
- `ref` = random pool assignment (anonymization) — now per-token / coreference.
- **Attention (softmax) over linear** — attention was needed for sparse retrieval.
- **Efficiency lessons**: host-gather images (avoid the O(N) in-jit gather);
  **one-hot matmul for high-contention embeddings** (MNIST is ~81% background →
  a gather's backward scatter serializes, 345× slower); device-sample
  positions/refs; jit the whole step.
- **Metrics**: retrieval (recall observed), label generalization, content
  generalization with **ink-only accuracy + a background baseline** (the raw
  pixel-accuracy is ~0.81 background and misleading).
- **Eval**: support from **TRAIN**, held-out query from **TEST**; *supervised*
  (full query) vs *supervised+* (more context); watch the train≈test gap.
- Models **converge fast** — modest step budgets; **context breadth** matters more
  than training length.
- The **hold-out principle** (parts present, joint absent) is now formalized.

## What changes vs archived
- Per-sample HRR aggregation → **token-level multi-layer attention**.
- Separate `r_label` field → **label as `pos_label`** (unified).
- Single associative read → **≥2 attention layers** (gather-then-match).

## Experiment plan
1. **exp1** — core token-level attention on MNIST: masked-token completion (pixels
   + label); establish label + content completion above the background/marginal
   baselines; train-support / test-query; ink metric.
2. **exp2** — hold-out & context-size sweep (does more context help; supervised vs
   supervised+).
3. **exp3** — **anonymized/permuted labels** → true in-context learning (class
   semantics inferred from context, not memorized).
4. **exp4** — ablations: # layers, `ref` on/off, position embedding, unified vocab.
5. **exp5+** — Fashion-MNIST / EMNIST; cross-dataset generalization.

## Open questions to confirm
- Token embedding: **additive learned role embeddings** (my default) vs a learned
  MLP combiner vs HRR role⊗filler?
- **# attention layers** (propose 2–4) and model width.
- Episode size / ref-pool size defaults.
