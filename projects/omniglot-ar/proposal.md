# Omniglot AR — Proposal

*Successor to `projects/universal-ar`. The token-level premise is unchanged; the
substrate is not. This proposal argues that MNIST was the wrong dataset for the
question universal-ar was asking, and that Omniglot is the right one.*

## What universal-ar established

The premise (see `projects/universal-ar/proposal.md`) is that a dataset is a flat
bag of `(pos, value, ref)` tokens and every task — classification, inpainting,
denoising — is one operation: complete a masked token's value. Classification is
predicting the value at `(pos_label, ref)`.

39 experiments on MNIST produced one clean, negative result:

| condition | 4v9 label accuracy | reading |
|---|---|---|
| deterministic labels (exp28) | 0.875 | the encoder can tell a 4 from a 9 |
| deterministic labels, PCA-32 (exp34) | 0.977 | ditto, and the architecture is not the bottleneck |
| anonymised labels (exp13, 15) | ~0.50 | **chance** |
| + MLP-combiner token embedding (exp22) | 0.461 | chance |
| + context-generated weights (exp24) | 0.508 | chance |
| + FiLM conditioning (exp25) | 0.422 | chance |
| + retrieval-only training data (exp26) | 0.516 | chance |
| + task-balanced loss, `W_LAB_GEN=8` (exp30) | 0.445 | chance |
| + PCA-32 features (exp31) | 0.477 | chance |

Only 0v1 — a pair separable by mean ink alone — ever cleared chance under
anonymisation. Held-out *pixel* completion (`ink_gen`) was 0.00 in every run that
used a pixel-bin vocabulary.

So: **the model learns class identity and cannot bind it to a per-episode label
token.** Six architectural interventions did not move it. That is a strong hint
the problem is not the architecture.

## Why MNIST cannot answer the question

1. **Ten classes, ~6 000 examples each.** Memorising a class prototype in the
   weights is always the fastest way down the loss surface. Anonymisation removes
   the *payoff* at evaluation but not the *pressure* during training: the label
   token is uncorrelated with the image across episodes, so the shortest path is
   to ignore the label pathway entirely. The gradient never has a reason to build
   the binding circuit.

2. **Train and test share classes.** universal-ar held out *samples*, not
   *concepts*. Its "generalisation" metric therefore measured within-class
   interpolation — a model that memorised ten prototypes scores perfectly on it.
   Nothing in the setup ever tested whether the mechanism extends to a class the
   model has not seen, which is the actual claim.

3. **MNIST digits have no part structure.** Proposal principle 6 — "bind seen
   parts into an unseen whole" — has no substrate. There are no parts.

Points 1 and 2 compound: under anonymised labels the task became *unlearnable by
memorisation* while the data still made memorisation the dominant gradient
signal. Chance was the predictable outcome, and it says nothing about whether
token-level in-context binding is achievable.

## Why Omniglot

Omniglot (Lake et al. 2015) was built to be the transfer-learning inverse of
MNIST, and it repairs all three defects:

1. **1 623 characters × 20 drawings.** Twenty examples is far too few to build a
   usable prototype in the weights. The only way to answer is to read the support
   set. The pressure that made MNIST degenerate is removed by the data itself.

2. **A native class-disjoint split.** The background set (964 characters, 30
   alphabets) and the evaluation set (659 characters, 20 alphabets) share no
   characters. Test episodes use characters the model has *never seen*, so
   memorisation is impossible **by construction** rather than by anonymisation.
   Any above-chance number is in-context learning, with no anonymisation trick
   needed — and label anonymisation, which broke MNIST, is here just the standard
   episodic protocol.

3. **Characters are composed of strokes**, grouped into alphabets. The hold-out
   principle has real substrate, and the alphabet grouping gives a second,
   coarser generalisation axis to test separately.

Two further practical wins: strokes are sparse and high-contrast, so
content-matching has strong signal; and the images are near-binary, which
collapses the pixel-bin vocabulary and removes the loss imbalance that let pixel
CE (~16 nats) swamp label CE (~0.05) in exp38/39.

## The claim under test

> A token-level attention model, trained only on episodes of Omniglot background
> characters, can classify **held-out characters from held-out alphabets** in
> context, above chance and above a pixel nearest-neighbour baseline.

If this fails too, the negative result is about the token-level formulation
itself rather than about MNIST — which is a genuinely informative outcome and the
reason for running it.

## Baselines (a result is only meaningful against these)

| baseline | 5-way 1-shot | why it is here |
|---|---|---|
| chance | 0.200 | floor |
| pixel 1-NN (cosine, 28×28) | ~0.40 | the number to beat to claim any learning |
| trained on background, tested on background chars | — | the memorisation gap; a large gap means it is not generalising |
| Lake et al. BPL | 0.968 | ceiling, for scale — not a target |

The pixel 1-NN baseline is computed in-repo, not quoted, because it depends on
the resize and inversion choices in `load_omniglot`.

## Experiment plan

1. **exp1** — 5-way 1-shot in-context classification, train on background, test on
   evaluation. The headline: does test accuracy clear chance and pixel 1-NN?
   Reports background-character accuracy alongside, as the memorisation gap.
2. **exp2** — N-way × K-shot sweep (N ∈ {5, 10, 20}, K ∈ {1, 5}). Does the
   mechanism degrade gracefully in N, as in-context learning should?
3. **exp3** — add masked-pixel completion to the objective. Does *one* mechanism
   do classification and inpainting together — the "universal" part of the claim
   that MNIST never got to test?
4. **exp4** — alphabet-level hold-out: train on 30 alphabets, test on the 20
   unseen ones, versus a within-alphabet control. Separates "new character" from
   "new writing system".
5. **exp5** — ablations: layer count, `ref` tag on/off, position embedding,
   support-set size.

## Deviations from universal-ar to carry forward

- Keep: unified value vocabulary, label as `pos_label`, learned position
  embeddings with no spatial prior, softmax attention, one-hot matmul for
  high-contention embeddings, `lax.scan` epoch loops.
- Drop: the anonymised-label apparatus as a *test* of in-context learning. It
  remains as the episodic label assignment, but the class-disjoint split is what
  now carries the claim.
- Add: an explicit pixel 1-NN baseline in every classification run, and a
  seen-characters control episode alongside every unseen-characters one.

## Status (updated 2026-08-06 — the plan above is left as pre-registered)

**exp1 ran and failed: chance, flat, for all 12 000 steps.** `exp2` was
therefore *not* the N-way × K-shot sweep listed above — sweeping episode shape
is uninformative when the headline shape is at chance. It became a difficulty
floor instead (2-way, double the observed pixels), and it too sat at chance.

The substrate argument in this document survives the result: memorisation
really is impossible here (seen and unseen both score chance), and the
information really is present (pixel 1-NN gets 0.431 and 0.664 on the same
pixels). What did not survive is the expectation that removing the MNIST
defects would be sufficient. Per "The claim under test" above, that makes this a
finding about the token-level formulation rather than about MNIST.

Revised order: exp5 ablations first (binarised values; ink-biased position
pool), then depth/width, then the sweep, and exp3's "one mechanism, many tasks"
test only once classification works at all. See `concepts.md` → Findings.

## Open questions

- Support-set breadth vs depth: 5-way 1-shot or 5-way 5-shot as the primary?
  (Plan: 1-shot as headline — it is the harder, more diagnostic case.)
- Binarise pixels or keep the 8-bit grey from bilinear downsampling?
- Should episodes draw all N classes from one alphabet (hard, confusable) or
  across alphabets (standard)? Both, as an exp2 axis.
