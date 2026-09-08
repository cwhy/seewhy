# AR-Recall — Concepts

Everything the project assumes, defined once. If a term is used anywhere in this
repository's `ar-recall` code or reports, it is defined here. Where a decision is
still open it says so explicitly rather than leaving a default to be inferred.

Companion documents: [`workflow.md`](workflow.md) for how to run things,
`../recall-gen/concepts.md` for the predecessor project this one restructures.

---

## 1. What the project is asking

Recall-gen represented one image as **one token** and asked a model to complete a
masked query image from a context of other images. It found that retrieval works
well inside the training distribution and collapses on new data
(`recall-gen/reports/22-what-comes-back.md`).

AR-Recall asks the same question when an item is **not** a token. Here an image
is a run of several hundred tokens, and the model must bind a label to content
spread across that run and hold it while thousands of other tokens go past. That
is the setting language models actually operate in, and it is not obviously the
same problem.

---

## 2. Data

### 2.1 Images

MNIST, at native resolution: **28 x 28 = 784 pixels**. No downsampling. Train and
test splits are the standard ones.

### 2.2 Value quantisation

A pixel's intensity, originally an integer in `[0, 255]`, is mapped to one of
**V = 16 value bins** by uniform binning of `[0, 1]`:

    bin(x) = min(floor(x * V), V - 1)        for x in [0,1]

Uniform, not quantile. Quantile bins would spread MNIST's background across
several bins and make the marginal baseline artificially weak. Uniform binning
keeps the fact that most of an MNIST image is background, which is a real
property of the data and which the baselines in section 6 have to account for.

`V` is a parameter; 16 is the default.

---

## 3. The token stream

### 3.1 Symbols

Three disjoint symbol classes in **one shared vocabulary**:

| class | count | meaning |
|---|---|---|
| **label** | `LAMBDA` (default 32) | an arbitrary name, with no fixed meaning |
| **position** | 784 | pixel index, raster order; position `p` always means pixel `p` |
| **value** | `V` (default 16) | a quantised intensity bin |

Vocabulary size is `LAMBDA + 784 + V` = 832 by default. Ranges are disjoint, so a
token id determines its class.

**Labels are arbitrary and positions are not.** A label symbol carries no meaning
across episodes — it is a pointer, and what it points at is decided fresh in each
episode (section 4.2). A position symbol means the same pixel in every episode.
This asymmetry is deliberate: it is what makes label binding an in-context
problem and position an ordinary learnable coordinate.

### 3.2 Triples

Every value in the stream is emitted as three consecutive tokens:

    <label>  <position>  <value>

so one pixel costs three tokens. A stream is a concatenation of such triples and
nothing else. Token `i` has **slot** `i mod 3`, which is 0 for label, 1 for
position, 2 for value.

### 3.3 Slot roles

The model receives, at every token, an embedding of the token id plus a learned
**role embedding** indexed by `i mod 3`. The role is redundant given the
vocabulary ranges but is cheap and makes the slot structure explicit.

---

## 4. An episode

### 4.1 Shape

    A(full)  B(full)  C(full)  D(full)   halfQ   ->   predict the other half of Q

Concretely, with `L` **context images** (default 4):

| phase | what is streamed | triples |
|---|---|---|
| context | `L` images, each all 784 positions in raster order | `784 L` |
| prompt | the **query image**, rows 0-13 only (its top half) | 392 |
| target | the query image, rows 14-27 (its bottom half) | 392 |

Total `784 L + 784` triples, i.e. `3 (784 L + 784)` tokens. Context images are
streamed in a random order; positions within an image are always raster order.

The **query image** is the image the episode is about. The **prompt half** is
rows 0-13 of it, which the stream shows. The **target half** is rows 14-27,
which the stream also contains (the model is trained on them) but which are what
evaluation scores. This mask is the same one recall-gen used: bottom 14 rows.

### 4.2 Labels are shuffled per episode

Each image in an episode gets a distinct label symbol, drawn uniformly without
replacement from the `LAMBDA` label symbols. A fresh draw every episode.

So the same label symbol names different images in different episodes, and the
same image gets different label symbols. **Nothing about a label can be
memorised in the weights.** The binding exists only inside the episode, which is
the whole point.

### 4.3 The present / absent condition — axis 1

The two arms differ in whether the query image also appears as a context image:

| arm | context | prompt | the target half is |
|---|---|---|---|
| **present** | `A B C D`, where `A` is the query image | half of `A` | **already in the stream** — every target pixel appeared verbatim in `A`'s context run |
| **absent** | `B C D E` — the query image is not among them | half of `A` | **not in the stream** — it has to be inferred |

In the present arm the query image is streamed twice: once in full as a context
image (under its label), and once again as the prompt half. A model that can
retrieve should score near 1.0 there; that is the ceiling, not a result.

This is exactly recall-gen's present/absent axis. `L` counts context images in
both arms, so the two have the same sequence length and their numbers are
directly comparable.

### 4.4 The held-out pairing — axis 2

Fix a set `H` of `(label symbol, position)` pairs, sampled once and fixed for the
life of the project. Default `|H|` is 1/16 of the `LAMBDA x 784` grid.

**During training**, no triple whose `(label, position)` is in `H` is ever
emitted. When label `l` is assigned to an image, the positions `p` with
`(l, p) in H` are simply skipped — so context runs have small holes, about 49
pixels out of 784 at the default rate.

**At evaluation**, target-half pixels are split by whether their pairing is in
`H`:

- **seen pairing** — this `(label, position)` occurred during training
- **held-out pairing** — it never did

The model has seen this label with other positions and this position with other
labels, never the two together. Axis 2 therefore measures **compositional
generalisation over the label-position binding**, not novelty of the image.

Note what axis 2 is not: it is not about whether the *image* was in the training
split. Images come from the standard MNIST train/test split, which is a separate
and additional axis the project may use later.

### 4.5 The 2x2

|  | seen pairing | held-out pairing |
|---|---|---|
| **present** (query image is in context) | A — retrieve, familiar pairing | B — retrieve, new pairing |
| **absent** (query image is not) | C — infer, familiar pairing | D — infer, new pairing |

A is the ceiling. D is the hard cell.

---

## 5. Model and training

### 5.1 Architecture

A pre-norm residual stack, `n_layers` blocks of (mixer, feed-forward), then a
linear head. Same shape as recall-gen's so results are comparable. Two mixers,
selectable, both **causal** — token `t` sees tokens `<= t` and nothing later:

- **`kda_chunkwise`** — the gated delta rule, `S <- S diag(a_t) + e_t k_t^T`,
  read `o_t = S_t q_t`. A bounded matrix state per head, so cost is linear in
  sequence length. See `workflow.md` for its chunking and its numerical
  precondition.
- **`attn_causal`** — ordinary softmax attention, as the reference.

**Attention cannot run the long configurations on the current box** (no flash
kernel for these shapes, so it pays `O(N^2)` memory and runs out past ~1,500
tokens). At `L = 4` and above the delta rule is the only mixer that fits. This is
a limitation of the hardware and the available kernels, not a finding.

### 5.2 Order information

The delta rule carries order in its recurrence. Attention does not, beyond the
causal mask. **Open decision:** whether to add a position-in-sequence encoding
(rotary, or learned absolute) so the two mixers are compared on equal terms.
Rotary is the safer default because absolute embeddings fix a maximum length and
these streams are ~12,000 tokens. Not yet implemented.

### 5.3 Loss

Cross-entropy at **value slots only**, over the `V` value symbols. Label and
position slots are not scored: they are drawn by the episode generator, so
predicting them is unlearnable noise that would add gradient without signal.

**Every value token is scored, including all the context images' pixels.** That
is where nearly all the supervision is: at `L = 4` an episode carries
`784 x 4 + 784 = 3,920` value targets, so a batch of 2 gives ~7,800 supervised
predictions per step. This is what makes the small batches that long sequences
force still trainable.

The 2x2 of section 4.5 is an **evaluation** slice over target-half tokens. It is
not a separate loss and it does not change training.

### 5.4 Sizes and cost

Measured on one RTX 4090, four layers, `d_model = 512`, 8 heads of `dk = 64`,
chunk 128 (`scripts/bench_mixers.py`):

| context images `L` | tokens | batch | ms/step | 48k-step run |
|---|---|---|---|---|
| 4 | 11,760 | 2-4 | ~170 | ~2.3 h |
| 8 | 21,168 | 2 | ~230 | ~3.1 h |
| 16 (recall-gen's M) | 39,984 | 1 | ~320 | ~4.3 h |

Memory is `batch x tokens`; every configuration that fits sits near 10 GiB and
the next batch up exceeds 24 GiB. The budget is roughly **35,000 token-samples
per step** whatever the shape. Buy back episode diversity with gradient
accumulation or both GPUs, not with a larger batch.

Default is `L = 4`.

---

## 6. Metrics, and what to read them against

All numbers below are measured by `scripts/baselines.py` on the MNIST test split
at `V = 16`, target half = bottom 14 rows. They involve no model.

### 6.1 Why plain accuracy will not do

**82.0% of MNIST pixels fall in bin 0.** So predicting "background" everywhere
scores **0.809** on the target half, and the position-conditional version of the
same trick scores **0.810** — one part in a thousand better. Against a ceiling of
1.0, plain value accuracy has a usable range of about 0.19, and most of a
reported number is background the model was never asked about.

| reference | target-half accuracy |
|---|---|
| chance, `1/V` | 0.0625 |
| marginal (always the most common bin) | 0.8086 |
| position marginal (most common bin at that position) | 0.8099 |
| copy ceiling (present arm, answer is verbatim in the stream) | 1.0000 |

Report it, but do not lead with it.

### 6.2 The primary metric: foreground accuracy

Value accuracy restricted to target-half pixels whose **true** bin is greater
than 0. That is 19.1% of the target half, and it is the part of the image that
carries the digit.

| reference | foreground accuracy |
|---|---|
| marginal | **0.0000** (it always says background) |
| position marginal | **0.0401** |
| copy ceiling | 1.0000 |

Floor 0.04, ceiling 1.0 — a range of 0.96 instead of 0.19. The conditioning is on
the *true* value, not the prediction, so it cannot be gamed by a model that
refuses to predict background.

### 6.3 The secondary metric: cross-entropy

Bits per value token, which is the loss itself and does not saturate:

| reference | bits / value token |
|---|---|
| uniform over `V` | 4.0000 |
| position-conditional (Laplace-smoothed, no context) | **1.0201** |
| perfect | 0 |

The position-conditional figure is the one that matters. A model scoring above
1.02 bits on the target half **has not used the context at all** — it is doing
something a lookup table of per-pixel histograms already does.

### 6.4 Graded error

Mean `|predicted bin - true bin|`, which distinguishes a near miss from a wild
one. The position marginal scores 2.09 over the whole target half and 10.43 on
foreground pixels alone — the second number being large precisely because
predicting background for a bright pixel is a 15-bin error.

### 6.5 What to quote

Lead with **foreground accuracy** and **bits per value token**, each beside its
position-marginal reference. Add overall accuracy and graded error for
completeness. This is recall-gen's lesson carried over: report the reference
point beside every number, and prefer a measure that degrades smoothly to one
that steps (`recall-gen/reports/22-what-comes-back.md`).

## 7. Worked micro-example

`LAMBDA = 4` labels `{L0..L3}`, images of 2 x 2 = 4 pixels, `V = 4` bins, top
half = row 0, `L = 1` context image, present arm.

Episode: images `X` (query) and labels drawn as `X -> L2`. Context is `X` itself
(present arm), so the stream is

    L2 P0 V1   L2 P1 V0   L2 P2 V3   L2 P3 V2      <- context: X in full
    L2 P0 V1   L2 P1 V0                            <- prompt: top half of X
    L2 P2 V3   L2 P3 V2                            <- target: bottom half, scored

24 tokens. The last two value tokens are what evaluation reports; every value
token is trained on. Had `(L2, P2)` been in `H`, both of its occurrences would
have been omitted from every training stream, and this episode would appear only
at evaluation.

In the absent arm the first line would be a different image under a different
label, and the last two values would have to be inferred.

---

## 8. Decisions still open

1. **Sequence position encoding** (5.2) — rotary, learned, or none. Affects the
   attention arm most.
2. **Hold-out fraction** for `H` — default 1/16, not yet justified by a
   measurement of cell sizes.
3. **Whether the context images' order should be fixed or random per episode.**
   Currently random. Fixed order would let the model use position-in-sequence as
   a proxy for label, which is exactly the shortcut the shuffling is meant to
   remove — so random is probably right, but it has not been tested.
4. **Train/test image split as a third axis.** Not used yet; all images currently
   come from the training split.

## 9. Findings

None. No training has been run.
