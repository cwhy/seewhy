# Report 5 — train short, test long; and how far the recall mechanism travels

Two evaluation-only experiments. No training happened for either: the KDA has no
length-dependent parameters, so a model trained at one context size runs at
another unchanged, and a model trained on MNIST runs on anything 28×28.

Rows: `transfer_length`, `transfer_dataset`.

---

## First, a correction to how Reports 1–4 talk about this

Those reports say the retrieval mechanism "generalises perfectly and for free"
and that it is "content-addressed, so it has nowhere to put a memorised
identity". That is too strong, and the second experiment below shows how much
too strong.

There is no free lunch here. **The retrieval mechanism was itself learned from
MNIST.** A key is not a distribution-free hash; it is a learned map from pixels
to a 64-dimensional address, and what makes two images land at different
addresses is a *similarity metric fitted to the training distribution*. Recall
training is not memorisation-free — it memorises at a coarser granularity.

That gives a better frame than the binary one:

| what gets stored in the weights | granularity | how far it travels |
|---|---|---|
| **route B** — individual images | one entry per training image | nowhere. 0.134 on training-pool images vs 0.561 on novel ones |
| **route A** — a similarity metric over the distribution | one function for the whole dataset | across digit classes, yes (1.000). Across datasets, only partly — measured below |

Neither route is free of the training data. The question is only how coarse the
dependence is, and the honest goal is to *minimise* the memorisation, not to
claim it has been eliminated.

---

## Experiment 1 — train short, test long

**The question.** Is the completion improvement at M=256 a property of large
contexts at inference time, or an artefact of what training selects for?

The prediction was recorded in Report 4 before running this: the short-trained
model should *not* improve at long context, because it never built a prior to
fall back on.

![the same three models at context sizes they never trained at](https://media.tanh.xyz/seewhy/26-08-12/recallgen_length_transfer.png)

### Result: the prediction holds, and more strongly than stated

Trained at M=16, evaluated on longer contexts (completion error on unseen
images, answer absent):

| test-time context | 4 | 16 | 64 | 256 |
|---|---|---|---|---|
| **trained at M=16** | 0.700 | 0.851 | 0.996 | 0.942 |
| for comparison, *trained* at M=256 | — | — | — | **0.561** |

The short-trained model does not merely fail to improve — it gets **worse**,
running up to the mean-image line. At a test context of 256 it scores 0.942
against the 0.561 that a model *trained* there reaches. Giving it a large
context at inference buys nothing, because there was never anything in it to
use.

Its retrieval degrades over the same range, which the capacity hypothesis
predicts — but the size of that degradation is easy to overstate, and an earlier
version of the figure above did. Identification accuracy runs 1.000 → 1.000 →
0.876 → 0.322, and **chance runs 0.250 → 0.063 → 0.016 → 0.004 alongside it**,
because chance is 1/M. At M=256, 0.322 is 82× chance, not a collapse.

`gain` is the metric without that problem — it needs no chance level — and it
runs +0.676 → +0.834 → +0.715 → +0.175. So the model is still retrieving
something at a context sixteen times larger than it trained on, and the honest
statement is that retrieval degrades substantially rather than fails.

**So the M=256 result is entirely a training-time selection effect.** Long
contexts do not unlock anything at inference; they change which solution gradient
descent finds.

### And the reverse: the retrieval circuit does not come back

Trained at M=256, evaluated at short contexts:

| test-time context | 4 | 16 | 64 | 256 |
|---|---|---|---|---|
| gain (what the answer being present is worth) | −0.011 | −0.002 | −0.002 | −0.014 |
| completion error | 0.541 | 0.545 | 0.558 | 0.545 |

Gain stays at zero at every length, including at M=4 where sixteen-fold spare
capacity is available. Completion error is flat to within 0.02 across a 64×
change in context size. **This model does not read its context at all**, and
shrinking the context back to a size it could easily hold does not make it start.

The completion-trained model behaves identically (gain −0.015 to −0.003, error
0.440–0.457, flat). Two models that took the in-weights route are
indistinguishable from each other at every context length.

**The two solutions are separate attractors, not two points on a continuum.**
Once training has selected one, inference-time context size does not move you
between them.

---

## Experiment 2 — how far does the recall mechanism travel?

**The question.** The recall-trained model identifies unseen MNIST digits, and
even unseen digit *classes*, at 1.000. How much of that is a general matching
ability and how much is MNIST?

Four pools of increasing distance from the training distribution, all 28×28 in
[0,1], all evaluated with the same model (trained on recall at M=16) and the same
16-image context size. Identification accuracy is the metric here because it needs no normaliser and so
is comparable across pools that have nothing else in common — and unlike the
length experiment, M is fixed at 16 throughout, so its chance level is a
constant 0.063 in every column.

![identification accuracy across four image pools](https://media.tanh.xyz/seewhy/26-08-12/recallgen_dataset_transfer.png)

| pool | what it is | identification accuracy |
|---|---|---|
| held-out MNIST | digits it never saw | **1.000** |
| Fashion-MNIST | same medium, entirely new content | **0.651** |
| MNIST, pixels permuted | *identical pixels and statistics*, spatial structure destroyed by one fixed permutation | **0.116** |
| random fields | blocky low-frequency noise | **0.222** |

Chance is 0.063.

### Result: the mechanism is substantially MNIST-shaped

This is your point, quantified.

* **Fashion-MNIST: 0.651.** Far above chance, so something genuinely general was
  learned — but a third of the retrievals fail on images that are still
  grayscale, still centred, still 28×28. Perfect identification does not survive
  a change of content.
* **Pixel-permuted MNIST: 0.116, barely above chance.** This is the sharpest
  result of the two experiments. The pixels are *exactly the same pixels*, with
  exactly the same marginal statistics and exactly the same pairwise distances —
  only the spatial arrangement changed, by one fixed permutation applied to
  every image. The mechanism collapses. So the learned keys are not reading
  "what distinguishes these images" in any layout-agnostic sense; they are
  reading MNIST's spatial structure specifically.
* **Random fields: 0.222.** Better than the permuted case, presumably because
  large smooth blobs are crudely separable by whatever low-frequency features
  the encoder has, but nowhere near usable.

Note that the permuted pool is, information-theoretically, exactly as easy as
MNIST — a nearest-neighbour matcher on raw pixels would score identically on
both. The model scores 1.000 and 0.116. The gap is entirely the learned encoder's
dependence on the training distribution.

---

## What this changes

**The paper's claim needs qualifying, and it survives the qualification.** The
finding was: retrieval training produces a general matching mechanism and no
knowledge. The correct version is:

> Retrieval training produces a matching mechanism that generalises *within the
> distribution it was trained on* — across unseen images and even unseen classes
> — and no knowledge usable when there is nothing to match. Both of the things it
> learns are memorised from the training data; they differ in granularity, and
> only the coarse one transfers at all.

Nothing in Reports 1–4 depends on the stronger version. The digit-split result
(identification 1.000 on classes never seen, completion 1.006 on the same
images) is unaffected — 0–4 and 5–9 are the same distribution in every sense that
matters to a pixel encoder, which is now demonstrated rather than assumed.

**The synthetic-data direction is the right response**, and these numbers say
what it has to beat. A model trained on a broader pool should push the
Fashion-MNIST number up from 0.651; the permuted number is the harder target,
because moving it requires an encoder whose notion of similarity is not tied to
spatial layout at all. Whether that is achievable — or even desirable, since
spatial structure is real information — is the open question. The floor is 0.063
and MNIST-only training reaches 0.116; anything above that is progress that can
be measured.

---

## Caveats

* Evaluation-only, so no seed variance is available here: these are single
  forward passes of single trained models. The models themselves come from runs
  whose seed spread is ±0.014 on gain.
* 256 evaluation episodes per cell rather than the 512 used elsewhere.
* Conditions A/C always use the MNIST training pool, in every dataset variant —
  only the novel pool changes. So the numbers quoted here are all from B/D.
* Completion error on the `noise` pool is not meaningful: the visible half of a
  random field carries no information about the hidden half, so the task is
  impossible by construction rather than merely hard.
* `Fashion-MNIST` uses its test split; `random fields` are 7×7 uniform noise
  upsampled 4×, which is blocky rather than smooth.

## Sources

`transfer_length` and `transfer_dataset` in
`projects/recall-gen/results.jsonl`. Regenerate with
`scripts/eval_transfer.py`, figures with `scripts/gen_viz_transfer.py`.
Full project write-up: <https://media.tanh.xyz/seewhy/paper/recall-gen_paper.html>
