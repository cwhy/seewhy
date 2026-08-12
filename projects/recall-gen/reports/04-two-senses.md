# Report 4 — does training on recall produce generalisation?

The question the project was built to answer, answered directly. Everything here
is already in Reports 1–3; what is new is the framing, because the earlier
write-ups used the word *generalisation* in two different senses on the same
page and that is what made the result read as self-contradictory.

**Short answer: no.** With one caveat that is worth separating out rather than
waving at.

---

## The word means two things here

| | the question | answer |
|---|---|---|
| **Sense 1 — the mechanism** | Does the look-up circuit still work on inputs it never trained on? | **Yes, perfectly** |
| **Sense 2 — the knowledge** | Did it learn anything *about digits* that helps when there is nothing to look up? | **No** |

Sense 1 is what "retrieval generalises" meant in Report 1. Sense 2 is what you
almost certainly meant by the question. They are not two grades of one thing, and
the cleanest way to see that is to put them on the same images.

## One picture, both senses

The model below was trained on MNIST digits **0–4 only**. It has never seen a 5,
6, 7, 8 or 9 in any role. Every image here is one of those unseen digits.

![retrieval and completion on the same unseen digits](https://media.tanh.xyz/seewhy/26-08-12/recallgen_two_senses.png)

*Top block:* the answer is one of the 16 images in the context. The model finds
it and reproduces it — errors 0.01–0.09, and it picks the right one of the 16
**every time** (identification accuracy 1.000, chance 0.063).

*Bottom block:* the same kind of digit, but its true image is nowhere in the
context. Now the model has to actually know something. The bottom row shows what
"predict the average digit" looks like — the strategy that scores exactly 1.0.
Across 512 episodes the model scores **1.006**.

Columns are drawn at fixed percentiles (10/30/50/70/90) of the per-sample error
within each block, not taken in file order. That matters: the first six samples
of the bottom block happen to score 0.88, 0.67, 0.81, 1.04, 0.60, 0.59 against a
median of 1.03, so showing them in order would have flattered the model
substantially. **53% of samples are worse than predicting the average digit.**

Look at the bottom block carefully, because the failure is not the one you would
guess. The model is not producing mush. It produces a **sharp, confident, wrong**
digit — a 5 where the truth is a 6, a 5 where the truth is a 5 but drawn
differently, a mangled shape where the truth is an 8. Averaged over the test set
that is worth exactly as much as a blur. It has learned to draw *a* digit bottom,
not *this* digit's bottom.

## Where the model actually sits

Condition D (context never seen, answer absent), M = 16, all on the same episodes:

![where the recall-trained model sits among reference strategies](https://media.tanh.xyz/seewhy/26-08-12/recallgen_where_it_sits.png)

| strategy | error |
|---|---|
| copy the closest context image | 1.575 |
| best soft look-up from the context | 1.002 |
| predict the average digit | 1.000 |
| **recall-trained model** | **0.852** |
| linear regression, no context at all | 0.645 |
| same architecture, trained to complete | 0.458 |

Two readings, both true and both worth having:

* It **beats every strategy available from the context alone.** The best soft
  look-up scores 1.002 — and that is not a weak baseline, it is the *ceiling* on
  what any mechanism can extract from 16 random digits when the answer is not
  among them. Sixteen digits simply do not tell you how to finish a
  seventeenth; the temperature the sweep picks is the one that flattens the
  weights into a uniform average, i.e. it reconstructs the mean image.
* It **loses to ordinary linear regression** fitted on the same training data
  with no context at all (0.645). So whatever knowledge it has is less than what
  one matrix multiply extracts from the training distribution.

And it moves the wrong way with training: 0.635 at step 500 → 0.852 at step
12 000, while its retrieval improves from 0.380 to 0.015. The small amount of
completion ability it has early is an accident of the read being diffuse before
the keys sharpen, and recall training spends it.

## "But it gets better with a big context" — it stopped retrieving

This is the result that looks like a counterexample, and it is the reason the
answer needs care.

![gain and completion error against context size](https://media.tanh.xyz/seewhy/26-08-12/recallgen_sweep_two_senses.png)

The blue line is **gain** = error(answer absent) − error(answer present), both on
images the model has never seen. It is what having the answer in front of it is
worth. Orange is the completion error.

At M = 256 completion has improved to 0.561 — and gain has fallen to **0.004**.
For scale, models trained *never to retrieve* score −0.002, 0.006 and 0.010 on
the same metric. The recall-trained model is in that group. Three further
signatures agree:

* its best score, 0.443, sits exactly on the completion-trained ceiling
  (0.458, 0.450);
* it does four times better on images from its training pool than on novel ones
  (0.134 vs 0.561) — its advantage tracks what is in its *weights*, not what is
  in its context;
* it is far below the look-up ceiling (0.786), so the answers are not being
  extracted from the context at all.

So at large context the "recall-trained" model is no longer a recall model. It
memorised the digit distribution in its weights and ignores the context —
exactly what the completion-trained model does. Generalisation did not emerge
*from* recall; it appeared in the space recall vacated.

Shrinking the **memory** at fixed context (16 images, identical episodes, same
parameter count) reproduces the same trade — gain 0.835 → 0.646, completion
0.852 → 0.681 — which is what rules out "a bigger context is simply more
informative".

## The sharpest version: novel digit classes

Same model, same episodes, as novelty increases:

![identification accuracy and completion error under the digit split](https://media.tanh.xyz/seewhy/26-08-12/recallgen_digit_split.png)

Finding it: **1.000, 1.000, 1.000**. Completing it: 0.712, 0.705, **1.006**.

Novel images cost the completion nothing (0.712 → 0.705). Novel *classes* take
it to exactly the average-image level. Meanwhile identification does not move at
all. The retrieval mechanism transfers completely across the split; the
completion ability does not transfer at all.

That asymmetry has a plain explanation. Retrieval needs a map from image to key
that keeps distinct images distinct — a generic property of pixels, indifferent
to which digits exist. Completion needs the conditional distribution of the
bottom half given the top, which is specific to the shapes in the training set.
A prior for 0–4 applied to a 7 is not merely uninformative; it is wrong — which
is why the *completion*-trained model scores 1.224 there, worse than the
average image.

## And the recall solution is not even a useful starting point

2 000 steps of completion training, from the recall-trained weights against
random initialisation. Same budget, schedule, data, seed:

| initialised from | error |
|---|---|
| recall-trained weights | 0.439 |
| random noise | 0.454 |

Worth 0.015, about 3% relative — and the seed spread on the recall configuration
is ±0.014, so this is at the noise floor. Whatever recall training builds, a
model that has to generalise does not want it.

## The answer

> **Recall training teaches a general-purpose matching mechanism and no
> knowledge.** The mechanism generalises completely — to unseen images, and even
> to unseen digit classes, at perfect identification. It carries nothing with it.
> Where the model looked like it had started to generalise, it had stopped
> recalling.

## What would change this answer

Stated plainly, because a negative result is only as strong as its scope:

* **A task where the context is actually informative.** At M = 16 the ceiling on
  using the context is 1.002 — there is nothing to reward. A task where 16
  in-context examples genuinely determine the answer (a rule to infer, rather
  than a picture to finish) could give recall training something to generalise
  *toward*. That is the single most important untested case.
* **A re-readable context.** This is a fixed-size recurrent state by design, so
  the finding is about compressive memories. Attention does not have the
  capacity cliff that drives half the story here.
* **A metric that does not reward hedging.** MSE scores a sharp wrong answer and
  a blur alike, which is exactly what the bottom block of the first figure shows.
  A likelihood-based objective would rank them differently.
* **One dataset, one mask shape.** MNIST bottom-halves are unusually predictable
  from the top, which flatters both abilities.

## Sources

Every number above is a row in `projects/recall-gen/results.jsonl`; the full
write-up with methodology is at
<https://media.tanh.xyz/seewhy/paper/recall-gen_paper.html>. Figures regenerate
with `scripts/gen_viz_twosenses.py`. Rows: `exp1`/`exp10`/`exp11` (recall,
3 seeds), `exp2`/`exp7`/`exp12` (completion), `exp3` (mixed), `exp6`/`exp4`/`exp5`
(context sweep), `exp15`–`exp17` (state sweep), `exp8`/`exp9` (digit split),
`exp13`/`exp14` (fine-tuning), `baselines_*` (reference strategies).
