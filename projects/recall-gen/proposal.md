# Recall-Gen — what to do next

Written after the first phase (18 training runs, 2 transfer evaluations, paper
published). This is a plan, not a report: the numbers below are either already
measured — in which case they are cited — or predictions, in which case they are
labelled as such so they can be checked rather than quietly forgotten.

---

## Where the project stands

Three things are established well enough to build on:

1. **Recall training produces a similarity metric and no knowledge.** Retrieval
   transfers to unseen images and unseen digit classes at identification 1.000;
   completion of an absent target ends at 0.852, worse than a linear regression
   that ignores the context (0.645), and *worsens* through training.
2. **The metric is bound to its training distribution.** 1.000 on held-out MNIST,
   0.651 on Fashion-MNIST, 0.116 on MNIST under a fixed pixel permutation against
   chance 0.063. Nothing learned here is free of the data; the two things a model
   can learn differ in *granularity*, not in whether they are memorised.
3. **Generalisation appears only where retrieval fails**, it is selected at
   training time, and the two solutions are separate attractors that
   inference-time context size does not move between.

## What we learned about running this kind of experiment

Separate from the findings, and more transferable. Most of these were bought by
getting something wrong first.

**Measure the model-free ceiling before training anything.** The single most
explanatory number in the project — the soft look-up scoring 1.002 at M=16 — cost
about a minute of GPU and was computed *after* the first two training runs. It
retrospectively explained both of them. Had it come first, the task would have
been designed differently. This is now Step 0 above.

**A derived metric puts a subtraction between the reader and the data.** `gain`
(= error absent − error present) was convenient and compact, and reporting the
two conditions side by side turned out to say the same thing while surviving
scrutiny better. Raw numbers also travel: the same pair works in a table, a
figure and a sentence without redefinition.

**The intuitive metric can be not just noisy but inverted.** Identification
accuracy ranked the *only* model still retrieving at M=256 dead last, behind two
models that never retrieve at all, because it rewards output quality and does not
ask how the output was reached. A confounded metric does not merely add noise; it
can return the exact opposite of the right answer.

**A chance level that moves will break a column.** Identification accuracy's
chance is 1/M, which falls 64-fold across the context sweep. One dashed line
labelled "chance at M=16" made a score of 82× chance read as a collapse.

**Never draw sample figures in file order.** The first six samples of the
absent-target condition scored 0.88, 0.67, 0.81, 1.04, 0.60, 0.59 against a
median of 1.03. Drawing them in order would have made the model look far better
than it is. Columns are now chosen at fixed percentiles of per-sample error and
labelled with their own scores.

**Compositing cuts both ways.** The model emits 784 pixels but is scored on 392,
so the visible half is unconstrained noise — showing the raw output made every
completion look broken. Pasting the true visible half back in fixes that and
*also* flatters, because a correct top half carries most of the percept. Both
fixes were needed: composite, plus a per-image error label and a
predict-the-average-digit row for reference.

**A small systematic anomaly deserves a model-free probe.** The absent-target
condition scoring slightly *better* than the answer-present one looked like a
broken harness. Ridge regression — which cannot see the context — reproduced it
exactly, proving it was a normaliser artefact. That probe also produced a
measured resolution floor (~0.02) which now licenses the claim that two numbers
are "the same".

**A claim that sounds like a deduction from the architecture is still empirical.**
"The mechanism is content-addressed, so it has nowhere to store a memorised
identity, so it generalises" survived several drafts because it sounds like it
follows from the design. It is false: the metric is fitted to MNIST and collapses
to 0.116 under a pixel permutation. Anything of that shape should be tested, not
reasoned to.

**Arbitrary data is an instrument, not a target.** No free lunch: a retriever
assuming nothing cannot beat one assuming something true. The permuted-pixel
score reports *which* assumption the model came to rely on; raising it would mean
discarding spatial structure, which is real. Pools that violate a true property
of the data must never be averaged with pools that satisfy it.

**Operational.** Deterministic seeding is worth the discipline — re-running exp1
and exp2 months-of-edits later reproduced them to four decimals, which is what
made re-running for a missing checkpoint safe. Checkpoint the number you intend
to quote: the completion ceiling is the minimum of a U-shaped curve, and no
checkpoint existed at that step, so it could be cited but never looked at. And
never rewrite `results.jsonl` while any job might append — the wait-for-idle
guard in `scripts/tmp/wave2.sh` exists because the first version of it raced.


## The binding constraint, and why it shapes everything below

The most consequential number in the project is a baseline, not a model result:

> At M = 16, the best possible soft look-up from the context scores **1.002** on
> an absent-target query — identical to ignoring the context entirely.

Sixteen unrelated digits do not tell you how to finish a seventeenth. So on this
task **no objective could reward context-based generalisation, because there is
none available**. Every "the model failed to use its context" result in the paper
is, at least partly, a statement about the task rather than about the model.

That is the thing to fix first. Broadening the data (§C) or blocking memorisation
(§A2) are both worth doing, but neither creates in-context signal that the task
does not contain.

### Step 0, from now on: measure the ceiling before training anything

Had we computed that 1.002 before running exp1, the task would have been designed
differently. Every new task variant below is gated on a model-free measurement
first — mean-image, ridge, 1-NN and soft look-up — which costs about a minute of
GPU and decides whether the variant is worth training on at all.

**Gate:** if the soft-look-up ceiling on the absent-target condition is not
meaningfully below 1.0, the variant does not get a training run.

---

## Phase A — two cheap corrections (about 1 GPU-hour total)

Both are small, both make everything after them cleaner, and one of them is a
real open question.

### A1. Shared queries between the present and absent conditions

Conditions B and D currently draw their query images with different seeds, so
their normalisers differ by up to 2% and normalised differences below ~0.02 are
uninterpretable. Fix: draw the Q queries first, then build one context containing
them and one without. Denominators become equal by construction.

*Cost:* a change to `lib/evalsets.py`, then re-score existing checkpoints — no
retraining. *Effect:* removes the noise floor; makes "these two numbers are the
same" exact rather than merely much larger than the noise.

### A2. Block memorisation and see what the recall objective does then

The paper's central mechanism claim is that recall training has two routes to a
lower loss, and that the in-weights route is a *memorisation* route because the
target is always a training image. If that is right, closing the memorisation
shortcut should change the outcome at M = 256.

Give the context pool unlimited fresh images — random shifts, small rotations,
elastic warps — so no image is ever seen twice. At M = 256 the model can then no
longer memorise its way to a low loss, and must either learn a transferable prior
or fail.

| outcome | what it means |
|---|---|
| learns a genuine prior (D improves on *novel* images, C ≈ D) | the recall objective **can** yield knowledge; memorisation was merely the cheaper route |
| fails outright (both C and D near 1.0) | the objective yields nothing without the shortcut; the paper's conclusion strengthens |

*Prediction:* failure, but weakly held — this is the most genuinely uncertain
experiment in the plan, which is why it is early. It is also the number the paper
explicitly says it lacks.

---

## Phase B — make the context informative (the main event)

The task must be one where the context actually constrains the answer. Two
designs, cheapest first. **Both are gated on the Step-0 measurement.**

### B1. Structured context: the context is *about* the query

Instead of 16 unrelated digits, draw the context so it carries information about
the target — for example, all context images of the same digit class as the
query, or a set of transformations of a small number of underlying images.

The soft-look-up ceiling should drop well below 1.0 by construction, because a
weighted average of sixteen 7s *is* a good prediction of the bottom of another 7.

*Measure first.* If the ceiling at M=16 does not fall below about 0.8, the design
is not doing its job and should be iterated before any training.

Then re-run the triad — recall / completion / mixed — and ask the paper's
question again in a regime where the answer can be yes: **does a recall objective
learn to use a context that is genuinely worth using?**

*Prediction:* recall training now produces real in-context generalisation,
because for the first time retrieval and generalisation are not mutually
exclusive — the nearest neighbour is *informative* rather than merely present.
If that holds, the paper's conclusion needs its scope narrowed from "retrieval
training buys no generalisation" to "…when the context contains nothing to
generalise from", which is a much more interesting and more defensible claim.

### B2. A rule to infer rather than a picture to finish

The stronger version, and more work: an episode presents (input, output) pairs
under some transformation — rotate, reflect, recolour, shift — and the query is a
new input under the same transformation. Completion cannot be done from a prior
at all; the rule is only available in the context.

This is the cleanest possible form of the original question, and it is where a
negative result would be most damaging to the "in-context learning is retrieval"
position. It is also a genuinely new task, so it is sequenced after B1.

---

## Phase C — broaden the metric (natural data, per the no-free-lunch argument)

The goal is *not* assumption-free retrieval, which buys nothing. It is a
similarity metric whose assumptions are the ones natural data actually satisfies
— broader than "MNIST spatial statistics", but not empty.

### C1. Train on a wider natural pool

Train the recall model on MNIST + Fashion-MNIST + KMNIST (and further natural
28×28 sets), then measure transfer.

* **Target:** the Fashion-MNIST number, currently 0.651, on a model that has not
  trained on it — i.e. hold one set out and measure transfer to it.
* **Instrument, not target:** the permutation probe, currently 0.116. It reports
  which regularity the metric came to depend on. Driving it up would mean
  discarding spatial structure, which is real information; a model that scored
  well there would be *worse*, not better.

### C2. Synthetic data as controlled structure

Synthetic generators are useful here for a specific reason: they let us dial one
regularity at a time — spatial locality, stroke continuity, object permanence
under transformation — and ask which ones the metric picks up. That is a
different use from "more data", and the deliverable is a map from *assumption
present in the data* to *transfer gained*, not a single headline number.

---

## Phase D — mechanism, once the above is settled

**D1. Locate the capacity crossover.** The two-route account says retrieval's
error floor rises with M while the in-weights floor is flat, so there is a
crossover. Sweep M at two or three values of `d_k`. If capacity sets the switch,
the crossover moves with `d_k`; if it sits at the same M regardless, something
else is doing the work. Note the account is already known to be incomplete —
exp17 runs `d_k = 8` with 32 heads at M = 16 and retrieval does not collapse, so
heads distribute items and the constant is not pinned down.

**D2. A harder dataset for the digit split.** Perfect retrieval and chance-level
completion on the same unseen digits is the cleanest separation in the project,
and it deserves a dataset where retrieval is not nearly free — Omniglot is the
obvious candidate and is already in `shared_lib.datasets`.

---

## What could overturn what is already published

Worth stating plainly, since the paper is out:

* **B1 succeeding would narrow the headline claim** from "retrieval training buys
  no generalisation" to "…on a task whose context contains no usable signal".
  That is a scope correction, not a reversal — every measured number stands — but
  it is a significant one and the paper would need rewriting around it.
* **A2 finding a genuine prior** would soften the mechanism story: memorisation
  would be the route taken rather than the only route available.
* **A1 will not change any conclusion** — every comparison the paper rests on is
  far larger than the 0.02 floor — but it will make the "same number" claims
  exact.

Nothing in the plan is expected to overturn the transfer results, which are
evaluation-only and rest on a model-free control (ridge).

---

## Sequencing and cost

| | what | GPU | gate |
|---|---|---|---|
| 1 | A1 shared queries | ~10 min (re-scoring only) | — |
| 2 | A2 memorisation blocked | ~1 h | — |
| 3 | B1 ceiling measurement | ~2 min | **decides whether 4 runs** |
| 4 | B1 triad in the new regime | ~1 h | ceiling < 0.8 |
| 5 | C1 wider natural pool | ~2 h | — |
| 6 | B2 rule-inference task | ~half a day incl. task code | — |
| 7 | D1, D2 | ~2 h | — |

Steps 1–4 are the ones that would change what the project says. 5–7 deepen it.

---

## Decisions I would like your call on

1. **Order.** I would do A1 → A2 → B1-gate → B1, because B1 is the one that can
   change the paper's claim and A2 is the one the paper admits it is missing.
   Your stated direction is the breadth programme (C), which shares no
   infrastructure with B and could equally run first — but if B1 succeeds it
   changes what C's numbers *mean*, so I would rather know first.
2. **B1's context design.** Same-class contexts are the cheapest way to make the
   context informative, but "same class" is a label-derived shortcut that MNIST
   happens to provide and a real setting would not. Transformations of a few base
   images are more principled and need a generator. Which do you want?
3. **Whether to hold the paper.** It is published at a stable URL and would need
   rewriting if B1 lands. The alternative is to mark §8 with a note that the
   scope correction is pending. I lean towards the note.
