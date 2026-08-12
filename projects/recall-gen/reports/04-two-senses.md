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

## What "trained to complete" is, and what it actually learned

The 0.458 bar deserves unpacking, because it is the reference everything else is
measured against and it is not quite what its name suggests.

It is **the same model, the same data, the same loss**. One thing changes: which
image the query asks about.

| arm | the query's true image is… | so the task is… |
|---|---|---|
| recall (exp1) | always one of the 16 context images | find it and copy it |
| completion (exp2) | drawn fresh from the pool, essentially never in the context | there is nothing to find — fill the hole from what you know about digits |

Here is that run over training:

![the completion arm's training curve](https://media.tanh.xyz/seewhy/26-08-12/recallgen_completion_curve.png)

Colour is the image pool; line style is whether the answer was in the context.
Two things to read off it.

**The pairs coincide.** A lies exactly on C, and B exactly on D, for the whole of
training. Having the answer sitting in the context is worth *nothing* to this
model — target presence is not a variable it responds to. It never learned to use
the context at all.

**The two pairs separate, and the novel one turns around.** The training-pool
pair keeps falling to 0.041. The unseen-image pair bottoms out at **0.458 at step
1000** and then climbs to 0.672. That is memorisation: the model is not learning
what digits look like, it is learning what these 60 000 specific digits look
like, and the better it gets at that the worse it does on anything else.

So 0.458 is an early-stopped number, and it measures *what a good digit prior in
the weights buys you* — not what using the context buys you. It is the right
ceiling for the comparison, and it comes with that caveat attached.

## What each arm actually draws

Three trained models, the same five query images, none of which is in any of
their contexts:

![what each arm draws on the same queries](https://media.tanh.xyz/seewhy/26-08-12/recallgen_arms_compare.png)

The failure modes are visibly different:

* **recall-trained, 16 in context** (aggregate 0.852) — fragmented and noisy. The
  read is selective, nothing in the state matches, and what comes back is
  incoherent. Look at the last column: the truth is a 0 and the output is a
  broken ring with debris in it.
* **recall-trained, 256 in context** (aggregate 0.561) — clean, confident,
  plausible digits that are the *wrong* digit. Truth 9, drawn 4. This is a model
  with a prior, drawing from it.
* **trained to complete, best step** (aggregate 0.458) — smooth and slightly
  blurry. It hedges, which is what MSE rewards; the last column is a soft blob
  rather than a committed ring.

Per-column numbers are for those columns only. They are chosen at percentiles of
the *average* difficulty across the three arms so the selection cannot favour
one of them, but they are five images, not the aggregate — the aggregates over
512 episodes are 0.852, 0.561 and 0.458.

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

## Why does a longer context produce memorisation? — a hypothesis

This is an explanation, not a measurement. It fits everything observed, it makes
predictions that have not been tested yet, and it is stated here so those
predictions can be checked rather than quietly abandoned.

**There are two routes to a lower recall loss, and they are always both open.**

/ Route A — in-context retrieval. Write the target into the state, read it back.
/ Route B — in-weights prediction. Predict the target's hidden half from its
  visible half, using knowledge stored in the weights.

Gradient descent takes whichever is cheaper to improve locally. Nothing in the
objective expresses a preference between them.

**Route B, on this task, is a memorisation route — not a knowledge route.** The
recall target is always one of the context images, and the context is always
drawn from the training pool, so *the target is always a training image*. "Learn
to predict this specific image's bottom half from its top half" is a 60 000-entry
lookup table, and at 12 million query samples it is very learnable. A general
prior about digits would be strictly worse at the training objective than the
lookup table, so it is not what gets selected. This is why route B shows up as
memorisation rather than as understanding — and it is measured, not assumed: the
M = 256 model scores 0.134 on training-pool images against 0.561 on novel ones. A
general prior would give the same number twice.

**Route A's floor rises with context size; route B's does not.** Retrieval
requires the state to keep M items apart. The keys that address the memory are
$d_k = 64$ numbers per head, and the delta rule's write of one item partially
overwrites another exactly to the extent their keys overlap. Sixteen items in a
64-dimensional key space can be made near-orthogonal easily; 256 cannot, at any
setting of the weights. So route A's achievable error is a rising function of M,
while memorising 60 000 images is the same job whatever M is — route B's floor is
flat.

**The switch is therefore a crossover, and it should be sharp.** Below it,
retrieval is far better than any prior could be (0.015 against a best-possible
0.458), so route A wins outright and route B is never developed. Above it, route
B wins and route A stops receiving gradient — and because both routes drive the
same output head, the retrieval circuit is not merely unused but actively
degraded. That is the M = 256 model: gain 0.004, indistinguishable from a model
that was never asked to retrieve.

### What this does *not* yet explain

If key dimensionality were the whole story, the crossover would sit near
$M approx d_k$. It does not straightforwardly: exp17 runs at $d_k = 8$ with 32
heads and M = 16 — twice the per-head key dimension — and retrieval does not
collapse there at all (identification accuracy 1.000, gain 0.646). Multiple heads
evidently distribute items across several small key spaces, so the effective
capacity is well above $d_k$. The hypothesis survives in its qualitative form
(capacity, not information) but the constant is not pinned down.

### Two tests that would settle it

1. **Move the crossover.** Sweep M at a small $d_k$. If capacity sets the switch,
   the collapse should arrive at a proportionally smaller M. If it arrives at the
   same M regardless, something else is doing the work.
2. **Block memorisation.** Give the context pool unlimited fresh images (random
   shifts and elastic warps, so no image ever repeats). Route B can then no
   longer be taken by memorising. At M = 256 the model must either learn a
   genuine, transferable prior — or fail outright. **Which of those happens is
   the most informative number missing from this project**, because it asks
   whether the recall objective can be made to produce knowledge when the
   memorisation shortcut is closed off.

## Next: train short, test long

A prediction worth recording before it is run, because it separates two readings
of the whole result and it costs nothing — the architecture has no
length-dependent parameters, so a model trained at one context size can simply be
evaluated at another with no retraining at all.

The question: **is the improvement at M = 256 a property of large contexts at
inference time, or an artefact of what training selects for?**

| | what happens | what it would mean |
|---|---|---|
| **Predicted** | Train at M = 16, test at M = 256: retrieval degrades (the state is overloaded) and completion stays bad, near 0.85 — it has no prior to fall back on. Train at M = 256, test at M = 16: gain stays near 0, retrieval does not come back. | The M = 256 result is a *training-time* selection effect. The two solutions are separate attractors, and a big context at inference buys nothing on its own. |
| **The alternative** | Train at M = 16, test at M = 256: completion improves toward 0.56 anyway. | Large contexts do carry usable signal at inference and the model can exploit it — which would substantially soften the paper's conclusion. |

The predicted outcome follows directly from the hypothesis above: the model
trained at M = 16 took route A and never built route B, so there is nothing for
a larger context to unlock. The alternative is what you would expect if the
context were genuinely informative and the M-sweep had merely been teaching the
model to read it.

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
