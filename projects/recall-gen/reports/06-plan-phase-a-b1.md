# Report 6 — the noise floor removed, memorisation blocked, and a context worth using

Three plan steps in one sitting: A1 (shared queries), A2 (block memorisation),
and B1 (make the context informative), plus the model-free gate measurement that
B1 was conditioned on.

Rows: `*_sharedq` (15 re-scorings), `exp18`, `exp19`, `exp20`–`exp23`,
`baselines_M16_r14_{class,knn}*`.

Everything below is normalised MSE — model error over the error of predicting the
train-set mean image — on the four standard conditions. `D_novel_absent` is the
one that carries the project's question: complete a held-out image whose answer
is *not* in the context.

---

## A1 — the queries are now shared, and the floor was hiding something

The present and absent conditions used to draw their query images separately.
Every number is divided by `mse_mean`, which depends only on the queries, so the
two conditions were divided by different denominators — by up to **1.54%** on the
train pool and 0.20% on the held pool. That put a ~0.02 floor under every
present-vs-absent comparison.

Now a pool is drawn once. The absent context is the filler; the present context
is the same filler with Q randomly chosen slots overwritten by the queries. The
denominators are equal by construction and the measured spread is 0.0000%. Slots
are random rather than at the end because the state decays along the sequence, so
a fixed slot would confound presence with recency.

`scripts/rescore.py` re-scored every existing checkpoint — no retraining — into
`<exp>_sharedq` rows. No conclusion moved. But the floor was not merely noise:

| | present | absent | |
|---|---|---|---|
| exp2 completion-trained | 0.0488622 | 0.0488595 | five figures |
| exp9 completion, novel classes | 0.0861175 | 0.0861168 | five figures |
| exp9 completion, seen classes | 0.0477748 | 0.0477746 | five figures |
| exp13 recall→completion fine-tune | 0.031474 | 0.031673 | **0.6% apart** |

The three completion-trained models are not "the same within noise" — their
output is *invariant* to whether the answer is in the context, to five
significant figures. And exp13, the fine-tune, is resolvably **not** invariant: a
0.6% residual sensitivity that the old floor would have swallowed whole.

---

## A2 — blocking memorisation removes memorisation and creates nothing

Every training image is independently warped (random affine + elastic, see
`core.augment`), so the pool never repeats and nothing about a specific image is
worth storing in weights. The warp is applied *before* the target is chosen, so a
target-present query is still exactly a context image — recall stays exactly as
solvable, and only the pool's finiteness is removed. Evaluation is unchanged and
un-warped, so every number is comparable to the run it pairs with.

| | A seen+ | B nov+ | C seen− | D nov− | id(B) |
|---|---|---|---|---|---|
| M=16 finite pool (exp1) | 0.015 | 0.017 | 0.869 | 0.843 | 1.000 |
| M=16 **infinite** (exp19) | 0.025 | 0.024 | 1.027 | 1.004 | 1.000 |
| M=256 finite pool (exp5) | 0.133 | 0.569 | 0.134 | 0.572 | 0.456 |
| M=256 **infinite** (exp18) | 0.517 | 0.513 | 0.581 | 0.573 | 0.404 |
| ridge (ignores the context) | | | 0.638 | 0.649 | |

Two clean readings, neither of them the outcome the plan predicted.

**Memorisation was real, and it is gone.** exp5's signature was C=0.134 against
D=0.572 — a seen image completed six times better than an unseen one, which is
memorisation and nothing else. With the pool made infinite that gap vanishes
entirely: C=0.581, D=0.573. The seen/novel distinction stops existing, which is
exactly what the instrument was built to do.

**Closing the shortcut created no new ability.** D is 0.572 before and 0.573
after — unchanged to a thousandth. Whatever prior the M=256 model has (and it is
a real one: 0.573 against ridge's 0.649, so it beats the best context-free linear
inpainter), it had that prior *already*, with the shortcut available. Blocking
memorisation did not push the objective into learning anything it was not
learning anyway.

**At M=16 the effect runs the other way.** exp19's absent-target error rises to
1.004 — *worse than predicting the mean image*. At M=16 retrieval is so cheap
that the little completion ability exp1 had was memorisation residue, and
removing the shortcut removes it. Retrieval itself is untouched (id 1.000).

So the plan's table has a third branch it did not list: memorisation is the route
taken, closing it costs nothing at M=256 and costs the residue at M=16, and in
neither case does the recall objective start yielding knowledge.

---

## The B1 gate — measured before training anything

Step 0 of the plan: measure the model-free ceiling first. At M=16 with an
i.i.d. context the best possible soft look-up scores **1.002** on an absent
target — identical to ignoring the context. That is the binding constraint the
whole plan is organised around.

Two constructions that make the context *about* the query (M=16, Q=1, so each
query gets all 16 context slots):

| context | soft look-up (D) | ridge (D) | 1-NN (D) |
|---|---|---|---|
| i.i.d. | 1.002 | 0.645 | 1.575 |
| same class | 0.743 | 0.649 | 1.212 |
| **16 nearest neighbours** | **0.552** | 0.631 | 0.969 |

Only the nearest-neighbour construction puts the context ahead of ignoring the
context. Same-class clears the plan's written gate (< 0.8) but not the criterion
that actually matters — at 0.743 it is still worse than ridge, so an objective
could reach 0.743 without ever reading the context. Sixteen arbitrary 7s say
less about the bottom of this 7 than its sixteen nearest neighbours do.

The `knn_offset` dial works as designed, and gives informativeness as a
continuous knob rather than a binary:

| ranks skipped | 0 | 64 | 512 |
|---|---|---|---|
| soft look-up (D) | 0.552 | 0.701 | 0.886 |

---

## B1 — does a recall objective use a context that is worth using?

The triad, re-run in the new regime. **Yes, and then it throws it away.**

| | A seen+ | B nov+ | C seen− | D nov− | best D | id(B) |
|---|---|---|---|---|---|---|
| knn ctx, recall (exp20) | 0.022 | 0.036 | 0.589 | 0.666 | **0.505** | 0.988 |
| knn ctx, completion (exp21) | 0.107 | 0.610 | 0.069 | 0.611 | 0.463 | 0.266 |
| knn ctx, mixed (exp22) | 0.098 | 0.470 | 0.091 | 0.566 | 0.463 | 0.400 |
| class ctx, recall (exp23) | 0.025 | 0.027 | 0.747 | 0.761 | 0.595 | 1.000 |
| i.i.d. ctx, recall (exp1) | 0.015 | 0.017 | 0.869 | 0.843 | 0.635 | 1.000 |
| soft look-up ceiling, knn | | | | | 0.552 | |
| ridge, knn eval set | | | | | 0.631 | |

The comparison to make is exp20 against exp1: same objective, same architecture,
same schedule, same number of steps. **Only the context construction differs.**

* On an i.i.d. context the recall-trained model's best absent-target error is
  0.635 — indistinguishable from ridge (0.645), which cannot see the context.
  Nothing in that number came from the context, and nothing could have.
* On a nearest-neighbour context it reaches **0.505** at step 1000 — below ridge
  (0.631) *and* below the best pure soft look-up (0.552). That is in-context
  generalisation, from an objective that only ever asked it to retrieve.
* Retrieval is undamaged — id 0.988 on novel images, with fifteen *near
  duplicates* as distractors rather than fifteen unrelated digits.

### The 0.505 really is coming from the context

Beating ridge is not on its own evidence of context use: exp18 reaches 0.573 on a
task whose context is worthless, so a nonlinear prior can clear that bar. The
claim needed measuring, so `scripts/ctx_ablation.py` holds the model and the
queries fixed and replaces the context — with another episode's knn context
(same statistics, same near-duplicate structure, wrong query) and with an i.i.d.
one. Row `ctx_ablation`.

| | proper | swapped | i.i.d. | context is worth |
|---|---|---|---|---|
| exp20 at its best (step 1000) | **0.505** | 0.785 | 0.781 | 0.281 |
| exp20 at the end (step 12000) | 0.666 | 1.358 | 1.194 | 0.693 |
| exp23 at its best | 0.592 | 0.730 | 0.745 | 0.138 |

Settled: give exp20's best checkpoint the wrong neighbours and it scores 0.785,
worse than ridge. Its 0.505 is not a prior — it has a *bad* prior — it is 0.28 of
context, read from sixteen images that do not contain the answer. And the
same-class arm's smaller number comes with a proportionally smaller context
dependence (0.138), which is the gate's ceiling ordering showing up in the
trained models.

### What the degradation actually is

exp20 ends at 0.665 having peaked at 0.505, and the obvious reading — that
continued recall training throws the context away — is wrong. The ablation says
the opposite: the final model is *more* context-dependent than the best one
(0.693 against 0.281), and collapses to 1.358 without the right context, far
worse than the mean image.

So recall training does not stop reading the context. It narrows what it reads it
*for*: from "what do these sixteen similar images say about the answer" to "which
of these is the answer". The second is worth more under the training objective
and worth nothing when the target is absent. The U-shape the paper reports is
this specialisation, not a loss of context use.

The same-class arm confirms the gate's reading: exp23 reaches 0.595, better than
ridge but far short of the knn arm, in proportion to its weaker ceiling. The
construction that gives the model more to work with is the one it works with
more.

---

## What this does to the published claim

The paper says retrieval training buys no generalisation. On the evidence here
that needs its scope narrowed, which is what §8 flagged as pending:

> Retrieval training buys no generalisation **when the context contains nothing
> to generalise from**. When the context does contain something, retrieval
> training finds it — and then specialises away from it, narrowing context use
> from prediction to identification.

Every measured number in the paper stands. What changes is that the headline is
now known to be a statement about the task as much as about the objective, and
the second clause is a new and more interesting finding than the first.

## What A2 does to the mechanism story

The paper's two-route account survives intact but gets sharper. Memorisation is
demonstrably the route taken at M=256 — remove the shortcut and the seen/novel
gap disappears completely. But it is not what was *blocking* generalisation:
removing it left D unchanged at 0.573. The reason the M=256 model does not
generalise better is not that memorisation was easier; it is that on an i.i.d.
context there was nothing to generalise from. That is the same constraint B1
lifted, and B1 is where the movement happened.
