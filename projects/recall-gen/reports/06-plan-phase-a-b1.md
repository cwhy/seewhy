# Report 6 — recall training generalises once the context is worth reading

Recall training — an objective that only ever asks the model to find and copy
one of its context images — produces genuine in-context generalisation once the
context actually constrains the answer. On a nearest-neighbour context (exp20),
the best absent-target error reached is **0.505**: below context-blind ridge
regression (0.631) and below the best achievable pure soft look-up from that
same context (0.552). On the original i.i.d. context, the identical objective,
architecture and schedule (exp1) reaches only **0.635** — indistinguishable from
ridge. Only the context construction differs between the two runs.

![exp20 against its reference points](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_exp20_curve_vs_refs.svg)

That is the headline result the project was built to find. It forces a scope
correction on the published paper (last section below) and it comes with a
second, independent finding: continued recall training does not stop reading the
context — it **narrows what it reads it for**, from predicting the answer to
identifying it (own section below).

Rows: `exp20`–`exp23`, `baselines_M16_r14_knn*`, `baselines_M16_r14_class*`,
`ctx_ablation`. All context sizes below are M=16, Q=1. `exp1` (i.i.d. arm) is
cited once, for its D-curve minimum only — see the note on provenance where it
appears.

A methods fix (A1, shared queries between the present/absent conditions) and a
memorisation-blocking experiment (A2) were run alongside this but answer
different questions; A1 is noted below only where it affects how a number here
should be read, and A2 is left for its own report.

---

## The model-free gate — measured before any training

Step 0 of the plan: before training anything, measure what the context alone
is worth. At M=16 with the *original* i.i.d. context, the best possible soft
look-up scores 1.002 on an absent target — identical to ignoring the context.
Two constructions make the context *about* the query instead:

| context | soft look-up (D) | ridge (D) |
|---|---|---|
| i.i.d. (original) | 1.002 | 0.645 |
| same class | 0.743 | 0.649 |
| **16 nearest neighbours** | **0.552** | 0.631 |

Only the nearest-neighbour construction puts the context ahead of ignoring it.
Same-class passes a naive "< 0.8" gate but is still worse than ridge — an
objective could reach 0.743 without ever reading the context. `knn_offset`
turns this into a dial rather than a binary, by skipping ranks before taking the
16 nearest:

![the knn_offset dial](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_knn_offset_dial.svg)

| ranks skipped | 0 | 64 | 512 |
|---|---|---|---|
| soft look-up (D) | 0.552 | 0.701 | 0.886 |

Two example contexts for the *same query*, held fixed, at rank-0 (the
construction used below). The query is shown with its bottom half greyed —
that half is what the model never sees, shown for legibility rather than as
the true pixels the model would be handed. Episode 0 of the fixed-seed draw;
this figure illustrates a construction, not model output, so no percentile
selection applies:

![one query, two context constructions](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_knn_vs_iid_context.png)

---

## The triad, in the regime the context is worth reading

M=16, Q=1. `best D` is the minimum of the D-curve over training (`history.nmse`);
`id(B)` is identification accuracy on novel, target-present episodes.

| run | A seen+ | B nov+ | C seen− | D nov− | best D | id(B) |
|---|---|---|---|---|---|---|
| knn ctx, recall (exp20) | 0.022 | 0.036 | 0.589 | 0.666 | **0.505** | 0.988 |
| knn ctx, completion (exp21) | 0.107 | 0.610 | 0.069 | 0.611 | 0.463 | 0.266 |
| knn ctx, mixed (exp22) | 0.098 | 0.470 | 0.091 | 0.566 | 0.463 | 0.400 |
| class ctx, recall (exp23) | 0.025 | 0.027 | 0.747 | 0.761 | 0.595 | 1.000 |
| — | | | | | soft look-up ceiling, knn: **0.552** | |
| — | | | | | ridge, knn eval set: **0.631** | |

exp20 against exp1 is the comparison that isolates the cause: same objective,
same architecture, same schedule, same number of steps — **only the context
construction differs.**

* On an i.i.d. context (exp1), the recall-trained model's best absent-target
  error is **0.635** — the minimum of exp1's own D-curve, indistinguishable from
  ridge (0.645 on the i.i.d. eval set). Nothing in that number came from the
  context, and nothing could have: the gate above already showed the i.i.d.
  ceiling is 1.002.
* On a nearest-neighbour context (exp20), it reaches **0.505** — below ridge
  (0.631) *and* below the best pure soft look-up (0.552). That is in-context
  generalisation, from an objective that only ever asked it to retrieve.
* Retrieval is undamaged: id(B) = 0.988 on novel images, with fifteen *near
  duplicates* as distractors rather than fifteen unrelated digits.

*Provenance note:* exp1's 0.635 is the minimum of the D-curve in the original
`exp1` row's `history` (step 500). It is **not** taken from `exp1_sharedq` —
that row was re-scored with A1's shared-query fix and carries no `history`, so
it cannot supply a best-over-training number. The two rows measure the same
quantity at different points and must not be quoted from the same cell.

### The 0.505 really is coming from the context

Beating ridge is not on its own evidence of context use — a large enough memory
can beat ridge on a task whose context is worthless (see the paper's M=256
result). The claim needed a direct measurement: `scripts/ctx_ablation.py` holds
the model and the query fixed and swaps the context — for another episode's knn
context (same statistics, same near-duplicate structure, wrong query) and for an
i.i.d. one.

![context ablation: proper, swapped, and i.i.d. context](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_ctx_ablation.svg)

| | proper | swapped | i.i.d. | context is worth |
|---|---|---|---|---|
| exp20 at its best (step 1000) | **0.505** | 0.785 | 0.781 | 0.281 |
| exp23 at its best (step 500) | **0.592** | 0.730 | 0.745 | 0.138 |

Give exp20's best checkpoint the wrong neighbours and it scores 0.785 — worse
than ridge. Its 0.505 is not a general prior it happens to also have; it is
0.281 of context, read from sixteen images that do not contain the answer. The
same-class arm's smaller number comes with a proportionally smaller context
dependence (0.138), matching the gate's ceiling ordering.

The same swap, in pixels, on exp20's *final* checkpoint (the D=0.666 model,
proper/swapped/i.i.d. = 0.666/1.358/1.194 above):

![exp20 final, three contexts, in pixels](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_completion_ablation.png)

Columns are the same six episodes as the figure below, chosen once, model-free,
at fixed percentiles (p5…p95) of the soft look-up baseline's per-sample error on
the proper knn context — not by any trained model's error, so the choice cannot
flatter whichever row picked them. Every shown image is a **composite**: the
model only ever predicts the hidden (bottom) half, so the true visible half is
pasted back before display; the number in the corner of each panel is that
panel's own hidden-pixel MSE, normalised the same way as the table (divide by
`mse_mean` to compare against 1.0) — and each is a **single sample**, so it can
and does invert the aggregate (p95 here: swapped scores 0.07 against proper's
0.14). The claim is the aggregate over all 512 episodes (0.666/1.358/1.194
above), not any individual panel. Under the swapped and i.i.d. contexts the
completions visibly stop tracking the query — the predicted hidden halves stop resembling the query's digit and instead show overlapping, double-exposed strokes from unrelated digits, and every column's error roughly doubles or worse (0.01–0.14 proper vs. 0.05–0.11 swapped and 0.03–0.14 i.i.d. on these six). The pixels confirm the aggregate collapse.

---

## Continued training does not stop reading the context — it narrows what it reads it for

exp20 ends training at D=0.666 having peaked at 0.505, and the obvious reading —
that continued recall training throws the context away — is wrong. The same
ablation, run on the final checkpoint instead of the best one, says the
opposite:

| | proper | swapped | i.i.d. | context is worth |
|---|---|---|---|---|
| exp20 at its best (step 1000) | 0.505 | 0.785 | 0.781 | **0.281** |
| exp20 at the end (step 12000) | 0.666 | 1.358 | 1.194 | **0.693** |

The final model is *more* context-dependent than the best one — 0.693 against
0.281 — and it collapses to 1.358, far worse than predicting the mean image,
when handed the wrong context.

So recall training does not disengage from the context as training continues.
It narrows what it reads the context *for*: from "what do these sixteen similar
images say about the answer" (worth something when the target is absent) to
"which of these sixteen is the answer" (worth nothing when it is). The second
is what the training objective rewards, so it is what gradient descent keeps
sharpening, and the D-curve's rise after step 1000 is that specialisation, not
a loss of context use.

The claim predicts a specific pixel-level contrast: the best checkpoint should
look like a plausible blend consistent with the visible half (it has never been
rewarded for picking one specific neighbour when the target is absent), and the
final checkpoint should look like a sharp, confident, and *wrong* specific
digit (it has been rewarded for exactly that on every present-target episode,
and cannot tell present from absent from its input alone).

![the specialisation claim, in pixels: mean image, soft look-up, exp20 best, exp20 final](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_completion_specialisation.png)

Same six episodes as the ablation figure above, same construction: columns are
fixed percentiles (p5…p95) of the soft look-up baseline's per-sample error
(model-free — the ranking uses no trained model), every shown image composites
the true visible half back onto the model's predicted hidden half, and each
panel is labelled with its own hidden-pixel error. The mean-image row is the
"no information" reference the compositing rule requires. **the pixels do not show the "confidently wrong digit" contrast the prose above predicted** — at no column does exp20 final commit to a digit shape other than the true one. But a real, smaller effect is visible on closer inspection (crop p41 and p77 to see it at full resolution): exp20 best's strokes are noticeably **blurrier and double-edged** — p41's 3 shows a faint second loop ghosted behind the first, p77's 0 has a soft, smeared bottom arc — while exp20 final's strokes are **sharper and single-edged**, higher-contrast, no ghosting. That is consistent with the specialisation story in direction — committing to one neighbour rather than blending several — even though it is not the specific wrong-digit failure predicted, and it is a matter of stroke sharpness, not digit identity: at these six columns four of six panel errors are within 0.02 of each other, and the one clear gap is the hardest column (p95: best 0.07, final 0.14). **The prose claim above is weakened accordingly**: the specialisation is established by the ablation and the D-curve; this figure adds a modest, real corroborating detail (sharper-but-not-wrong) rather than the strong contrast originally predicted.

---

## What this changes

**The published paper's headline needs its scope narrowed.** It currently reads
as "retrieval training buys no generalisation." The evidence above narrows that
to:

> Retrieval training buys no generalisation **when the context contains nothing
> to generalise from**. When it does, retrieval training finds it — and then
> specialises away from it, narrowing context use from prediction to
> identification.

`paper/sections/08-limitations.typ` currently marks this correction as
*pending*. It is now owed: the B1 gate and triad are the measurement that
section was waiting on, and the direction of the result is the one the plan
predicted. Every number already in the paper stands; what changes is that the
i.i.d.-context finding is now understood to be a statement about that task as
much as about the objective, and the specialisation clause is a new, sharper
finding that the paper does not yet contain.

## Sources

`results.jsonl` rows `exp20`–`exp23`, `baselines_M16_r14_knn*`,
`baselines_M16_r14_class*`, `ctx_ablation`, and `exp1` (cited once, for its
D-curve minimum). Context construction is `lib/evalsets.py`
(`ctx_mode="knn"|"class"|"iid"`); the ablation is `scripts/ctx_ablation.py`.
