# Report 8 — retrieval and identification are one operation at two temperatures

Report 7 used identification accuracy as the definition of "retrieval" and
concluded that degradation appears when retrieval is achievable — exp24
(id(B) 0.295) was read as having "no retrieval capability at all". That framing
is wrong, and this report replaces it. Retrieval is not a separate ability that
is present or absent; it is one softmax-over-context operation with a sharpness
knob, and identification accuracy only asks whether the kernel is sharp enough
to land on the exact item. The model-free soft look-up on the knn baseline shows
the conflict with no model in it at all: sharpening from tau=0.03 to tau=0.003
buys **0.359** on the objective training actually optimises (target present,
0.372 → 0.013, `baselines_M16_r14_knn_Q1`) and **costs 0.119** on the
generalisation objective (target absent, 0.553 → 0.672, same run). Retrieving
similar things and identifying the exact one are the same operation at two
temperatures; the training objective only ever pays for the sharp end.

![the temperature sweep: present vs absent error against tau, knn and i.i.d.](https://media.tanh.xyz/seewhy/26-08-18/recall-gen_tau_sweep.svg)

This does not overturn report 7's headline number — exp24's frozen mixer at
0.471 (`D`, novel, target absent) is still the project's best generaliser, below
the soft look-up ceiling (0.552) and ridge (0.631). Only the explanation for
*why* changes.

Rows: `baselines_M16_r14_knn_Q1`, `baselines_M16_r14`, `effective_tau2`,
`effective_tau`, cited against `exp1`, `exp20`, `exp24`, `exp26`, `exp27`,
`exp28` from reports 6 and 7. One question: what does a trained kernel do that
a frozen one cannot, and is achieved identification the right way to describe it?

---

## The mechanism, measured in trained checkpoints

Fitting each checkpoint's *effective temperature* — the tau whose model-free
soft look-up best reproduces the model's actual output, on the target-present
condition where sharpness is exercised (method below) — shows the same
sharpen-and-degrade trade directly in trained models:

| checkpoint | context | tau*(present) | B (target present) | D (target absent) |
|---|---|---|---|---|
| exp20 best (step 1000) | knn | 0.03 | 0.439 | 0.505 |
| exp20 final (step 12000) | knn | **0.00053** | 0.036 | 0.666 |
| exp1 final (trained) | iid | 0.0017 | 0.017 | 0.843 |
| exp24 best (frozen) | knn | 0.03 | 0.437 | 0.471 |
| exp24 final (frozen) | knn | 0.03 | 0.439 | 0.474 |
| exp26 final (frozen) | iid | 0.03 | 0.150 | 0.778 |
| exp27 final (frozen, 64) | knn+64 | 0.03 | 0.428 | 0.509 |
| exp28 final (frozen, 512) | knn+512 | 0.03 | 0.355 | 0.616 |

(exp1's D is its original `exp1` row's `history` final value, 0.843 — not the
`final` field's 0.852, and not the A1 re-scored `exp1_sharedq` row, also 0.843
but a different protocol; report 6 mixed these once already, so here only one is
used and named.)

![tau*(present) against D, trained vs frozen, exp20's arrow](https://media.tanh.xyz/seewhy/26-08-18/recall-gen_tau_vs_d.svg)

exp20 sharpens 57-fold between its best checkpoint and its last (tau 0.03 →
0.00053) and its D rises with it, 0.505 → 0.666, over the same interval. exp1
sharpens further still (tau 0.0017) and degrades further still (D 0.843). Every
frozen checkpoint sits at tau*=0.03 — none of them move, because there is
nothing in a frozen mixer for training to sharpen. That is the mechanism claim,
measured rather than asserted: training buys sharpness, and sharpness is what
degradation tracks in every trained run measured here.

## The retrievability dial: identification without training

exp24, exp27 and exp28 are the same frozen architecture (0.60M of 4.03M
parameters trained, mixer at random init) on the same knn context, differing
only in how far the nearest-neighbour distractors are from the query — 0, 64,
512 ranks out. None of them sharpen; tau*(present) stays at 0.03 for all three.
Identification accuracy still climbs:

| ranks skipped | 0 (exp24) | 64 (exp27) | 512 (exp28) | i.i.d. (exp26) |
|---|---|---|---|---|
| id(B) | 0.295 | 0.527 | 0.889 | 0.988 |
| degradation (final − best D) | +0.002 | +0.007 | +0.094 | +0.208 |
| best D | 0.471 | 0.502 | 0.522 | 0.570 |

![id(B) and degradation against ranks skipped, exp26 marked](https://media.tanh.xyz/seewhy/26-08-18/recall-gen_dial.svg)

So effective sharpness has two independent sources: training lowers tau (the
previous section), and context geometry spreads the distances a fixed-tau
softmax has to separate — when one item is much closer than the rest, even a
soft kernel concentrates on it by itself. exp24 → exp27 → exp28 shows geometry
raising identification with the kernel held fixed. Degradation rises across the
same three points too (+0.002 → +0.007 → +0.094), which is consistent with the
same "reachable retrieval invites drift" account from report 7 — except none of
these three ever trained a sharper kernel, so whatever is drifting toward it
here is the embedding/head, not the mixer's temperature.

## The complication: exp26 breaks the simple version of the claim

exp26 is frozen, its kernel never sharpens (tau*=0.03 throughout, identical to
exp24/27/28), and it still degrades 0.570 → 0.778 — the largest degradation of
any frozen run, larger than exp28's despite exp28 reaching a comparable id(B)
(0.889 vs. exp26's 0.988). Geometry explains exp26's high identification
(i.i.d. distances are well separated, so a soft kernel at tau=0.03 still picks
the right item), but geometry alone does not obviously predict *this much*
degradation from a kernel that never moves.

**Sharpening is a route to degradation, demonstrated cleanly in exp20 and exp1;
it is not the only one.** A plausible candidate for exp26's route, untested
here: the readout (embedding + head, the only trainable part) specialising to
emit whatever the fixed kernel already retrieves, which is a change in the
trained head rather than in the kernel's temperature. This is an open mechanism
question, not a resolved one — flagged here rather than folded into the tau
account it does not fit.

## Methods note: fitting temperature on the wrong condition gives nonsense

The first attempt (`effective_tau`, and the `D_novel_absent` column kept
alongside the valid fit in `effective_tau2`) fitted tau on the absent-target
condition and produced numbers that do not track sharpness at all: exp1 fitted
tau*=0.03 despite id(B) 1.000 (should be very sharp), exp27 fitted tau*=5.33,
exp28 fitted tau*=0.00095. When no context item is close to the query — the
absent-target condition, by construction — the softmax is diffuse at any
temperature, so that fit recovers the distance distribution of the distractors,
not the model's sharpness. The present-condition fit (used throughout this
report) is the valid one because that is the only condition where sharpness is
actually exercised. Both are stored in `effective_tau2`; this note is why the
present-condition numbers above are trustworthy and the absent-condition ones
are not used as evidence.

---

## What this changes

**Report 7's headline number is unchanged: exp24's frozen mixer at 0.471 is
still the project's best generaliser.** What changes is the explanation.
"Degradation appears when retrieval is achievable" implied retrieval is a
capability a model either has or lacks. The corrected account: retrieval is one
operation, degradation tracks how sharp it gets, and sharpness has two
independent sources — training (exp20, exp1) and context geometry (the dial).
Report 7's framing conflated "identification accuracy is low" with "no
retrieval", which mislabelled exp24 as retrieval-free when it is better
described as retrieval running at a temperature too high to identify one item —
the same operation the soft look-up baseline uses at tau=0.03.

**For the paper:** the two-attractor account becomes one mechanism — a
retrieval kernel with a temperature the training objective pays to lower — which
is a simpler and stronger claim than two separate modes. The caveat to carry
into the writeup is exp26: it shows a second degradation route, present even
with the kernel frozen, that is not yet identified. It should be named as an
open question, not folded into the temperature account.

**For the next run:** isolate exp26's route directly — freeze the head as well
(exp25's construction) but on the i.i.d. context, to see whether degradation
survives with *no* trainable readout at all. If it does not, the untested
candidate above (head specialisation) is confirmed; if it does, the second route
is something else again.

## Sources

`results.jsonl` rows `baselines_M16_r14_knn_Q1`, `baselines_M16_r14`,
`effective_tau2`, `effective_tau`, cited against `exp1`, `exp20`, `exp24`,
`exp26`, `exp27`, `exp28` from reports 6 and 7. Temperature fitting is
`scripts/effective_tau.py`; the dial runs are `exp24`/`exp27`/`exp28`, produced
by the knn-offset parameter in `lib/train.py` / `lib/evalsets.py`.
