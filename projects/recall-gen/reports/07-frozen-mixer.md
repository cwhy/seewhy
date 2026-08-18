# Report 7 — the project's best generaliser has a random mixer

exp24 freezes all four KDA layers at their random initialisation and trains only
the embedding and output head — 0.60M of 4.03M parameters. On the
nearest-neighbour context it reaches **0.471** absent-target error (`D`, novel,
target absent) — the lowest in the project, below the soft-look-up ceiling
(**0.552**) and below context-blind ridge (**0.631**) — and it gets there
**monotonically**: its best value is essentially its last checkpoint (0.471 at
step 11500, 0.474 at step 12000), where the fully-trained exp20 peaks at 0.505 by
step 1000 and then decays to 0.666.

![exp20 vs exp24 (knn) and exp1 vs exp26 (iid)](https://media.tanh.xyz/seewhy/26-08-18/recall-gen_frozen_curves.svg)

Training the sequence-processing stack is not what produces this project's best
result. But the obvious conclusion — "no trainable mixer, so no overfitting" — is
wrong, and the second half of this report is why.

Rows: `exp24`, `exp25`, `exp26`, `ctx_ablation2`, cited against `exp1`, `exp20`
and the M=16 knn/iid gates from report 6. This report answers one question: what
does training the KDA mixer actually buy? It does not cover A2
(memorisation-blocking, exp18/19) — different question, own report.

---

## The frozen mixer is not immune to degradation

exp26 is the control: same freeze (KDA layers at random init, only embedding +
head trained, 0.60M of 4.03M parameters), same architecture, same schedule, but
on the *original i.i.d. context* instead of knn. It degrades exactly like the
fully-trained i.i.d. run does — best D 0.570 (step 1000) to final D 0.778 (step
12000), against exp1's fully-trained best D 0.635 to final D 0.852 (i.i.d. eval
set: soft look-up ceiling 1.002, ridge 0.645).

So an untrained mixer is perfectly capable of the drift that hurts exp1. The
embedding and head alone are enough to do it. "No trainable weights, no
overfitting" does not fit the data — the frozen embedding+head drifts just as
badly as the fully-trained model when the context is i.i.d.

## What actually predicts degradation: achieved retrieval

What fits all five runs (exp1, exp20, exp24, exp25, exp26) is that degradation
tracks how well the model *can* retrieve, not which parameters are trainable:

![degradation vs identification accuracy, all runs](https://media.tanh.xyz/seewhy/26-08-18/recall-gen_degradation_vs_retrieval.svg)

* exp1 (trained, iid): id(B) 1.000 — near-perfect retrieval — degrades 0.852 −
  0.635 = 0.217.
* exp20 (trained, knn): id(B) 0.988 — degrades 0.666 − 0.505 = 0.161.
* exp26 (frozen, iid): id(B) 0.988 — degrades 0.778 − 0.570 = 0.208.
* exp24 (frozen, knn): id(B) **0.295** — cannot retrieve — degrades 0.474 − 0.471
  ≈ **0** (improves, if anything).
* exp25 (frozen incl. head, knn): id(B) **0.051** — retrieval is essentially
  gone along with everything else (D stuck at 2.97, far past the mean-image
  reference of 1.0) — degrades 2.974 − 2.974 ≈ 0. It sits at the same
  no-retrieval, no-degradation corner as exp24, for a different reason: it never
  learns to do anything at all (see below).

exp24's distractors are the query's own 16 nearest neighbours — retrieving one
specific neighbour is not a well-posed target, since several are nearly the
query itself. Its id(B) of 0.295 reflects that, not model weakness: it never has
anything to retrieve into. exp26's distractors are unrelated i.i.d. images, so
even with random mixer weights the embedding+head pair finds the one that
matches — id(B) 0.988 — and once retrieval is reachable, continued training
drifts toward it and generalisation decays, exactly as it does for the
fully-trained i.i.d. model.

**What learned mixer weights buy is discriminating near-duplicates, not
retrieval as such.** exp24 does not "have no retrieval circuit" in general — it
simply has nothing to discriminate, because every context item is already close
to the query. The random KDA layers pass along enough of the query's identity
that embedding+head can retrieve from *unrelated* distractors (exp26) but not
from *near-duplicate* ones (exp24). Degradation appears precisely where that
retrieval is achievable.

## exp24 is reading its context, not falling back on a prior

Beating ridge is not on its own evidence of context use (report 6). The same
`ctx_ablation` control run on exp24 (`ctx_ablation2`) settles it:

![context ablation: exp20 vs exp24, best and final](https://media.tanh.xyz/seewhy/26-08-18/recall-gen_frozen_ablation.svg)

| | proper | swapped | i.i.d. | context is worth |
|---|---|---|---|---|
| exp20 best (step 1000) | 0.505 | 0.785 | 0.781 | 0.281 |
| exp20 final (step 12000) | 0.666 | 1.358 | 1.194 | 0.693 |
| exp24 best (step 11500) | **0.471** | 0.763 | 0.694 | **0.292** |
| exp24 final (step 12000) | **0.474** | 0.764 | 0.694 | **0.290** |

exp24 gives up more from a context swap (0.292) than exp20 ever managed at its
best checkpoint (0.281) — with a random mixer, 0.60M trained parameters, and no
retrieval capability at all. 0.471 is not a context-free inpainting prior; it is
read from context more than the fully-trained model's best result ever was, and
that dependence does not decay as training continues (0.292 → 0.290), unlike
exp20's collapse from 0.281 to 0.693.

In pixels, at fixed model-free percentiles of the soft look-up's per-sample
error (same construction as report 6 — the ranking uses no trained model, and
every panel composites the true visible half back over the predicted hidden
half, labelled with its own hidden-pixel error):

![exp20 vs exp24 completions, same episodes](https://media.tanh.xyz/seewhy/26-08-18/recall-gen_frozen_completion.png)

exp24's completions (best and final are visually near-identical, matching their close D scores of 0.471/0.474) are cleaner and closer to the true target than exp20 final's at every one of the six columns (per-panel errors 0.01/0.02/0.02/0.03/0.03/0.06 vs. exp20 final's 0.01/0.03/0.05/0.04/0.05/0.14). The gap is clearest at the hardest column, p95: exp20 final shows visible doubling — a second, ghosted stroke behind the main one — where exp24 does not. That is not the "confidently wrong digit" failure mode; both models complete the right digit shape. It is a direct, visual confirmation of the aggregate gap (0.666 vs. 0.471/0.474): exp24 produces a cleaner completion at every sampled difficulty.

## exp25 pins the boundary

exp25 goes one step further and also freezes the head, training only the input
embedding (`W_pix`, `W_msk`, `role`). It fails outright: 2.869 (C) / 2.974 (D),
against exp24's 0.427 / 0.474, and id(B) collapses to 0.051. A fixed random
readout cannot be compensated for by the embedding alone — the trainable head is
load-bearing. "The embedding alone is enough" is too strong; embedding *and*
head, with the mixer frozen, is what reaches 0.471.

## What is still open

**Parameter count is not separated from mixer-freezing.** exp24 trains 0.60M
parameters against exp20's 4.03M — 16× fewer. exp26 makes the parameter-count
story unlikely as the sole explanation (same 0.60M budget, still degrades on
iid), but it does not close it: a fully-trained model at matched trainable count
(a small mixer, trained, on the knn context) is the control that would. That run
has not been done.

---

## What this changes

**The two-attractor account from report 6 survives, and sharpens.** It is not
"trained weights find generalisation, random weights find nothing" — exp24 beats
every trained model in the project. It is: **the attractor a model lands in
depends on whether retrieval is reachable from its representation**, not on
whether the sequence-processing weights are trained. Training finds retrieval
when the context makes retrieval possible (exp1, exp20 present-target; exp26)
and specialises toward it, degrading absent-target performance as it does.
When retrieval is not reachable — near-duplicate distractors and a frozen mixer
— there is nothing to specialise toward, and the embedding+head keep improving
generalisation for the full 12,000 steps.

For the published paper: the retrieval-crowds-out-generalisation claim should be
stated in terms of *reachable retrieval*, not *trained weights*, and exp24/exp26
are the pair of runs that forces the distinction. The matched-trainable-count
control above is the next run that would close the remaining gap.

## Sources

`results.jsonl` rows `exp24`, `exp25`, `exp26`, `ctx_ablation2`, cited against
`exp1`, `exp20`, `ctx_ablation` from report 6. Freezing is
`lib/train.py:freeze_labels` / `Run.train_only`; the ablation is
`scripts/ctx_ablation.py`.
