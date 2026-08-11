# Report 2 — generalisation appears exactly when retrieval fails

exp1, exp3–exp7, exp10–exp12, and the model-free baselines at M = 4, 16, 64, 256.
Bottom 14 rows hidden, 12 000 steps, 4-layer KDA, 4.03M params, state fixed at
16 384 numbers throughout.

Normalised MSE (model MSE ÷ mean-image MSE): **1.0 = no better than ignoring the
input**, 0 = perfect.

## A better question than "does it generalise"

Report 1 measured completion quality on absent-target episodes and found the
recall-only model mediocre. That number conflates two things. The sharper
measure is

**gain = nMSE(D) − nMSE(B)**

Both conditions use context images the model has never seen; they differ *only*
in whether the query's true image is among them. gain is what having the answer
in the context is worth. A retriever scores high; a model that ignores the
context scores zero. Unlike identification accuracy it cannot be inflated by a
model whose completions happen to be good enough to pick the right neighbour —
which matters, because at large M that is exactly what happens.

## The sweep

Recall training, one seed each, state fixed at 16 384 numbers:

| M | context content | gain | final D | best D | soft look-up ceiling on D |
|---|---|---|---|---|---|
| 4 | 3 136 | 0.840 | 0.854 | 0.777 | 1.237 |
| 16 | 12 544 | 0.834 | 0.852 | 0.637 | 1.002 |
| 64 | 50 176 | 0.622 | 0.658 | 0.535 | 0.886 |
| 256 | 200 704 | **0.004** | 0.561 | **0.443** | 0.786 |

Seed spread at M=16 (exp1/10/11): gain 0.834 / 0.828 / 0.808; B 0.017 / 0.018 /
0.018; identification accuracy 1.000 in all three.

Completion-trained references: exp2 (M=16) best D 0.458, gain −0.002; exp7
(M=64) best D 0.450, gain 0.006; exp12 (M=16, seed 1) best D 0.458, gain 0.010.

## Finding — the trade is total, not gradual

At M = 256 the recall-trained model's gain is **0.004**. That is not "reduced
retrieval"; it is the same number the completion-trained models post (−0.002,
0.006, 0.010), and those never retrieve by construction. **A model trained
exclusively on retrieval has stopped retrieving.**

And its best D, 0.443, sits on the completion-trained ceiling — 0.458 at M=16,
0.450 at M=64. Trained on opposite objectives, the two arms converge on the same
solution: memorise the training distribution in the weights and ignore the
context. Condition A equals condition C (0.128 vs 0.134) and B equals D (0.556
vs 0.561), which is the signature of a model for which target presence is simply
not a variable.

So the honest reading of "pure recall generalises at large context" is the
opposite of the hopeful one. Generalisation did not *emerge from* recall. Recall
became impossible, the objective stopped rewarding it, and gradient descent
found the only other minimum available — the ordinary one.

## The improvement is not coming from the context

The look-up ceiling improves with M (1.237 → 1.002 → 0.886 → 0.786), so one
could tell a story where the model tracks a context that is becoming genuinely
informative. Three things rule it out.

1. The model is **below** the ceiling at every M, by a widening margin. At
   M=256 it reaches 0.443 against a ceiling of 0.786. Whatever produces 0.443 is
   not extraction from the context.
2. gain is zero at M=256. If the context were the source, removing the target
   from it would not be free — but everything else in the context is still there,
   and the model does the same thing either way.
3. Condition C beats condition D by a factor of four at M=256 (0.134 vs 0.561).
   The advantage tracks *whether the image was in the training pool*, not what is
   in the context.

The ridge baseline, a single linear map fitted on the training pool that ignores
the context entirely, scores 0.63–0.65 at every M. The recall-trained model is
worse than that at M = 4, 16; roughly equal at M = 64; and better only at M =
256, where it has stopped retrieving.

## Finding — the two abilities do not compete

exp3 trains on a 50/50 mixture. It reaches best D 0.459 — the full completion
ceiling, no worse than training on completion alone — while keeping gain 0.134,
far above the completion-trained models' ~0.00.

This rules out the most natural explanation of Report 1's finding. The
recall-only model is not failing to complete because the state is busy holding
16 images; the same state does both when asked. It fails because a recall
objective supplies **no gradient at all** toward completion, and at M=16 the
baselines say why: the best possible use of the context on an absent-target
episode scores 1.002, i.e. exactly the same as ignoring it. There is nothing
there to be rewarded for finding.

## Caveats

* The M-sweep changes two things at once — it overruns the memory *and* enlarges
  the context. exp15/16/17 shrink the state at fixed M=16 to separate them; that
  is the load-bearing control for the claim above and is reported in Report 3.
* Every sweep point is one seed. The M=16 row is three.
* Completion-trained runs overfit; "best D" is early-stopped. Final-step D is
  much worse (0.669 at M=16).
* MSE rewards hedging. The completion grids show the recall-only model's
  absent-target output is not a plausible digit, so its numerical margin over the
  mean-image prior overstates its visual one.

## Figures

* exp1 learning curves — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp1_curves.svg>
* exp4 (M=64) — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp4_curves.svg>
* exp5 (M=256) — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp5_curves.svg>
* exp1 completions — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp1_grid.png>
* exp5 completions — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp5_grid.png>
