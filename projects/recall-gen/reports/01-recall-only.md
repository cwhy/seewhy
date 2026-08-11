# Report 1 — recall generalises perfectly, completion barely at all

exp1, exp2, `baselines_M16_r14`. M=16 context images, bottom 14 rows hidden,
12 000 steps at batch 256, 4-layer KDA, 4.03M params.

All numbers are **normalised MSE** on the hidden pixels: model MSE divided by the
MSE of predicting the train-set mean image. **1.0 means "no better than ignoring
the input entirely"**; 0 is perfect.

## The four conditions

| | target IS in context | target NOT in context |
|---|---|---|
| context images seen in training | **A** — the only thing exp1 trains on | **C** |
| context images never seen | **B** | **D** |

## Reference points (no model involved)

| | A | B | C | D |
|---|---|---|---|---|
| predict the mean image | 1.000 | 1.000 | 1.000 | 1.000 |
| **ridge** — global linear inpainter fitted on the train pool, context ignored | 0.646 | 0.642 | 0.633 | 0.645 |
| **nn1** — copy the hidden half of the closest context image | 0.000 | 0.000 | 1.573 | 1.575 |
| **knn** — softmax-weighted average of the context hidden halves, best temperature | 0.000 | 0.000 | 1.004 | 1.002 |

The last row is the important one. `knn` is the strongest thing reachable by
attending to the context and nothing else — and it is exactly the shape of
computation linear attention performs. **At M=16 its best achievable score is
1.002: identical to ignoring the context.** Sixteen random MNIST digits contain
essentially no information about how to finish a seventeenth. The optimal
temperature the sweep picks is the one that flattens the weights into a uniform
average, i.e. it reconstructs the mean image.

So on C/D there is nothing to retrieve, and unlike A/B, retrieval is not merely
unhelpful — the hard look-up answer (1.57) is *much worse* than the trivial prior.

## Results

| exp | trained on | A | B | C | D | id acc A | id acc B |
|---|---|---|---|---|---|---|---|
| exp1 | recall only | **0.015** | **0.017** | 0.851 | 0.852 | 1.000 | 1.000 |
| exp2 | completion only | 0.041 | 0.672 | 0.040 | 0.669 | 1.000 | 0.778 |

`id acc` is the fraction of queries where the model's output is closest, on the
hidden pixels, to the correct one of the 16 context images. Chance is 0.0625.

## Finding 1 — retrieval is fully content-addressed

exp1 scores 0.015 on A and **0.017 on B**, with identification accuracy 1.000 on
both. The model has never seen a single image in the B episodes; it retrieves
them perfectly anyway. Whatever it learned is a mechanism for matching a partial
image against a set held in the state, not a memorised table. The A→B transfer
is essentially free.

This is the "does pure recall generalise?" question answered in its easy
direction, and the answer is an unambiguous yes.

## Finding 2 — the completion ability is real but small, and training destroys it

exp1 on C/D lands at 0.851. Read against the reference points, that is:

* **better than every pure-context strategy** (1.00 for the best soft look-up,
  1.57 for hard look-up). The model is not doing look-up when look-up is wrong —
  it only points at the look-up pick 41% of the time.
* **worse than ridge regression** (0.64). A single linear map fitted on the same
  training pool, ignoring the context entirely, beats it comfortably.
* **worse than the same architecture trained to complete** (exp2's best D, 0.458).

So a recall-only model does acquire *something* generalisable — it beats the
prior, and it beats anything the context alone can offer — but it does not reach
even a linear regression on the data it trained on.

And the effect **runs backwards during training**. exp1's C/D curve rises
monotonically: 0.64 at step 500, 0.71 at 1 000, 0.80 at 2 000, 0.85 at 12 000,
while A/B falls from 0.38 to 0.015. The completion ability is highest when
recall is *worst*, and it is spent as recall sharpens. The generalising fragment
looks like a transient of early training rather than a by-product of the recall
solution.

## Finding 3 — the completion-trained model does not use the context either

exp2 is meant to be the ceiling, and it is a strange one. Its A (0.041) and C
(0.040) are the same number: **having the target sitting in the context is worth
nothing to it.** It did not learn to look; it memorised the 60 000 training
images into its weights. The proof is B (0.672) and D (0.669) — also the same
number as each other, and 16x worse than A. On novel images it has nothing.

Its D curve worsens monotonically from 0.458 at step 1 000 to 0.669 at 12 000,
which is textbook overfitting. **0.458 (early-stopped) is the honest ceiling**
for this architecture on this task, and it is only modestly better than ridge.

Both training modes therefore fail to make the model use the context to
generalise, for opposite reasons: exp1 uses the context but only to copy, exp2
generalises but ignores the context. Neither is surprising once the baseline
table is on the table — **at M=16 there is no gain available from using the
context to generalise**, so no objective would push either model to.

## What that implies for the next experiments

The interesting regime must be one where the context is actually informative.
`knn` at M=16 is 1.00; it can only improve with M, since nearest-neighbour is a
universal learner in the limit. The M-sweep (exp4 M=64, exp5 M=256, exp6 M=4)
asks whether the pure-recall model tracks a look-up ceiling that is genuinely
worth tracking. exp3 (half the queries generalising) asks how much explicit
signal it takes to buy the completion ability back.

## Figures

* exp1 learning curves — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp1_curves.svg>
* exp2 learning curves — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp2_curves.svg>

## Caveats

* One seed each. exp10/exp11 (recall) and exp12 (completion) replicate.
* "Novel" here means a novel *image* from the same ten classes, not a novel
  class. The digit-split variant is not run yet.
* The mask is always the bottom half. Nothing here has been checked against
  random-pixel masking, where retrieval is easier and completion is much easier.
