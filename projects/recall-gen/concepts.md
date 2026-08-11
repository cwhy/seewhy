# Recall-Gen — Concepts

Does a model trained **only to retrieve** learn anything that generalises?

## Task / data

One MNIST image = one token. An **episode** is

```
[ ctx_1 ... ctx_M ]  [ qry_1 ... qry_Q ]
```

* `ctx_i` — a full 28x28 image, flattened to 784 floats in [0,1].
* `qry_j` — the same 784 vector with the **bottom `MASK_ROWS` rows zeroed**, plus a
  binary mask channel so the model knows which pixels were removed.
* Target for `qry_j` — the true pixel values at the masked positions.

Two independent image pools:

| pool | source | seen in training |
|---|---|---|
| `TRAIN` | MNIST train split (60 000) | yes |
| `HELD`  | MNIST test split (10 000) | never |

## Model & loss

A 4-layer KDA (Kimi Delta Attention) linear RNN — a matrix-valued memory `S`
written by the delta rule, per-channel forget gate `alpha`, write strength `beta`:

```
forget   S~ = S . Diag(alpha_t)
predict  vhat = S~ k_t
correct  e = beta_t (v_t - vhat)
write    S  = S~ + e k_t^T
read     o_t = S q_t / sqrt(dk)
```

**Context tokens write, query tokens never write** (`beta` gated to 0), and every
token reads the *completed* state. The state is therefore the **only** channel
from context to query — a hard information bottleneck of
`N_HEADS x DK x DV = 4 x 64 x 64 = 16 384` floats, against `M x 784` floats of
context content. This is the "hidden-state gram" the project is about.

Loss: MSE on masked pixels only.

## The 2x2 test matrix

Training uses **condition A only**: pure recall of an image that is present in
the context.

| | target IS in context | target NOT in context |
|---|---|---|
| context from `TRAIN` pool | **A** — train condition | **C** |
| context from `HELD` pool  | **B** | **D** |

* **A -> B** — does retrieval generalise to images never seen in training, i.e.
  is the mechanism content-addressed or memorised?
* **A -> C/D** — what does a recall-only model do when there is nothing to
  recall? Fall back to a blurry prior, copy the nearest context image, or
  something better than both?

## Metrics

All on masked pixels only, pixel scale [0,1].

| metric | definition |
|---|---|
| `mse` | model MSE |
| `mse_mean` | baseline: predict the train-set mean image |
| `mse_nn` | baseline: of the M context images, take the one whose **visible** part is closest to the query's visible part, and copy its masked part. This is the best answer a pure look-up can give. |
| `nmse` | `mse / mse_mean` — 1.0 means "no better than the prior", 0 is perfect |
| `id_acc` | (A/B only) argmin over the M context images of the distance from the model output to each context image's masked part; correct when it equals the true target index. Chance = `1/M`. |

Two further model-free references are computed by `scripts/baselines.py` and
stored in their own results row: **ridge**, a global linear inpainter fitted on
the train pool that ignores the context entirely, and **soft look-up**, a
similarity-weighted average of the context images' hidden halves with the
temperature swept.

Soft look-up, not `mse_nn`, is the crux. It is the strongest answer obtainable
from the context alone, and it is precisely the shape of computation linear
attention performs — so it, and not hard 1-NN, bounds what a retrieval mechanism
can do. (An early version of this file claimed a recall model could not beat
`mse_nn` on C/D "by definition". That was wrong: every trained model here beats
it comfortably, because copying one neighbour is a far worse strategy than the
mean image.)

## Findings

Normalised MSE throughout: model MSE / mean-image MSE, so 1.0 = "no better than
ignoring the input". `gain` = nMSE(D) - nMSE(B), i.e. how much having the answer
in the context is worth on images the model has never seen. It is the cleanest
single measure of whether a model retrieves at all.

**1. Retrieval is content-addressed and transfers for free.** exp1/10/11
(3 seeds): condition A 0.015/0.016/0.016, condition B 0.017/0.018/0.018,
identification accuracy 1.000 on both. Images never seen in training are
retrieved exactly as well as training images.

**2. At M=16 the context carries no completion signal.** The best soft look-up
scores 1.002 on condition D — identical to predicting the mean image, and the
temperature the sweep picks is the one that flattens the weights into a uniform
average. Hard 1-NN is much worse, 1.575. So on absent-target episodes there is
*nothing to gain* from using the context, and no objective could push a model to.

**3. Recall training does not buy completion; it spends it.** exp1's C/D rises
monotonically 0.64 -> 0.71 -> 0.80 -> 0.85 while A/B falls 0.38 -> 0.015. The
final 0.85 is better than any pure-context strategy but worse than ridge
regression (0.64), which ignores the context entirely.

**4. The two abilities do not compete.** exp3 (half the queries generalising)
reaches the full completion ceiling (best D 0.459, same as exp2's 0.458) while
keeping substantial retrieval (gain 0.134). So the recall-only model's deficit
is not a capacity conflict — the objective simply never asks.

**5. Generalisation appears exactly when retrieval fails.** M-sweep, recall
training, `gain` / final D:

| M | gain | D | best D | soft-look-up ceiling on D |
|---|---|---|---|---|
| 4 | 0.840 | 0.854 | 0.777 | 1.237 |
| 16 | 0.834 | 0.852 | 0.637 | 1.002 |
| 64 | 0.622 | 0.658 | 0.535 | 0.886 |
| 256 | **0.004** | 0.561 | 0.443 | 0.786 |

At M=256 the recall-trained model has *no* retrieval gain — the same 0.00 as the
completion-trained models — and its best D (0.443) sits on the completion
ceiling (exp2 0.458, exp7 0.450). It stopped using the context and converged on
the same weight-memorised solution a completion objective finds. Its D is far
below the look-up ceiling at every M, so the improvement is not context-driven.

**6. Completion-trained models memorise rather than use context.** exp2/7/12:
condition A equals condition C to three decimals, and both are ~16x better than
B/D. Their D worsens monotonically after ~step 1 000, so the quoted ceiling is
early-stopped.
