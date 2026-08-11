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

`mse_nn` is the crux. A pure recall model cannot beat it on C/D by definition of
what it is doing. Beating it is evidence of real generalisation.

## Findings

(appended as they land)
