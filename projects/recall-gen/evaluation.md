# The standard evaluation

Every report from 12 onward measures the same two abilities — **find** an item
that is present in the context, and **predict** one that is not — and each
report found a way the numbers could mislead that the previous one had missed.
This file is those lessons as a fixed procedure. `scripts/standard_eval.py`
implements it and writes one `<exp>_stdeval` row per checkpoint.

Run it after any training run:

    uv run --no-sync python projects/recall-gen/scripts/standard_eval.py exp42

A report should **quote** that row, not re-derive it.

## The design

Three novelty bands crossed with two target conditions, and the whole 3x2 scored
under two context types.

| band | pool | means |
|---|---|---|
| A / C | `train` | items seen in training |
| E / F | `held_same` | new items, same world |
| B / D | `held` | a different world |

`held_same` is not optional. Without it, "the network has never seen this item"
and "the network has never seen this kind of item" are the same column, and
report 13 attributed to the second what belonged to neither.

Present (A/E/B) and absent (C/F/D) conditions **share their queries**. Every
number is normalised by the error of drawing the average training item, which
depends only on the queries; drawing the two conditions separately made the
denominators differ by up to 2% and put a ~0.02 floor under every
present-versus-absent comparison.

Both context types are scored at the **same Q**. A knn context hands each query
M/Q neighbours, so Q changes what the context *is*. A comparison across
different Q is not a comparison of context type.

## What every number needs beside it

**Normalised error** needs its 1.0 line named. 1.0 is "no better than the
average training item" — which is a weak constant when the test pool is a
different dataset, and therefore easier to beat than it looks. Quote the ridge
map (ignores the context) and, where the domain makes it relevant, predict-zero.

**Identification accuracy** needs two references, not one.

*Its ceiling.* Chance is 1/M at the bottom, but the top is not always 1.0. When
two context items can share a hidden half — sparse chess endgames especially — a
perfect reconstruction still loses the tie-break. `oracle_id_acc` scores the true
target against its own context and returns what is actually available. On chess
endgames it is 0.955.

*Its pool's nearest-rival distance.* Identification depends on how far the target
sits from its closest competitor, which is a property of the pool and the context
and has nothing to do with the network. Report 15 found this worth more than the
entire class-novelty effect report 13 had attributed to the network.
`nearest_distractor` returns it per episode; the row carries the median and the
lower quartile.

Two identification numbers are comparable only when they come from pools with
comparable margins. Otherwise the comparison is about the data.

The margin is a property of the *draw*, not just the pool: at 512 episodes a
median moves by a few hundredths between seeds. The row records `eval_seed` for
that reason. Quote one row rather than mixing numbers from different draws.

**On boards, quote piece accuracy too.** Normalised error and piece accuracy
disagree in *sign* on chess endgames: 0.652 reads as "better than the average
board" while 82.9% of squares correct is *worse* than the 90.6% from guessing
every hidden square empty. Squared error over one-hot planes rewards hedging;
counting pieces does not. `sq_acc` is logged automatically for board domains and
its all-empty reference lives in the baselines row.

## The two context types

`iid` — sixteen unrelated items. The baselines showed such a context carries
almost nothing about a seventeenth, so an absent-target episode has nothing to
use and no objective can reward using it.

`knn` — the query's own nearest neighbours by visible half. Informative, and by
construction **low-margin**: it is assembled from the query's closest matches, so
identification gets harder there even as everything else gets easier. On
Fashion-to-MNIST the median margin falls from 0.68 to 0.39.

Training context and test context are separate factors and the 2x2 separates
them. On Fashion-to-MNIST, completion was governed almost entirely by the test
context (0.88 to 0.53) and barely at all by the training context (0.878 against
0.860); identification went the other way.

## Model-free references

`scripts/baselines.py` writes these once per (domain, M, mask, context) and they
do not depend on any checkpoint:

| | |
|---|---|
| `mean` | draw the average training item — the 1.0 line |
| `zeros` | predict all-zero; on MNIST and on boards this is a real strategy |
| `ridge` | linear map, visible to hidden, fitted on the train pool, **ignores the context** |
| `nn1` | copy the hidden half of the nearest context item — best pure look-up |
| `knn_soft` | similarity-weighted blend of the context, temperature swept — the best the context alone allows, and the shape of computation linear attention performs |

`knn_soft` is the one that matters: a model beating it is doing something its
objective never asked for.

## Domains

`lib/domains.py` owns what a token is. Adding one means supplying its width,
mask, pools, class labels and renderer; nothing in `lib/train.py` changes.

| domain | token | mask | novel pool |
|---|---|---|---|
| `mnist`, `fashion_mnist` | 784 = 28x28 grey | bottom 14 rows | classes 5-9 |
| `chess` | 832 = 8x8x13 piece planes | files a-d | endgames, <=10 pieces |
| `fashion_to_mnist`, `mnist_to_fashion` | 784 | bottom 14 rows | the other dataset |

A cross-dataset pair shifts the novel dataset's labels by ten and concatenates
the two test splits, so "novel dataset" and "novel class" are the same operation
and the split machinery needs no special case.

Note the architecture is **blind to geometry**: `W_pix` is a dense matrix over a
flat vector, so a mask over image rows and a mask over board files are the same
kind of object to it. Any new domain only has to supply a flat vector, a mask,
and a notion of identity.
