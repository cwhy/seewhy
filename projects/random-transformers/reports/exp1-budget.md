# Warmup decides the baseline, not the frozen model

A fully trained width-1024 transformer scores **0.14** on needle-in-a-haystack,
against a chance level of 0.008. It stays there for 10,000 steps — the budget the
paper itself specifies. The *same architecture*, with its entire interior frozen
at random initialisation, scores **1.00**.

The gap is not a property of the models. It is a learning-rate warmup our first
implementation lacked. Only one of the two conditions needs it.

## Why this was worth chasing

Taken at face value, the first pass said random transformers beat fully trained
ones on three of four tasks. That is a much stronger claim than the paper makes.
Such a result should be assumed to be a harness bug until proven otherwise. It
was one.

## What the plateau was

The fully trained model sat at loss 3.70 with accuracy 0.1416. Dead flat, from
step 500 to step 10,000.

That accuracy is not noise. Suppose a model learns to answer with one of the
values that appeared in the context, but never learns *which* one. It would score
`E[1/k]` for `k ~ U[1,30]`, which is **0.1332**. The model had learned to
restrict its answer to the context and nothing more. It never formed the
retrieval circuit.

## What the authors' code says

Checked afterwards. `experiment_stream.py` sets `warmup_steps = 500` and
`lr_scheduler_type = "cosine"`. Neither appears in the paper. Our bisected value
of 500 matches theirs exactly, and the cosine decay explains the one setting we
could not stabilise below.

## The four settings

All on `needle/full/1024`, seed 0.

| learning rate | warmup | result |
|---|---|---|
| 1e-3 | none | 0.14, flat for 10,000 steps at the paper's own budget and batch |
| 3e-4 | none | climbs to ~0.40 and stalls |
| 1e-3 | 500 steps | 0.92 at step 500, then destabilises to 0.63 |
| **3e-4** | **500 steps** | **0.9985 at step 500, 0.9998 at step 1000** |

Embedding-only training reaches 1.00 under all four.

## What it changes

**For the replication:** every experiment now uses lr 3e-4 with a 500-step
linear warmup, in *every* condition, so no comparison is between differently
tuned optimisers. This is a declared deviation — the paper specifies 1e-3 and
mentions no warmup. The 92 rows from the first pass are kept in `results.jsonl`
under the `nowarmup_` prefix rather than deleted; they are the evidence for this
report.

**For reading the paper's result:** this is small independent support for the
paper's own thesis. Embedding-only training was insensitive to an optimiser
detail that decided total failure or perfect accuracy for full training.

That is what you would expect if the random model is not searching for a circuit.
It searches only for an encoding of one that already exists. There is no fragile
search to disrupt.

Worth stating plainly: this was noticed by accident, as a side effect of a bug.

**For the next run:** the convergence probe that set our budgets was run on the
random condition only. That is the actual methodological error here. A shared
budget must be calibrated on the *slowest* condition, not the fastest.
