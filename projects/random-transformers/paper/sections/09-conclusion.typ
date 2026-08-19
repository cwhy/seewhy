#import "/template.typ": *

= Conclusion <sec-conclusion>

A transformer whose attention and feed-forward weights are frozen random numbers
can do modular arithmetic, retrieve a value from a list, add ten-digit numbers,
and check whether parentheses balance. It reaches 1.000 on all four, from five
seeds out of five, against chance levels of 0.005, 0.008, ~0 and 0.681.

We reimplemented the paper in JAX from its text, and every central claim
reproduced. Nineteen of twenty cells in the main table land within 0.15 of the
published numbers once the baselines are given a learning-rate search. The one
that does not is our LSTM on associative recall, and we could not close it.

The mechanism reproduced too. Ten principal directions out of 1024 explain most
of the variance, while ten neurons explain almost none — a twenty-fold gap that
rules out a sparse-subnetwork account. Capacity for arbitrary facts is an order
of magnitude lower than under full training, 0.20 bits per parameter against
2.40. And the paper's own falsification test holds: a random transformer imitates
a narrow target circuit almost perfectly, and degrades sharply once the target
needs more than about sixteen dimensions.

Two things we can add to the original.

First, the paper's stated optimiser is not sufficient to reproduce its fully
trained baselines. A 500-step warmup and cosine decay are needed, both present in
the authors' code and absent from the paper. This is invisible in the paper's own
results because it affects only the baseline condition.

Second, that omission is itself weak evidence for the paper's thesis.
Embedding-only training reached 1.000 under every optimiser setting we tried;
full training ranged from 0.14 to 1.000 over the same four. A model that only has
to find an encoding of an existing circuit has no fragile search to disrupt. The
same asymmetry shows up in seed variance: one fully trained run in five never
left the plateau, while every embedding-only run landed on exactly 1.000.

What we would not conclude is that random transformers are good models. They are
worse at memorization by a factor of twelve, and need roughly eight times the
trainable parameters to match a fully trained language model. The interesting
claim was never that freezing helps. It is that the frozen network already
contains more than nothing, and that a suitable choice of input and output
encoding is enough to reach it.

= Appendix: reproduction

All results come from `projects/random-transformers` in the `seewhy` repository.

#kv(
  ("commit", "see git log for the commit that adds projects/random-transformers"),
  ("hardware", "2x NVIDIA RTX 4090, 24 GB each"),
  ("framework", "JAX with Optax; no PyTorch or HuggingFace in the training path"),
  ("data", "TinyStories via HuggingFace datasets; all other tasks generated in-process"),
)

Task encodings are verified before any run:

```
uv run python projects/random-transformers/scripts/tmp/verify_tasks.py
```

This decodes samples from every task and checks them against an independently
written implementation of each rule. It is the check the rest of the paper rests
on.

Experiments, in the order they must run:

```
uv run python projects/random-transformers/experiments1.py   # main table
uv run python projects/random-transformers/experiments2.py   # embedding ablation
uv run python projects/random-transformers/experiments3.py   # width sweep
uv run python projects/random-transformers/experiments4.py   # memorization
uv run python projects/random-transformers/experiments5.py   # subspace analysis
uv run python projects/random-transformers/experiments6.py   # circuit imitation
uv run python projects/random-transformers/experiments7.py   # language modeling
uv run python projects/random-transformers/experiments8.py   # baseline lr search
```

`experiments5.py` reads parameters saved by `experiments1.py` and
`experiments4.py`, so it must follow both. `RT_TASKS` splits a run across GPUs by
task; `RT_DEPTHS` does the same for the language-modeling sweep.

Figures and the paper:

```
uv run python projects/random-transformers/scripts/gen_report.py
uv run python -m shared_lib.publish projects/random-transformers/paper --check
uv run python -m shared_lib.publish projects/random-transformers/paper --stable
```

== Which rows back which numbers

Every row in `results.jsonl` carries an `experiment` field and a `cell` key.

#table(
  columns: 2,
  stroke: 0.5pt + luma(200), inset: 5pt, align: (left, left),
  table.header([*Section*], [*Rows*]),
  [Main table],          [`exp1_<task>`, cell `<task>/<mode>/<width>/<seed>`],
  [Width sweep],         [`exp3` plus the width 16 and 1024 endpoints from `exp1_<task>`],
  [Embedding ablation],  [`exp2`, plus the `random` rows from `exp1_<task>`],
  [Memorization],        [`exp4`, fields `full_set_acc` and `bits_per_trainable_param`],
  [Subspace analysis],   [`exp5`, fields `pc_<layer>` and `neuron_<layer>`],
  [Circuit imitation],   [`exp6`, field `final_kl`],
  [Language modeling],   [`exp7`, field `eval_loss`; completions in field `sample`],
  [Baseline lr search],  [`exp8`],
)

Three superseded generations of the main table are kept in the same file, under
the prefixes `nowarmup_`, `constlr_` and `head8_`. They are the evidence for the
optimiser discussion in @sec-limitations and are not used in any figure.
