# Omniglot AR — Concepts

Algorithmic details for `projects/omniglot-ar`. The scientific argument lives in
[proposal.md](proposal.md); this file defines the task, the model, and the exact
meaning of every metric.

## Task / data

**Dataset.** Omniglot via `dpdl-benchmark/omniglot`, loaded by
`shared_lib.datasets.load_omniglot()`. The two HuggingFace splits are Lake et
al.'s originals:

| split | characters | alphabets | images |
|---|---|---|---|
| background (`_bg`) | 964 | 30 | 19 280 |
| evaluation (`_ev`) | 659 | 20 | 13 180 |

The character inventories are **disjoint**, and the loader raises if they ever
overlap. Images are resized to 28×28 (bilinear) and **inverted**, so ink is high
and "ink" means `> 0`, matching MNIST and letting the pixel-bin vocabulary mean
the same thing on both.

**Episode.** `lib/tasks.py`. An episode holds `n_way` characters × (`k_shot`
support + `n_query` query) drawings, flattened to a bag of `(pos, value, ref)`
tokens:

- a **support** drawing contributes `n_ctx` pixel tokens plus a label token
  whose value is the class's slot;
- a **query** drawing contributes `n_ctx` pixel tokens plus a label token whose
  value is `MASK` — that token is the prediction target.

Every drawing carries its own `ref` tag, drawn from a pool of `v_refs` and
re-drawn per episode. All drawings in an episode observe the **same** random
position pool; otherwise support and query would describe disjoint pixels and
cross-drawing matching would be ill-posed rather than merely hard.

**Why there is no shortcut.** Label slots are permuted per episode, so a
memorised class→slot map is worthless. A query's label appears nowhere among its
own tokens, so the only route to it runs through matching the query's pixels
against a different drawing of the same character. `ref` cannot shortcut it: a
query's `ref` never co-occurs with a label. This is exactly the shortcut that
made universal-ar's exp35/36 vacuous.

**Vocabulary.** Values are pixel bins `0..n_bins-1` ∪ label slots
`n_bins..n_bins+n_way-1` ∪ `MASK`. Positions are `0..img_size²-1` ∪ `pos_label`.

## Model & loss

`lib/models.py`. A post-LN-free pre-norm transformer over the token bag:

```
e   = pos_emb[pos] + val_emb[value] + ref_emb[ref]      (learned, additive)
x   = e; repeat L times: x += MHA(LN(x)); x += MLP(LN(x))
out = LN(x) @ head_W + head_b                            (over the value vocab)
```

No causal mask — the bag is a set, and `pos` is a *field*, not sequence order.
Defaults: `D_MODEL=256`, `N_LAYERS=4`, `HEAD_DIM=32`. These match
`universal-ar/experiments39.py` exactly, on purpose: this project varies the
data, not the model, so outcomes are comparable.

**Loss.** Cross-entropy on the masked query-label tokens only, averaged over
scored tokens. Pixel completion is added as an auxiliary term in exp3, not here.

**Efficiency.** One-hot matmul for embeddings (a gather's backward scatter
serialises on the mostly-background pixel values — 345× slower on MNIST);
`jax.checkpoint` per layer; gradient accumulation via `lax.scan`.

## Metrics

| metric | definition |
|---|---|
| `acc_ev` | **the headline.** N-way accuracy on episodes built from *evaluation* characters — never seen in training. Argmax restricted to the `n_way` label slots. |
| `acc_bg` | same, on *background* characters (seen in training). |
| `open_ev` | `acc_ev` with the argmax over the whole value vocabulary, pixel bins included. Lower than `acc_ev` whenever the model answers a label query with a pixel bin. |
| `train_acc` | N-way accuracy on the training episodes themselves. |
| `nn_ev`, `nn_bg` | pixel 1-NN (cosine) over the **same** `n_ctx` observed pixels the model sees. Computed in-repo, not quoted. |
| chance | `1 / n_way`. |

Restricting the argmax to label slots matters: an untrained head can answer a
label query with a pixel bin, which scores zero and conflates "wrong class" with
"emitted no class at all". `acc_ev` is the N-way decision; `open_ev` is the
stricter reading, and the gap between them is diagnostic on its own.

**Reading a result.** `acc_ev > nn_ev` is the bar for claiming in-context
learning. `acc_bg − acc_ev` is the memorisation gap; a large gap with
`acc_ev ≈ chance` reproduces the universal-ar failure on a new substrate, which
is itself a publishable-in-the-notebook negative.

## Reports

Reports are Typst, not markdown — see `report/` and the "Reports" section of
[workflow.md](workflow.md). Plots go through `shared_lib.typst_plot`
(gribouille grammar-of-graphics) rather than matplotlib, so a figure is data +
a declarative spec instead of imperative drawing code.

## Findings

**exp1–exp10 (2026-08-06/07) — chance on every run on the real task, including a
positive control. The first table is the original 12k-step batch-16 sweep; exp8–10
are the later 25k-step batch-64 runs and appear further down.**

| run | change | chance | 1-NN | unseen | seen |
|---|---|---|---|---|---|
| exp1 | baseline, 5-way 196px | 0.200 | 0.431 | 0.209 | 0.203 |
| exp2 | 2-way, 392px (easier) | 0.500 | 0.664 | 0.531 | 0.500 |
| exp3 | + label field | 0.200 | 0.431 | 0.188 | 0.200 |
| exp4 | + binarised values | 0.200 | 0.431 | 0.169 | 0.250 |
| exp5 | + ink-biased pool | 0.200 | 0.459 | 0.181 | 0.253 |
| exp6 | coarse 10x10, fully observed | 0.200 | 0.606 | 0.203 | 0.222 |
| exp7 | **identity query** (positive control) | 0.200 | **1.000** | 0.191 | 0.166 |

Each run's 1-NN floor is on that run's own observed pixels; exp5/6/7 change the
pool or pairing, so floors are not comparable across those rows.

**The decisive result is not in the table.** `scripts/tmp/plumbing.py` leaks the
answer into the query's own tokens in stages (2-way, 16px, binary, 0.5M params,
2 layers, 1500 steps):

| condition | what it requires | accuracy |
|---|---|---|
| `self` | read one field of its own label token | **1.000** (loss 0 by step 300) |
| `own-pixels` | attend by `ref`, pool 16 tokens, read a field | **1.000** |
| `none` | the real task: content matching | 0.500 |

So the loss, target, forward pass, head, **and `ref`-keyed attention all work.**
The capability that was missing is **content-dependent matching** — attending to
a token because its value resembles one's own.

**And that capability IS learnable.** `scripts/tmp/match.py` pushes batch to 64,
lr to 1e-3, adds the ink pool, 3 layers, 2-way, identity queries:

```
identity/ink/196  step 1000  loss 0.6934  acc 0.492
                  step 2000  loss 0.6930  acc 0.461   <- flat at ln 2
                  step 3000  loss 0.1033  acc 0.961   <- phase transition
                  step 5000  loss 0.0007  acc 1.000
```

A textbook abrupt onset. What separated this from exp7 (chance at 12 000 steps)
was **not** the step budget:

| | exp1–exp7 | the run that learns |
|---|---|---|
| effective batch | 16 | 64 |
| learning rate | 3e-4 | 1e-3 |
| pool | uniform (bar exp5) | ink-biased |
| steps | 12 000 | 6 000 |

Crossing a plateau is a signal-to-noise problem: whether the (small, nonzero)
gradient toward the matching circuit is visible above minibatch noise is set by
batch and step size, not by waiting longer at too small a batch. **exp1–exp7
were under-resourced, not under-trained** — which is why none of the task-side
interventions (label field, binarising, ink pool, coarse) registered.

**But exact and approximate matching dissociate — and that is the real finding.**
The transition is for *exact* matching (identity queries). On the real task the
same recipe is flat at ln 2 after 25 000 steps:

| run | matching required | 1-NN | model |
|---|---|---|---|
| `identity/ink/196` | exact | 1.000 | **1.000** |
| exp8 | approximate, 28x28 196px, 2-way | 0.729 | 0.488 |
| exp9 | approximate, 28x28 196px, 5-way | 0.480 | 0.228 |
| exp10 | approximate, 10x10 fully observed | 0.805 | 0.488 |

exp10 was the sharpest test of "approximate is just far-away exact": coarser
images make two drawings of a character differ in far fewer pixels, and the 1-NN
floor rises to 0.805. It made no difference.

**Why the margin collapses.** The label-field circuit computes a per-position
soft vote (attend at this position, weight by value agreement, read the label)
and the query's label token *averages* them. With `a+` = agreement fraction with
the correct support and `a-` with a wrong one, the pooled signal ∝ `a+ - a-`:

- exact:       `a+ = 1.0`, `a- ≈ 0.7` → margin ≈ 0.3
- approximate: `a+ ≈ 0.8`, `a- ≈ 0.7` → margin ≈ 0.1, a small difference between
  two large noisy sums both dominated by shared background.

Pixel 1-NN succeeds on the same pixels because it **normalises** and takes an
argmax across candidates — a global operation over whole drawings. An additive
per-position accumulator cannot express that. This is a limitation of the
*representation*, not the optimiser, and it explains why every task-side knob
(ink pool, binarising, coarsening) failed: each nudges the margin, none replaces
the additive accumulation with a normalised comparison.

exp9 (5-way) was re-run to completion: 0.228 vs chance 0.200 (1.3 SE, inside
noise) against a 0.480 floor. Both episode widths behave the same way.

**Consequence for the plan.** Judge runs in this family by *whether they crossed
a transition*, not by accuracy at a fixed step count — "chance at 12 000 steps"
may just mean "reported before the transition". Tune batch and lr before
touching the architecture; they were the binding constraint and are cheap to
sweep. The proposal's sweep / masked-pixel / alphabet-holdout plan all presuppose a
working approximate match and stay parked. The next intervention should change
*what is accumulated*: a normalised similarity, or a pooled per-drawing summary
token to compare against. Both reintroduce a representation of a *sample* —
exactly what the token-level premise set out to dissolve, which is the tension
worth writing up.

Note train accuracy sits at chance alongside test throughout. That is expected,
not a bug: class-to-slot assignment is re-drawn per episode, so there is no
memorisable component to fit.
