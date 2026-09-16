#import "/template.typ": *

= Conclusion

The comparison replicates: all three posted accuracies sit inside our five-seed
distributions, the parameter counts match exactly, and PC-ALM's forty-fold
training cost against backpropagation reproduces to within a factor of 1.02. As
arithmetic about three particular runs, the original table is correct.

As a ranking of learning rules it is not, and the two reasons are separable and
both measurable. The three rows use two different architectures, because
Forward-Forward needs a short wide network and the PC-ALM benchmark is a tall
thin one; backpropagation run on Forward-Forward's own architecture beats it by
2.5 points, and Forward-Forward run on the tall thin one loses 31. All three
rows use one learning rate, which is the best rate for exactly one of them —
Forward-Forward, the winner; give each arm its own best rate and the order of
those three rows reverses, to backpropagation 0.9374, PC-ALM 0.9277,
Forward-Forward 0.8869. What survives both corrections is the source paper's own claim, which
we can isolate because PC and PC-ALM differ here by one scalar: the dual
variables are what keep a local rule aligned with backpropagation as the network
gets deeper, worth 24 points of accuracy at 32 layers and a gradient cosine of
0.89 against 0.81.

The two arms we added to the original three answer a question it could not.
Plain predictive coding is the ablation that isolates PC-ALM's dual variables,
and EGGROLL — which never differentiates anything — reaches 0.9025 in one epoch
with an update whose cosine to the true gradient is 0.094, and is the only rule
here whose accuracy is unchanged from 2 to 64 layers. A method with no notion of
a layer has no credit-assignment problem to fail at, which is what makes the
other three rules' depth curves interesting rather than inevitable.

Two findings were not on the list of things we set out to measure.
Forward-Forward's deep failure is a readout failure rather than a representation
failure — a linear probe on the 31-layer network's own activations reaches
0.8919 where its goodness sweep gets 0.5751 — which suggests the method's
classifier, not its learning rule, is what does not scale with depth. And
Forward-Forward is the only arm here that gets *worse* with more training, on
both readouts, which makes its one-epoch showing in the original table a
best-case number in a way the other rows' are not.

= Appendix: reproduction <appendix-repro>

#kv(
  ("commit", "see the repository at the revision that adds projects/local-learning"),
  ("hardware", "one NVIDIA RTX 4090, JAX 0.8.1, CUDA 13, float32"),
  ("data", "MNIST and Fashion-MNIST via HuggingFace datasets, standardised per section 3"),
  ("command", "uv run python projects/local-learning/scripts/run_experiments.py --bg expN"),
  ("figures", "uv run python projects/local-learning/scripts/gen_report.py"),
)

Every row of `projects/local-learning/results.jsonl` is one cell, keyed by
`experiment`, carrying its full hyperparameters, parameter count, timings,
per-epoch curve and final metrics. The headline numbers come from:

#table(
  columns: (auto, 1fr),
  align: (left, left),
  [Table 1, figures 1-3], [`exp1_*`, `exp2_*` — 30 cells],
  [Learning-rate grid, figure 4], [`exp3_*` (50 cells), `exp7_*` (12 cells)],
  [Tuned comparison, figure 5], [`exp9_*` — 30 cells],
  [Wall-clock curves, figure 6], [`exp4_*` — 15 cells, 10 epochs each],
  [Depth sweep, figures 7-8], [`exp5_*` — 84 cells],
  [Fashion-MNIST check], [`exp6_*` (9 cells), `exp8_*` (6 cells)],
  [EGGROLL configuration], [`exp10_*` (24), `exp11_*` (4), `exp15_*` (12 cells)],
  [EGGROLL population, figures 9-10], [`exp12_*` — 12 cells],
  [EGGROLL table rows], [`exp13_*` (15 cells), `exp16_*` (5 cells)],
  [EGGROLL depth and wall-clock], [`exp14_*` — 21 cells],
  [Shared-GPU timing calibration], [`exp17_*` — 9 cells],
  [EGGROLL on the shared rate grid], [`exp18_*` — 10 cells],
)

Each experiment file's module docstring states the hypothesis it was written to
test. `lib/methods.py` contains backpropagation, PC and PC-ALM, `lib/ff.py`
Forward-Forward and `lib/es.py` EGGROLL; `lib/train.py` holds the timing
protocol, and `lib/model.py` the two architectures.
