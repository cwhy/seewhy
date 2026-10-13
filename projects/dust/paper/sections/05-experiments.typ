#import "/template.typ": *

= Experiments <sec-experiments>

Each run is a file `experimentsN.py` in the project, launched through `scripts/run_experiments.py`
and appending rows to `results.jsonl`. This table is the bridge between the paper and the repo.

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, left),
  [Run], [Question it answers], [Source],
  [exp1], [Does our backprop reproduce the paper's baseline at 1M and 10M tokens? (The positive control for the model, the data, and the SGD protocol.)], [`experiments1.py`],
  [exp2], [Backprop checkpoints at 10M and 100M tokens, the weights at which gradient alignment is measured.], [`experiments2.py`],
  [exp3], [Does our DUST reach the published loss at 256 and 1,024 draws, with the authors' settings?], [`experiments3.py`],
  [exp4, exp9], [DUST at 64 draws, for which no settings were released: the 12-point grid, then three seeds at its winner.], [`experiments4.py`, `experiments9.py`],
  [exp5, exp8], [How strong can EGGROLL be made? The Adam rate and noise-scale grid, then the paper's own Adam constants.], [`experiments5.py`, `experiments8.py`],
  [exp6], [How well does each estimate align with backprop's gradient, as population grows?], [`experiments6.py`, `lib/align.py`],
  [exp7], [Is our backprop baseline tuned as well as the paper's?], [`experiments7.py`],
  [exp10], [The 1M-token ladder for EGGROLL at 64, 256 and 1,024 passes, three seeds.], [`experiments10.py`],
  [exp11], [The 10M-token column at 256 draws and passes: does DUST's distance to backprop reproduce?], [`experiments11.py`],
  [exp12, exp13], [The 100K-token ladder: tuning grids for all three methods, then three seeds at each winner, with edge winners extended.], [`experiments12.py`, `experiments13.py`],
  [exp14, exp16], [Can anything close EGGROLL's gap to the paper's? Noise scale extended downward; rank-shaped fitness; float32.], [`experiments14.py`, `experiments16.py`],
  [exp15], [EGGROLL's alignment at 65,536 and 131,072 passes: the paper's claim of under 0.05 at 128k.], [`experiments15.py`],
  [exp17], [The 4,096-population rung for both methods, one seed: does EGGROLL's ladder stay straight?], [`experiments17.py`],
  [exp18], [Is DUST's closeness to backprop a learning-rate-group artefact? Backprop with DUST's rates for the value embeddings and the residual scalars, three seeds.], [`experiments18.py`],
  [exp19], [Backprop under Adam and under Muon: three matrix learning rates each, seed 42.], [`experiments19.py`, `lib/optim.py`],
  [exp20, exp21, exp22], [DUST under Adam and Muon: three matrix learning rates at 64 draws (exp20) and 256 draws (exp21 Adam, exp22 Muon), seed 42.], [`experiments20.py` to `experiments22.py`],
  [exp23], [EGGROLL with Muon on the hidden matrices: three rates at 64 and 256 passes, seed 42.], [`experiments23.py`],
  [exp24], [Each winner of exp19 to exp22 repeated on seeds 42, 43 and 44: backprop, DUST at 64 and 256 draws, Adam and Muon.], [`experiments24.py`],
  [exp25], [DUST at 1,024 draws under Adam and Muon, one seed, at the 256-draw winner's rate.], [`experiments25.py`],
  [exp26], [EGGROLL under Muon: 64 and 256 passes on three seeds, 1,024 on one, at the winner of exp23.], [`experiments26.py`],
  [exp27], [DUST at 4,096 draws under Muon, one seed, at the 256-draw winner's rate.], [`experiments27.py`],
  [exp28], [Rate check for DUST at 1,024 draws under Muon: 0.002 and 0.005 against exp25's 0.003, seed 42.], [`experiments28.py`],
)

Parity with the authors' code is a separate check, not a numbered run: `scripts/tmp/parity_export.py` and
`parity_torch.py` feed identical randomised weights and a fixed batch to both implementations, and one
row of `results.jsonl` records the outcome.

== Controls

*Positive controls.* A comparison of two zeroth-order methods is only readable if the pipeline can
succeed at all. Three things must hold before the ladder means anything, and each is a run above.
Backprop on our model must reach the paper's backprop loss (exp1): if it did not, the model, the data
or the optimiser would be wrong, and no difference between DUST and EGGROLL could be attributed to the
methods. DUST at the authors' settings must reach the authors' number (exp3): if it did not, our port of
the estimator would be suspect. And the two estimators, ours and theirs, must agree on identical weights
(parity). Had any failed, the ladder would have been a measurement of the port.

*The negative control for the metric.* The unigram floor of 6.994 (@sec-task) is the loss of a
model that has learned no context. Every ladder carries it. A method sitting above it has not yet
learned what a lookup table knows.

*What the alignment run would have shown the other way.* If DUST's cosine to backprop did not rise
with population, or rose no faster than EGGROLL's $sqrt(P slash d)$, then DUST's advantage in the ladder
could not be explained by a better gradient estimate, and would need another explanation.

*Tuning is fair or the comparison is not.* Backprop, DUST and EGGROLL are each tuned on validation
loss, on one seed, and the winner is then repeated. The released DUST and backprop settings stand in for
that step where they exist. exp7 checks that our backprop is no worse tuned than the released one; exp8
checks that EGGROLL is no stronger under the paper's Adam constants than under ours.
