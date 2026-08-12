#import "/template.typ": *

// OBLIGATIONS
//  - What is now known that was not known before. Two paragraphs.
//  - No new evidence here.
//  - The reproduction appendix is not optional.

= Conclusion

Trained on retrieval alone, a model with a fixed-size memory learns a similarity
metric and nothing else. The metric is genuinely general in one direction — it
retrieves images it has never seen, and digit classes it has never seen, at
perfect identification — and it is not general in another: it drops to 0.651 on
Fashion-MNIST and to 0.116, near chance, on MNIST images whose pixels have been
permuted, which are the same pixels with the same statistics and the same
pairwise distances. Nothing the model learns is free of its training data. What
distinguishes the two things it *could* learn is granularity: individual images,
which transfer nowhere, or a metric over the distribution, which transfers within
it. Asked to complete an image that is not in its context, the model does worse
than a linear regression fitted on its own training data, and gets worse at it as
its retrieval improves. Training on a mixture reaches the full completion ceiling
while keeping most of the retrieval, so this is not two abilities competing for
one memory — a retrieval objective simply never asks for the other one, and at the
context sizes where retrieval is possible there is nothing in the context to ask
for: the best achievable use of sixteen context images, on a query they do not
contain, scores exactly as well as ignoring them.

The apparent counter-example dissolves twice over. Enlarging the context does
improve completion — and it does so by destroying the retrieval: at 256 context
images the model gains 0.004 from having the answer present, against 0.835 at 16,
scores four times better on images from its training pool than on novel ones, and
lands on exactly the ceiling a completion-trained model reaches. Shrinking the
memory at fixed context reproduces the same trade, which rules out the reading in
which a larger context is simply more informative. And the effect does not exist
at inference at all: handed 256 images, the model trained at 16 gets *worse*
(0.942, against 0.561 for a model trained there), while the model trained at 256
does not recover retrieval when handed a context small enough to fit. The two
solutions are separate attractors, chosen at training time.

Three directions follow specifically. The digit-split result — perfect retrieval
and chance-level completion on the same unseen digits — is the cleanest
separation here and deserves a harder dataset where retrieval is not nearly free.
The transfer numbers give a broader-training programme its targets: 0.651 is the
soft one, and 0.116 the hard one, since moving it requires a similarity metric
not tied to spatial layout — which may not even be desirable, spatial structure
being real information. And the sharpest untested question is whether the recall
objective can be made to yield knowledge at all: give the context pool unlimited
fresh images so that memorising individual ones is impossible, and at $M = 256$
the model must either learn a transferable prior or fail outright. Which of those
happens is the number this paper most conspicuously lacks.

= Appendix: reproduction <appendix-repro>

#kv(
  ("commit", "122bb8c"),
  ("hardware", "one NVIDIA RTX 4090; every run under 25 minutes"),
  ("environment", "uv run --no-sync python  (on the GPU box)"),
  ("launch", "python projects/recall-gen/scripts/run_experiments.py --bg exp1"),
  ("baselines", "python projects/recall-gen/scripts/baselines.py --M 16"),
  ("figures", "python projects/recall-gen/scripts/gen_report.py"),
)

Every number in this paper comes from `projects/recall-gen/results.jsonl`. Each
row carries its full hyperparameter set, parameter count, wall-clock, the
per-condition final metrics, and the per-evaluation curves, so every figure here
can be redrawn without rerunning anything.

#align(center, table(
  columns: 2, stroke: none, align: left, inset: 5pt,
  table.hline(stroke: rule),
  [*claim or figure*], [*rows*],
  table.hline(stroke: rule),
  [retrieval transfers to novel images], [`exp1`, `exp10`, `exp11`],
  [completion decays with training (divergence figure)], [`exp1`],
  [completion ceiling], [`exp2`, `exp7`, `exp12`],
  [the two abilities do not compete], [`exp3`],
  [context-size sweep (context-size figure)], [`exp6`, `exp1`, `exp4`, `exp5`],
  [state-size control (state-size figure)], [`exp1`, `exp15`, `exp16`, `exp17`],
  [digit split], [`exp8`, `exp9`],
  [fine-tuning probe], [`exp13`, `exp14`],
  [all model-free reference points],
  [`baselines_M{4,16,64,256}_r14`, `baselines_M16_r14_split`],
  table.hline(stroke: rule),
))
