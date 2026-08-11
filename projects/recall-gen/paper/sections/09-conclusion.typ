#import "/template.typ": *

// OBLIGATIONS
//  - What is now known that was not known before. Two paragraphs.
//  - No new evidence here.
//  - The reproduction appendix is not optional.

= Conclusion

Trained on retrieval alone, a model with a fixed-size memory learns a retrieval
mechanism that is genuinely general — it works on images it has never seen, and
on digit classes it has never seen, at perfect identification accuracy. That is
the whole of what it learns. Asked to complete an image that is not in its
context, the same model does worse than a linear regression fitted on its own
training data, and it gets worse at it as its retrieval improves. Training on a
mixture instead reaches the full completion ceiling while keeping most of the
retrieval, so this is not two abilities competing for one memory. A retrieval
objective simply never asks for the other one, and at the context sizes where
retrieval is possible there is nothing in the context to ask for: the best
achievable use of sixteen context images, on a query they do not contain, scores
exactly as well as ignoring them.

The apparent counter-example dissolves on inspection. Enlarging the context does
improve the model's completions — and it improves them by destroying the
retrieval. At 256 context images the recall-trained model gains nothing from
having the answer in front of it (0.004, against 0.835 at 16), scores four times
better on images from its training pool than on novel ones, and lands on exactly
the ceiling that a completion-trained model reaches. Two opposite objectives
converge on one solution: memorise the distribution in the weights and ignore
the context. Shrinking the memory while holding the context fixed reproduces the
same trade, which is what rules out the alternative explanation that a larger
context was simply more informative. Generalisation did not emerge from
retrieval here; it appeared in the place retrieval vacated.

Two directions follow specifically. The digit-split result — perfect retrieval
and chance-level completion on the same unseen digits — is the cleanest
separation in the paper and deserves a harder dataset, where retrieval is not
nearly free. And the state-size sweep stops short of the collapse: it changes
the trade in direction but never breaks retrieval, so where exactly the
transition sits, and whether it is sharp, is unmeasured.

= Appendix: reproduction <appendix-repro>

#kv(
  ("commit", "b2303c6"),
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
