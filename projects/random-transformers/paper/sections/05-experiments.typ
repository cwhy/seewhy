#import "/template.typ": *

= Experiments <sec-experiments>

Seven experiments, each answering one question. Each stage's outcome set the
next stage's configuration, so they are listed in the order they were run.

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  table.header([*Run*], [*Question*], [*Source*]),
  [exp1], [Do random transformers do the four algorithmic tasks, and does width matter?], [`experiments1.py`],
  [exp2], [Which of the three embedding matrices actually have to be trained?], [`experiments2.py`],
  [exp3], [What does accuracy look like *between* width 16 and width 1024?], [`experiments3.py`],
  [exp4], [How much arbitrary information fits in the trainable parameters?], [`experiments4.py`],
  [exp5], [Is the mechanism a low-dimensional subspace, or a sparse subnetwork?], [`experiments5.py`],
  [exp6], [Can a random transformer imitate a circuit that needs many dimensions?], [`experiments6.py`],
  [exp7], [Can a random transformer model natural language?], [`experiments7.py`],
)

== Controls

*The positive control is the `full` condition, and it is in every cell.* Every
random-transformer number has a fully trained number beside it from the same
harness, same task encoding, same budget, same seeds. This is what separates "the
random model cannot do this" from "our implementation of this task is broken":
if a cell fails in both conditions, the task is at fault, not the freezing.

*The negative control is the chance level*, derived in @sec-task rather than
assumed, and printed beside every accuracy. It is what makes the parenthesis
results readable, since chance there is 69.4% rather than something negligible.

*A third control is on the task encodings themselves.* An incorrectly encoded
task trains without complaint and reports a plausible-looking number. Before any
experiment ran, and after every change to the task code, a verification script
decoded samples from each task and checked them against an independently written
implementation of the rule — that the decimal output really is the sum of the
inputs, that the needle query really resolves to its marker's value, that the
parenthesis label agrees with a hand-written Dyck check, that the modular
addition train and test splits are disjoint. All checks pass; they are what the
rest of the paper rests on.

== Deliberate design choices

*exp5 trains nothing.* It loads the parameters saved by exp1 and exp4 and
measures their activations. This is deliberate: an analysis that retrains its
own models can silently end up describing different models from the ones the
results tables describe.

*exp1 and exp3 share their endpoints.* The width sweep runs widths 32 through
512; widths 16 and 1024 come from exp1, which ran them at more seeds. The
figures join the two.

*exp6 is the falsification test.* Subspace selection predicts that random
transformers fail specifically when the target computation needs many
dimensions. Circuit imitation is constructed so that the number of dimensions
the target needs is a knob we turn. A subspace-selection story that did not
predict degradation there would not be testable.
