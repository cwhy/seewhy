#import "/template.typ": *

= Experiments

#table(
  columns: (auto, 1fr, auto),
  [Run], [Question], [Source],
  [exp1], [Do backprop, PC and PC-ALM reproduce the reported accuracies and the 40x training-time gap on the deep-narrow residual network?], [`experiments1.py`],
  [exp2], [Does Forward-Forward reproduce its reported accuracy — and what does backprop do on Forward-Forward's own architecture?], [`experiments2.py`],
  [exp3], [Is any method being held back by a learning rate chosen for another?], [`experiments3.py`],
  [exp4], [At equal training wall-clock rather than equal epochs, what is the ordering?], [`experiments4.py`],
  [exp5], [How does each rule behave as depth grows from 2 to 64 layers?], [`experiments5.py`],
)

Every cell is one (rule, architecture, seed) run, written to `results.jsonl`
with its full hyperparameters, timings and per-epoch curve. Cells are skipped if
already present, so a crashed sweep resumes rather than restarting.

The five arms of the headline comparison are: the three rules on the residual
network at $N = 32$, $L = 32$ (56,128 parameters), Forward-Forward on its own
two-layer network at 66 units (56,232), and two arms the comparison being
replicated does not have — backpropagation on Forward-Forward's architecture
(56,902) and Forward-Forward on a 31-layer stack of 32 units (56,800).

== Controls

*Positive control.* Backpropagation on the two-layer plain network. This is an
ordinary MNIST classifier trained in the ordinary way; if the harness, the data
pipeline or the timing code were broken, it is the arm that would show it,
because its expected behaviour is known independently of anything in this paper.
A one-epoch accuracy in the low nineties means the pipeline is sound and every
other arm's number is about the method, not the plumbing.

*The control that isolates the dual variables.* Standard predictive coding, at
identical $T$, $eta_h$, architecture and seed, differing from PC-ALM only in
$alpha = 0$. Any accuracy difference between the two is attributable to the
Lagrange multipliers and to nothing else. Had PC matched PC-ALM here, the
paper's mechanism would have no support at this scale.

*The control that isolates architecture from learning rule.* The comparison
being replicated matches parameter counts across three rows whose architectures
differ — 32 layers of 32 units against 2 layers of 66. Running backpropagation
on the second architecture, and Forward-Forward on the first, crosses the two
factors. If Forward-Forward's advantage is a property of the rule it should
survive the transplant; if it is a property of the shape, backpropagation should
inherit it.

*What a different outcome would have shown.* Had backpropagation on the
two-layer network landed at Forward-Forward's accuracy rather than above it, the
architecture explanation would be dead and the ranking in the original table
would stand as a statement about learning rules. Had PC-ALM at depth 32 matched
PC, the replication would have contradicted the source paper rather than
supporting it.
