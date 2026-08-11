#import "/template.typ": *

// OBLIGATIONS
//  - A table mapping each run to the question it answers and the file that
//    produced it: expN | question | source file. This is the bridge between
//    the paper and the repo, and the only place experiment numbers belong.
//  - State the controls. A positive control (something that MUST succeed if
//    the pipeline works) is what separates "the model cannot do this" from
//    "the harness was broken" — without one, a negative result is unreadable.
//  - Say what each experiment would have shown had it come out the other way.

= Experiments

#table(
  columns: (auto, 1fr, auto),
  [Run], [Question], [Source],
  // [exp1], [Does the baseline learn the task at all?], [`experiments1.py`],
)

#todo[
  Fill the table, and describe the controls below it.
]

== Controls

#todo[
  Positive and negative controls, and what each rules out.
]
