#import "/template.typ": *

= Results

#callout(title: [Chance, and flat throughout])[
  Neither run cleared chance at any point in 12 000 steps. Across all 49
  evaluations, unseen-character accuracy stayed within $[0.150, 0.228]$ for
  exp1 (chance 0.200) and $[0.453, 0.539]$ for exp2 (chance 0.500) — every
  value inside noise. The loss reaches $ln N$, the entropy of a uniform guess
  over the label slots, by step 250 and never leaves. Making the task easier
  changed nothing.
]

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*run*], [*task*], [*chance*], [*pixel 1-NN*], [*model, unseen*], [*model, seen*],
  [exp1], [5-way 1-shot, 196 px], [0.200], [0.431], [0.209], [0.203],
  [exp2], [2-way 1-shot, 392 px], [0.500], [0.664], [0.531], [0.500],
)

Evaluation is 64 episodes — 320 queries for exp1, 128 for exp2 — so the
standard error is about 0.022 and 0.044 respectively. exp2's 0.531 is
three-quarters of a standard error above chance and 0.13 #emph[below] the
nearest-neighbour floor; it is noise, not a signal.

exp2 exists to separate difficulty from structure: two classes instead of five,
and twice the pixels per drawing. It landed on chance just as flatly. Whatever
is failing is not a difficulty ceiling that a gentler first target would clear.

#fig(
  include "/figures/learning_curves.typ",
  caption: [exp1, 5-way. The three series answer three different questions:
    #emph[train] is whether it is fitting at all, #emph[seen] is how much of
    that is memorisation, #emph[unseen] is the actual claim. All three sit on
    the chance line for the entire run.],
)

#fig(
  include "/figures/learning_curves_exp2.typ",
  caption: [exp2, 2-way with double the observed pixels — the easier task,
    with the same outcome.],
)

#fig(
  include "/figures/loss_curve.typ",
  caption: [exp1 cross-entropy on the masked query-label tokens. It reaches
    $ln 5 approx 1.609$ almost immediately and never leaves.],
)

#fig(
  include "/figures/floor_comparison.typ",
  caption: [Each run against its own floors. Chance differs between the runs,
    so the bars are grouped per run — the question is never which run scored
    higher, but whether each cleared the floor it had.],
)

== What the model did learn

Not nothing. From the first evaluation after initialisation — step 250, in both
runs — the open-vocabulary accuracy equals the slot-restricted accuracy
#emph[exactly], and stays equal for every one of the remaining 47 evaluations.
That means the model always answers a label query with a #emph[label slot] and
never with a pixel bin: out of a 14-value vocabulary it learned, almost
immediately, to emit only the values that are syntactically legal in that
position. It learned the #emph[form] of the answer and nothing about its
content.

That is a sharper failure than "it did not learn". The loss is not a model
making wrong guesses; it is a model making no guess, having correctly
identified that under its representation the five slots are indistinguishable.

== What this rules out

#table(
  columns: (auto, 1fr),
  [*explanation*], [*status*],
  [Memorisation of class prototypes],
  [Ruled out. Seen and unseen characters both score chance — there is no
   memorisation gap because there is nothing to memorise.],
  [The information is not in the tokens],
  [Ruled out. Pixel 1-NN over exactly the same observed pixels reaches 0.431
   and 0.664. A cosine distance extracts what the model does not.],
  [The task is too hard a first target],
  [Ruled out by exp2 — halving the classes and doubling the pixels moved
   nothing.],
  [The label pathway is broken],
  [Ruled out. The model reliably emits label slots, so the query token reaches
   the head and the vocabulary routing works.],
)

What remains is the cross-drawing content-matching circuit itself: aggregating
196 pixel tokens per drawing, comparing them position-wise against other
drawings, and routing the winner's label back to the query. Every component
except that one is demonstrably working.

== What this does not establish

The budget was 12 000 steps at an effective batch of 16 — about 192 000
episodes, with 3.4 M parameters. Circuits of this shape (aggregate, match,
copy) are exactly the kind known to appear abruptly after long flat stretches,
and nothing here rules out emergence at 10× the budget or width. The honest
statement is *did not emerge under this budget*, not *cannot emerge*.

#callout(title: [Where this leaves the premise])[
  `proposal.md` predicted that if Omniglot failed too, the finding would be
  about the token-level formulation rather than about MNIST. That is where we
  are. The substrate argument was sound — memorisation really is impossible
  here, and the two-way control really is easy — and the formulation still did
  not learn. The next move is the ablations in exp5, not the sweep in exp2 as
  originally planned.
]
