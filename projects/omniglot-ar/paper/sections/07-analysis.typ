#import "/template.typ": *

= Analysis <sec:analysis>

== What the failure signature rules out

The exp1/exp2 pair eliminates four candidate explanations between them.

#figure(
  table(
    columns: (auto, 1fr),
    align: left,
    [*explanation*], [*status*],
    [Memorised class prototypes],
    [*Ruled out.* Seen and unseen characters both score chance. There is no
     memorisation gap because there is nothing memorised to produce one. (Note
     that with per-episode slot assignment there is nothing memorisable in
     principle, which is why train accuracy is at chance too.)],
    [The information is not in the tokens],
    [*Ruled out.* Pixel 1-NN over exactly the same observed pixels reaches 0.431
     and 0.664, and 1.000 in exp7. A cosine distance extracts what the model
     does not.],
    [The task is too hard a first target],
    [*Ruled out by exp2 and exp7.* Halving the classes, doubling the pixels, and
     finally making the query a literal copy of its support all moved nothing.],
    [The label pathway is broken],
    [*Ruled out.* From the first evaluation onward the open-vocabulary and
     slot-restricted accuracies are exactly equal, so the query token reaches
     the head and vocabulary routing works.],
    [The matching step itself],
    [*What remains* — and §7.4 pins it down directly.],
  ),
  caption: [Elimination across the seven runs.],
)

The model learned, almost immediately, to emit only values that are
syntactically legal in a label position — five of a fourteen-value vocabulary —
and then stopped. The loss is not a model making wrong guesses; it is a model
making #emph[no] guess, having correctly identified that under its
representation the slots are indistinguishable. It learned the #emph[form] of
the answer and nothing about its content.

== The circuit the token layout demands

Under the exp1 layout, where the class appears only on label tokens, a query's
label token must:

+ *Gather* — attend to the pixel tokens sharing its own `ref`;
+ *Match* — compare those against other drawings' pixels, position by position;
+ *Route* — recover #emph[which] `ref` won and read the label token carrying it.

Hop 3 looked like the culprit. To compare drawings at a position, attention must
attend across the tokens sharing that position, and a softmax-weighted sum
returns their #emph[average]: whichever drawing agreed, the output is a blend,
and the winner's identity is destroyed by the very operation that must compute
it. A model could only recover it by dedicating heads to particular `ref` tags,
but `ref` is re-drawn every episode, so no head can specialise.

== The label field, and its refutation

That reasoning predicted a fix: carry the class on #emph[every] token of a
support drawing, so a query pixel can read a #emph[label] embedding directly
rather than a drawing identity. What gets averaged becomes the label, which is
exactly the soft vote the task needs. Three hops become two.

#callout(title: [The prediction failed])[
  exp3 implements this and lands at 0.188 against chance 0.200. exp4 (binarised
  values), exp5 (ink-biased pool) and exp6 (coarse, fully observed) add further
  help on top and land at 0.169, 0.181 and 0.203. exp7 makes the match exact —
  the query drawing #emph[is] its support drawing — and lands at 0.191 with its
  own 1-NN floor at 1.000.
  The hop count was not the binding constraint.
]

== Localising the failure directly

Rather than infer further, we removed the matching step in stages, leaking the
answer into the query's own tokens by increasing amounts. Two classes, 16
observed pixels, binary values, identity queries, a 0.5 M-parameter two-layer
model, 1500 steps.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    [*condition*], [*what the model must do*], [*accuracy*],
    [`self`], [Read one field of its own label token.], [*1.000*],
    [`own-pixels`],
    [Attend to the tokens sharing its `ref`, pool over 16 of them, read a field.],
    [*1.000*],
    [`none`],
    [The real task: match its pixels against another drawing's, then pool.],
    [0.500],
  ),
  caption: [Positive controls. `self` and `own-pixels` both reach 1.000 with the
    loss at zero inside 300 steps. Chance is 0.500.],
) <tab:controls>

This is the central result of the paper, and it is sharp:

- The loss, the target, the forward pass and the output head are correct — a
  broken pipeline cannot drive a loss to zero.
- *`ref`-keyed attention works.* The model readily learns to attend to the
  tokens sharing a tag and pool a field across them. The "gather" hop is not the
  problem, and neither is the `ref` mechanism that the whole formulation rests
  on.
- What never emerges is *content-dependent matching*: attending to a token
  because #emph[its value resembles mine].

The distinction matters. `ref` matching is a lookup against a fixed learned
embedding — the query for a tag can be learned once and reused. Content matching
requires the attention score to be high exactly when key and query values
#emph[agree], for every value, which is a bilinear identity over the value
embedding space. It is expressible by the architecture. It is not reached by
gradient descent from random initialisation on this signal.

== It does emerge — the plateau is crossable

The claim that content matching is unreachable is *false*, and a further
diagnostic refutes it. Pushing every knob toward the easy end — two classes,
identity queries, an ink-biased pool, eight bins, three layers, batch 64,
learning rate $10^(-3)$, 6000 steps — produces this:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (right, right, right),
    [*step*], [*loss*], [*accuracy*],
    [1000], [0.6934], [0.492],
    [2000], [0.6930], [0.461],
    [3000], [0.1033], [0.961],
    [5000], [0.0007], [1.000],
  ),
  caption: [`identity/ink/196`: flat at $ln 2 = 0.693$ for 2000 steps, then an
    abrupt transition to a solved task. The same run at 64 observed pixels
    reaches 1.000 as well.],
) <tab:transition>

This is a textbook phase transition, and it changes the reading of everything
above. The circuit is not merely expressible — it is *learnable*, and it appears
suddenly after a long flat stretch, exactly as induction-head formation does
@olsson2022.

== What exp1–exp7 actually got wrong

The difference between exp7 (chance at 12 000 steps) and the run in
@tab:transition is not the step budget. It is the resources for crossing the
plateau:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    [], [*exp1–exp7*], [*the run that learns*],
    [effective batch], [16], [64],
    [learning rate], [$3 times 10^(-4)$], [$10^(-3)$],
    [observation pool], [uniform (exp5 aside)], [ink-biased],
    [steps], [12 000], [6 000],
  ),
  caption: [What separates a run that crosses the plateau from one that does
    not. Note the successful run uses *half* the steps.],
)

Crossing a plateau is a signal-to-noise problem: the gradient pointing toward
the matching circuit is small but not zero, and whether it is visible above the
minibatch noise is set by batch size and step size, not by how long one waits at
too small a batch. exp1–exp7 were under-resourced, not under-trained — and no
amount of the interventions in §7.3 could compensate, which is why none of them
moved anything.

#callout(title: [Correction])[
  An earlier version of this analysis concluded that content-dependent matching
  "is not reached by gradient descent from random initialisation on this
  training signal". That conclusion was drawn from seven runs that shared a
  batch size and learning rate, and it is wrong. The controls in @tab:controls
  remain valid — they localise *which* capability is missing — but the
  explanation for why it was missing was an optimisation deficit, not a
  structural one.
]

== The exact / approximate dissociation

Matching emerges on #emph[identity] queries and solves completely. On the real
task it does not, and this is now well tested rather than assumed:

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, right),
    [*run*], [*matching required*], [*1-NN*], [*model*],
    [`identity/ink/196`], [exact — query values equal support values], [1.000], [*1.000*],
    [exp8], [approximate, 28×28, 196 px], [0.729], [0.488],
    [exp10], [approximate, 10×10 fully observed], [0.805], [0.488],
  ),
  caption: [Same recipe, same budget class. Exact matching solves in 3000 steps;
    approximate matching is flat at $ln 2$ after 25 000, even where nearest
    neighbour reaches 0.805.],
) <tab:dissoc>

exp10 was the sharpest available test of "approximate is just far-away exact":
at $10 times 10$ fully observed, two drawings of a character differ in far fewer
pixels, and the 1-NN floor rises from 0.729 to 0.805 accordingly. It made no
difference. The two are not the same problem with a difficulty knob between them.

== Why the margin collapses

The circuit the label field enables computes, for each observed position, a soft
vote: attend to the tokens at that position, weight by value agreement, read the
label. The query's label token then #emph[averages] those votes.

That average is where approximate matching dies. Write $a^+$ for the fraction of
positions at which the correct support agrees with the query, and $a^-$ for a
wrong one. The pooled signal is proportional to $a^+ - a^-$.

- *Exact matching*: $a^+ = 1$ by construction, while $a^- approx 0.7$ (two
  unrelated drawings still agree wherever both are background). Margin ≈ 0.3
  against a clean maximum.
- *Approximate matching*: $a^+ approx 0.8$, $a^- approx 0.7$. Margin ≈ 0.1, and
  it is a small difference between two large, noisy sums both dominated by
  shared background.

Nearest neighbour succeeds on exactly this data because it does something the
token circuit cannot: it #emph[normalises] the comparison and takes an argmax
across candidates — a global operation over whole drawings. The token-level
circuit can only accumulate per-position agreement additively, and an additive
accumulator cannot recover a 0.1 margin buried in a background-dominated sum.

This is a limitation of the #emph[representation], not of the optimiser, and it
explains why the task-side interventions all failed in a way the earlier
hop-counting story did not: an ink-biased pool (exp5) raises $a^+ - a^-$ a
little, binarising (exp4) sharpens what agreement means, and coarsening (exp6,
exp10) raises both — none changes the fact that the margin is a difference of
sums rather than a normalised comparison.

== The remaining gap

What would follow from the account above is a comparison that is normalised and
global rather than additive and per-position: a pooled per-drawing
representation to compare against, or an explicitly normalised similarity in the
score function. Both amount to reintroducing a representation of a #emph[sample]
— which is precisely the structure the token-level premise set out to dissolve.
That tension, rather than any single accuracy number, is this project's result.
