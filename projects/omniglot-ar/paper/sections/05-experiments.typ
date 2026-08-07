#import "/template.typ": *

= Experiments <sec:exp>

Five runs, each changing exactly one thing from a named predecessor, so that
whichever change moves the number is identified. All share the architecture,
optimiser, schedule and 12 000-step budget of §#link(<sec:method>)[4].

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, 1fr),
    [*run*], [*N*], [*C*], [*bins*], [*label field*], [*change from*],
    [exp1], [5], [196], [8], [—], [the baseline setup],
    [exp2], [2], [392], [8], [—], [exp1: easier task, difficulty floor],
    [exp3], [5], [196], [8], [yes], [exp1: label on every support token],
    [exp4], [5], [196], [2], [yes], [exp3: binarised values],
    [exp5], [5], [196], [8], [yes], [exp3: ink-biased observation pool],
    [exp6], [5], [100], [8], [yes], [exp3: 10×10 image, fully observed],
    [exp7], [5], [196], [8], [yes], [exp3: query image #emph[is] its support],
  ),
  caption: [The experiment chain. $N$ is classes per episode, $C$ observed
    pixels per drawing. Every row differs from its stated predecessor in one
    respect only.],
) <tab:chain>

== exp1 — the target task

5-way 1-shot, 196 of 784 pixels observed from a pool shared across the episode's
drawings, trained on background characters and evaluated on evaluation
characters. This is the direct test of the claim in §#link(<sec:task>)[3] and the
control for everything after it.

== exp2 — the difficulty floor

A diagnostic that separates two very different readings of a chance result: the
circuit is learnable but 5 classes over 196 sparse pixels is too hard a first
target, or the formulation cannot learn cross-drawing matching at all and making
the task easier changes nothing.

So everything that can be made easier is: two classes instead of five, raising
chance to 0.500, and 392 pixels instead of 196, roughly doubling the ink each
drawing exposes. Architecture, optimiser and budget are untouched.

== exp3 — the label field

The intervention argued for in §#link(<sec:analysis>)[6]: carry the class on
every token of a support drawing rather than only on its label token. Identical
to exp1 in every other respect, so exp1 is its control and the label field is
the only variable.

== exp4 — binarised values

exp3 with two intensity bins instead of eight. At 18.7% ink, most tokens carry
bin 0 under an 8-bin vocabulary, so agreement between two drawings is dominated
by their agreeing about blank background. Two bins makes "agrees on ink" and
"agrees on background" the only outcomes, which is the sharpest form of the
comparison the exp3 circuit must compute.

== exp5 — ink-biased observation

exp3 with the observed positions drawn from where the *support* drawings have
ink, rather than uniformly over all 784. This attacks the same
signal-to-noise problem as exp4 but from the sampling side rather than the
vocabulary side: instead of sharpening what a background match looks like, it
stops spending ~160 of the 196 tokens on background at all.

The pool is derived from support drawings only. Deriving it from the queries
would let the pool itself carry information about the drawings being classified.
The nearest-neighbour floor is recomputed under the same pool, so it remains a
floor for this run rather than for exp1's more permissive uniform sampling — the
two runs' baselines are not comparable to each other, and neither is quoted
across rows.

== exp6 — coarse and complete

exp3 at $10 times 10$ with all 100 positions observed. This halves $C$, which is
the factor diluting each scored token's gradient on the way back through the
label token's pooling step, and separately removes partial observation as a
confound: every drawing shows its whole content, so nothing is hidden from the
match.

== exp7 — the positive control

Each query drawing #emph[is] its class's support drawing. Matching is then
exact: at every observed position the query's value equals its support's, and
nearest neighbour scores 1.000 by construction.

This is not a generalisation test — an identity query leaks its own answer, and
its unseen-character accuracy is not a claim about anything. It asks one
question: can the apparatus fit a task where the match is free? A chain of
negative results is only worth reading if the answer is yes.

== Localising controls

exp1–exp7 vary the task. The controls in §#link(<sec:analysis>)[7.4] instead
vary #emph[how much of the answer is given], leaking the target into the query's
own tokens in stages, to find which single step in the circuit fails. They use a
deliberately small setting — two classes, 16 observed pixels, binary values, a
0.5 M-parameter two-layer model — because their purpose is to establish what the
apparatus #emph[can] do, and a smaller apparatus makes that a stronger
statement.
