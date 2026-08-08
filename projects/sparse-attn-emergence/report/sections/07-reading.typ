#import "/template.typ": *

= How to read this

== What holds

*Emergence is abrupt and its timing is seed-random.* Two independent 16-seed samples of the
same configuration put median $t^*$ at 885 and 923 steps with spreads of 5.4× and 4.0×, and the
effect reappears on the content-keyed task (7 of 16 seeds, median $t^* approx 26 900$). This is
the paper's central claim and it is the portable one.

*The jump is the pattern being found.* Deleting the head whose attention matches the target
matrix takes second-half loss from 0.0000 to 4.23 nats — six times the no-knowledge plateau,
i.e. confidently wrong rather than ignorant — while deleting the least-aligned head costs 0.08.

*Difficulty is quantitative.* Learnability tracks $C(S, s)$, with a threshold that holds across
context lengths.

*The cellular-automata task is genuine in-context learning*, not memorisation of a rule pool.

== What does not

*The architecture ranking is task-family-specific and reverses.* Mixer > transformer > KDA on
position-keyed routing; KDA > transformer > mixer on content-keyed routing, with the mixer
unable to do the latter at all. Any statement of the form "architecture X learns sparse
attention patterns better" needs the kind of pattern attached to it.

*"Both extremes of sparsity are easy" is partly an artifact.* At $s = S$ every row of $A$ is
all-ones, so every output token is the same value and a model can score $1 - 0.5\/S$ by copying
its own previous output, leaving exactly $ln 2 \/ S$ loss — measured 0.0433 at $S = 16$ against
$ln 2\/16 = 0.04332$, and confirmed per position (0.488 on the first output token, 1.000 on the
other fifteen). The paper shares this construction.

*More examples do not help.* Packing 7 worked examples of the same map into one sequence
instead of 1 makes it 8× slower at $s = 3$ and turns a half-solvable cell unsolvable — while
delivering *more* supervised targets per step. With absolute position embeddings, more examples
means more separate patterns to find, not more evidence for one.

== Limits

One layer and $D = 128$ for the linear map, two to four for the others; $S <= 32$; 10k–30k
steps; 8–16 seeds; three architectures rather than the paper's seven; the real-language-model
half (Pythia, IOI) out of scope entirely. Individual $t^*$ values carry ~8% run-to-run noise
because GPU reductions are not bit-deterministic — nothing here rests on differences that small.

*The difficulty law is position-keyed too.* $C(S,s)$ governs the linear map and is essentially
uncorrelated with difficulty on k-of-m recall, where what matters is how discriminable the match
is. Sparser is easier for positional patterns and harder for content-keyed ones — opposite
directions, different governing quantities.

== The next experiment

Whether *architecture* interacts with the content-keyed difficulty axis the way it does with the
positional one. KDA was the best arm on induction and the worst on the linear map; if difficulty
there is set by match discriminability rather than candidate count, KDA's advantage should widen
as $k$ falls — the regime where the match is hardest to pick out. That is a sharper prediction
than anything measured here, and one sweep answers it.

#callout(title: [Reproducing])[
  Code, one file per experiment, and a `results.jsonl` row per configuration carrying
  hyperparameters and per-seed curves:
  #link("https://github.com/cwhy/seewhy")[`projects/sparse-attn-emergence`]. Every seed of a
  configuration trains simultaneously under one `jax.vmap`, so 16 seeds cost about what one
  costs — a full 16-seed, 10 000-step run of the base configuration takes 167 seconds.

  A companion web report covers the full replication claim by claim, including a page listing
  every error made along the way and how each was caught.
]
