#import "/template.typ": *

= Tasks and data <sec-task>

Seven tasks. Four are the algorithmic core, one measures raw storage, one is a
diagnostic built to have a tunable dimension, and one is natural language. Every
encoding below is the paper's, from its Appendix D.1.

All tasks are posed as next-symbol prediction over a single sequence, so one
model and one loss covers all of them. A *scored mask* marks which positions
count: at a scored position $t$, the model must predict the symbol at $t+1$.
Prompt positions and padding are excluded from both loss and accuracy.

== The four algorithmic tasks

*Modular addition.* The sequence is $[a, b, (a+b) mod 199]$ with
$a, b in [0, 198]$, and the final symbol is scored. The vocabulary is the 199
residues. All $199^2 = 39,601$ pairs are enumerated and split once, 95%/5%, into
37,621 training and 1,980 test pairs, with the same split for every run. This is
the only task with a genuine train/test split, so it is the only one where
accuracy measures generalisation rather than in-distribution performance.

*Needle in a haystack.* A list of marker–value pairs followed by a query:
$[m_1, c_1, m_2, c_2, dots, m_k, c_k, m_u]$, with the answer $c_u$ scored. The
number of pairs $k$ is uniform on $[1, 30]$. Values are uniform in $[1, 127]$.
Markers are $k$ *distinct* integers drawn from $[128, 157]$. The query symbol is
the asked marker plus $30$, so it is never literally equal to the marker it
refers to. The model must locate the earlier occurrence and read
off what followed it. Vocabulary 256, sequences padded to 62.

*Decimal addition.* Two ten-digit numbers, digits reversed so that carries
propagate left-to-right in reading order. Input digits are symbols 0–9, `+` is
10, `=` is 11; output digits are symbols 20–29 and 30 terminates the output. So
$39 + 71 = 110$ would be written (at two digits) as input `9 3 1 7` and output
`0 1 1`. Every output symbol is scored, including the terminator — 11 or 12
positions depending on whether the sum carries into an eleventh digit.
Vocabulary 31, sequence length 34.

*Parenthesis balancing.* A sequence of up to 60 parentheses followed by `?`,
whose position is scored; the answer is one of two symbols meaning balanced or
not. Balanced means equal counts *and* no prefix in which closers exceed
openers. Vocabulary 4 — two parenthesis symbols and two answer symbols, with the
answer symbols reusing the same integers. This is the extreme case for
embedding-only training: with four token embeddings and eighty positional ones,
there is very little to optimise.

Uniformly random parenthesis sequences would almost all be unbalanced, making
the task trivial. So we follow the paper's generate-then-mutate recipe. With
probability $1/3$ draw a uniformly random sequence, otherwise a uniformly random
balanced one. Then, each with probability $1/2$, apply a
geometrically-distributed number of random transpositions and of random symbol
flips. Uniform balanced sequences are drawn by the cycle lemma. On our test set
this yields *30.6%* balanced sequences.

== Chance levels, derived

A model that has learned nothing still scores above zero on some of these.
Every accuracy in @sec-results is reported beside the number below.

#kv(
  ("modular addition", "1/199 = 0.503% — the answer is one of 199 residues, uniform given nothing"),
  ("needle in a haystack", "1/127 = 0.787% — the answer is a value, uniform in [1,127]"),
  ("decimal addition", "~0% — 10 or 11 digits must all be right at once; a uniform guesser scores 1e-10"),
  ("parenthesis balancing", "69.4% — always answer 'unbalanced', the majority class on our test set"),
  ("memorization", "1/512 = 0.195% — the answer is uniform in [0,511]"),
)

The parenthesis figure is the one that matters. It is high, and it is *not*
zero-information: a reported 87% on this task is a much weaker result than 87%
would be elsewhere, a point we return to in @sec-results.

== The three remaining tasks

*Memorization.* For every pair $(x, y)$ with $x in [0, 511]$ and
$y in [512, 1023]$, an independent uniform label $z in [0, 511]$. That is
$512^2 = 262,144$ associations, each carrying $log_2 512 = 9$ bits, with no
structure whatsoever to exploit. There is deliberately no test split: the
question is capacity, not generalisation.

*Circuit imitation.* Inputs are uniformly random 40-symbol sequences over a
512-symbol vocabulary. The target is not a rule but another randomly initialised
transformer — three layers, two attention heads, and a *width we vary*. This is
the point of the task: it gives us a dial for how many dimensions the target
computation genuinely needs. The student must match the target's output
distribution.

Target models use an amplified initialisation (@sec-methodology), without which
their outputs are near-uniform and every student succeeds trivially.

*Language modeling.* TinyStories @eldan2023tinystories is a corpus of simple
short stories written by GPT-3.5 and GPT-4, with the vocabulary of a young
child. We chose it because small models learn it well enough for the comparison
to be about the models. We fit a 10,000-token byte-level
#gloss[BPE][byte-pair encoding: a tokeniser that merges frequent character pairs
into single symbols] tokeniser on the training split and pack the token stream
into non-overlapping 512-token contexts. Every position is scored.

== What makes these hard, and the shortcut we had to rule out

Three of the four algorithmic tasks cannot be solved by memorising input–output
pairs. Their inputs are drawn fresh from spaces far too large to enumerate.
There are $10^20$ decimal addition problems and $2^60$ parenthesis sequences. Modular addition *can* be memorised — all 39,601 pairs fit easily —
which is exactly why it is the one task with a held-out split.

The shortcut that needed ruling out is on the parenthesis task. There we
deviate from the paper, drawing training data from a fixed pool of 500,000
pre-generated sequences rather than an endless stream. The recursive generator
does not vectorise on a GPU. A finite pool invites memorisation. It
cannot happen here: the model's entire trainable capacity on this task is four
token embeddings, eighty positional embeddings and a four-row unembedding, and
the pool is 500,000 sequences. There is no configuration of those parameters
that stores the pool.
