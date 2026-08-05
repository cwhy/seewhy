# The paper in plain terms

*No prior reading assumed. This page explains what
[Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns](https://arxiv.org/abs/2606.25010)
(Baherwani, Chen, Qiu, Wilson & Izmailov, NYU, June 2026) argues, and why anyone should
care. What **we** measured is on the other pages.*

## The puzzle

When you train a language model, the number it optimises — prediction loss — improves
smoothly and predictably. Double the compute, get a reliably slightly better number. This is
so dependable it has its own name: scaling laws.

But the things people actually want from a model don't arrive smoothly. A model asked to
complete "Mary and John went to the store. John gave a drink to ___" is at chance for a long
stretch of training, then over a short window becomes reliably correct. Nothing in the loss
curve marks the moment. The capability just shows up.

This is called **emergence**, and it's awkward for two reasons. Practically, you can't
forecast what a model will be able to do from the metric you're watching. And scientifically,
"the ability appeared somewhere between checkpoint 8,000 and 9,000" is not an explanation.

Two families of explanation were already on the table. Maybe emergence is a **measurement
artifact** — accuracy is a harsh, all-or-nothing scoring rule, and if you measured the
model's probabilities instead you'd see smooth improvement all along. Or maybe capabilities
genuinely arrive in jumps, and something about training makes them do so. This paper argues
the second, and names the something.

## The claim

A capability like the one above needs the model to route information between specific
positions: to answer, it has to look back at "Mary" — not at "store", not at everything
equally. In a transformer that routing is done by **attention**.

The paper's argument is:

> A capability requires one **specific, sparse attention pattern**. Gradient descent has to
> *find* that pattern among a huge number of wrong ones. Until it does, the model is stuck;
> when it does, the capability appears within a few hundred steps. Because it's a search, the
> step at which it succeeds is substantially random.

So emergence isn't a threshold that scale crosses. It's the moment a search happens to
succeed. Bigger models search faster on average — but the *moment* stays stochastic.

### What attention is, just enough to follow

At each position, the model forms a query — loosely, "what am I looking for?" — and compares
it against a key at every earlier position. The comparisons become weights that sum to one,
and the position's new representation is a weighted blend of what it found. One such
comparison-and-blend unit is a **head**; a model has many, running in parallel.

Two properties matter here:

- The weights are **soft**. A head can spread attention evenly over 500 positions, or put
  nearly all of it on one. "Spread evenly" is roughly where training starts.
- A useful pattern is usually **sparse**: attend to two or three specific positions and
  ignore the rest. Getting from "spread evenly" to "these three" is the search in question.

## The trick: build a task where you know the answer

In a real language model you can't check whether a head found the right pattern, because
nobody knows what the right pattern is. So the paper builds tasks where the correct pattern
is fixed **by construction**.

**The linear map task.** Pick a secret 0/1 matrix `A` where each row has exactly `s` ones.
Show the model a random bit string `x`, then the string `Ax mod 2` — each output bit is the
parity (XOR) of `s` specific input bits. To predict output bit `i`, the model *must* attend
to exactly those `s` input positions. We know which ones, so "did it find the pattern?"
becomes something you can measure, not interpret.

The knobs are the ones the theory cares about: `s` sets how sparse the needed pattern is, and
the string length sets how many positions there are to search through.

**The cellular automata task.** A grid of cells updated by a local rule: each cell's next
value depends on its neighbours. The needed attention pattern is again known — a small window
around each cell — and the rule can be composed with itself to widen that window. Here the
model isn't told which rule is active; it has to infer it from the sequence, which makes this
the in-context version of the same question.

## What they found

**Emergence looks the same in the toy setting.** Loss sits flat at the value you'd get by
guessing, then drops sharply. Attention maps before and after the drop show a head switching
from near-uniform to the correct sparse pattern. The loss jump and the pattern being found are
the same event.

**Difficulty depends on sparsity and context length.** Short strings are learnable at any
sparsity. Longer strings develop a range of sparsity levels the model *never* learns within
the training budget.

**More heads help; bigger heads mostly don't.** Holding total width fixed and splitting it
into more heads reliably speeds learning — even 128 heads of size 1. Making each head bigger
past a modest size buys little. The natural reading: each head is another attempt at the
search, so more heads means more attempts, and that matters more than the capacity of any
one attempt.

**Architectures that don't search do better.** On the same data, an MLP-Mixer — which learns
position-mixing weights directly instead of computing them through a softmax competition —
learns the linear map *faster* than a transformer. If the bottleneck were the function's
complexity, this wouldn't happen; the function is the same. It happens because the mixer
doesn't have to search.

**And it holds in real language models.** Using the Pythia family (14M to 410M parameters)
and its saved training checkpoints, they track capabilities like copying, in-context
repetition, pattern completion, and the Mary-and-John task (known as indirect object
identification) across training. The capabilities appear abruptly, at times that vary, and
their appearance coincides with specific attention heads sharpening. Knocking out the
identified heads removes the capability, which is what makes the attribution causal rather
than correlational.

## Why it matters

**Forecasting.** If emergence is a search that succeeds at a random time, then "capability X
appears at scale Y" is a distribution, not a threshold — and predicting from a single
training run is unreliable in principle, not just in practice.

**Evaluation.** Two identical runs can differ in what they can do. Judging a recipe or an
architecture from one seed risks measuring a lucky draw. (Our [exp1](sparse_attn_emergence_exp1.html)
makes this concrete: at fixed everything, 16 seeds emerged between step 469 and step 2521.)

**Architecture.** "More heads beats bigger heads" is an actionable design claim, and it
comes with a reason rather than just a benchmark.

**The artifact debate.** This is evidence on the side of emergence being real: in the toy
task the loss itself — not a thresholded accuracy — sits flat and then falls.

## What we're checking, and what we've found

We re-ran the synthetic half at small scale, with more seeds than the paper uses (16 rather
than 3), because the central claim is about *variation between runs* and three samples can't
show a distribution. The real-model half is out of scope.

So far: the seed-randomness claim holds clearly and twice over, and the mechanism claim
survives a causal test — removing the aligned head takes loss from 0.0000 to 4.23, while
removing a comparable unaligned head costs 0.08
([exp4](sparse_attn_emergence_exp4.html)).

Two things came out differently. "Abrupt" is fair but softer than the figures suggest: the
drop takes about 40% of the time it took to start, not a single step. And one of the task's
extreme settings turns out to be **degenerate** — at maximum density every answer is the same
value, so a model can score 97% by copying its own previous output without learning anything.
Both details are on [exp2](sparse_attn_emergence_exp2.html), and the second one applies to the
paper's own setup, since we share the construction.

## Words you'll see on the other pages

| Term | Meaning here |
|---|---|
| **head** | one attention unit; our models have 8 |
| **attention pattern** | which earlier positions a head draws from, and how strongly |
| **plateau** | the flat loss stretch before the capability appears. Its value is `ln 2 ≈ 0.693` for a 50/50 guess |
| **seed** | the random number that sets initial weights and data order. Same seed, same run |
| **`t*`** | time-to-emergence: the training step where the capability appears |
| **ablation** | deliberately breaking one part of a trained model to see what depends on it |
| **in-context** | inferred from the current input rather than stored in the weights |
