#import "/template.typ": *

= Analysis <sec-analysis>

The results reproduce. This section asks whether the paper's explanation for them
survives contact with our data.

== The mechanism, and what would break it

The paper's account is *subspace selection*. The frozen stack computes a fixed
function $F$. Training picks an encoding into $F$ and a decoding out of it. If
the task is already computable somewhere inside $F$, the embeddings can steer the
representation into the subspace where it happens.

This makes three predictions. All three are testable, and we tested them.

*It should not be sparsification.* If training instead found a small set of useful
neurons, variance would concentrate on individual coordinates. It does not. Ten
principal components explain 0.76 to 0.94 of the variance; ten neurons explain
0.017 to 0.049. A twenty-fold gap is not consistent with a sparse subnetwork.

This distinction matters because the lottery-ticket literature @frankle2019lottery
@ramanujan2020whats found that random networks *do* contain useful sparse
subnetworks. Those are found by pruning. What happens here is not that.

*Capacity should be low.* A subspace has fewer directions than the full space, so
less can be stored in it. The memorization task confirms this: 0.20 bits per
trainable parameter against 2.40 for full training.

*Imitation should fail exactly when the target needs many dimensions.* This is the
prediction that could have gone the other way, and it is why the experiment
exists. It holds. Divergence rises 4.5-fold from target width 16 to 32, and
sixfold again to 128, while a fully trained student of the same size stays flat.

The explanation survives all three.

== Where concentration stops being a virtue

The subspace numbers behave in opposite directions on the two kinds of task, and
this is the most informative pattern in our data.

On the algorithmic tasks, random and trained models are similarly concentrated.
Both operate in low-dimensional subspaces, because the tasks need only
low-dimensional computation.

On memorization the random model is *more* concentrated than the trained one:
0.887 against 0.692 in the top ten components. It is also far worse at the task.
Here concentration is the problem. Storing 262,144 arbitrary facts requires using
many directions, and the random model cannot spread out into them.

So "operates in a low-dimensional subspace" is not a description of a good model.
It is a description of a constrained one. It looks like an explanation of success
on the algorithmic tasks and an explanation of failure on memorization, because it
is the same constraint in both cases. The tasks differ in whether the constraint
binds.

== Why the language modeling result is the one that fits least comfortably

Language modeling needs both things. It needs structure — agreement, closing
quotes, clause boundaries — and it needs arbitrary associations between words and
their meanings.

Our samples split exactly along that line. Grammar survives. Reference does not.
A model that writes "yellow policy" has the syntax of English and not its
semantics.

That is what the subspace account predicts, and it is satisfying. But it is also
the weakest evidence in the paper, ours included, because it rests on reading
samples. We report cross-entropy, which is measurable, and two completions, which
are not. We did not run a grammaticality metric. A reader should treat the
grammar claim as an observation, not a result.

== An accident that turned into evidence

We could not reproduce the paper's fully trained baselines at first. Our
implementation followed the paper's stated optimiser exactly, and under it a
fully trained width-1024 model plateaued at 0.14 on needle in a haystack. It
stayed there for 10,000 steps, which is the paper's own budget.

The cause was a missing learning-rate warmup and decay, absent from the paper and
present in the authors' code (@sec-limitations).

The relevant part is which condition noticed. Across four optimiser settings,
embedding-only training reached 1.000 every time. Full training ranged from 0.14
to 1.000 over the same four.

This is what the paper's thesis predicts, though the paper does not make the
argument. A fully trained model has to *find* a retrieval circuit by gradient
descent. That search has a plateau, and a bad optimiser schedule leaves it there.
A random transformer is not searching for a circuit. It is searching for an
encoding of one that already exists. There is no fragile search to disrupt.

We found this by accident, while chasing what we assumed was our own bug. It is
weak evidence — one task, one architecture, four settings. We report it because
it is the kind of observation a replication is positioned to make and the original
paper is not.

== The seed that did not make it

One further asymmetry. On needle in a haystack at width 1024, the fully trained
condition has a median of 0.916 and a seed range of 0.14 to 0.96. One run in five
never left the plateau, even with the corrected schedule.

The embedding-only condition has a range of 1.000 to 1.000, on every task, in
every seed we ran.

Reporting only medians would hide this. The variance is not noise around a mean.
It is bimodal: the fully trained model either finds the circuit or does not.
