# Recall on the synthetic prior fails by direction, not by size

> **Superseded in part by report 22.** The mechanics below hold: identification
> depends on the direction of the error rather than its size, the error leans
> toward the average of the context, and the memory addresses the right slot more
> often than the output names it. The framing does not. This report treats
> recovering identification as the goal, and measured as content rather than as an
> index the same outputs already score 0.987 — there was no shortfall to recover.
> Read the derivation here as a reason to stop using argmin over near-duplicates,
> not as a repair to the network.

Report 20 measured a network that reconstructs an item well and still cannot say
which item it is. It explained that as precision: the reconstruction lands
0.023 from the target while the nearest rival sits
0.013 away, so the output is not close enough to win. From there
it concluded that precision tracks capacity, and that scale is the lever.

The explanation does not survive its own control. Take the exact answer, corrupt
it with an error of **the same size the network makes**, point that error in a
random direction, and ask the same question. It identifies the right item
**0.998** of the time overall, and
**0.994** of the time in the quartile where the network scores
0.367.

Size was never the constraint. The network's error is not too big. It is aimed.

This report measures where it is aimed, and finds that a correction with no
trained parameters in it recovers a large part of the shortfall.

## The measurements, defined

Every term this report uses, in the order it needs them.

**Item, world, episode.** One *item* is a vector of 832
numbers. A *world* is a generative process that produces items — a random linear
map from a latent code whose dimension is drawn as low as 1, so items from one
world can lie almost on a line. An *episode* is sixteen items from a single
world, followed by a seventeenth.

**Context, query, mask.** The first sixteen items are the *context*; the network
sees them whole. The seventeenth is the *query*, and roughly half its
coordinates are erased by a *mask*. Erased coordinates are *hidden*; the rest are
*visible*. The network outputs the hidden coordinates.

**Recall.** In these runs the query is always an exact copy of one of the
sixteen context items. The answer is present, and the task is to find it.

**Identification.** The metric this report is about. Take the network's output,
measure its squared distance to each of the sixteen context items on hidden
coordinates only, and see whether the closest one is the item the query was
copied from. *Chance* is 1 in 16, or 0.062. The *ceiling* — feeding the
true answer in and asking the same question — is 0.999 here.
Hidden coordinates only, so a network that merely copies the visible half of the
query cannot score.

**Nearest rival, and the margin.** For a given episode, the *nearest rival* is
whichever of the other fifteen context items sits closest to the true target.
The *margin* is that distance. It is a property of the episode, computable with
no network involved. Episodes are split into four *quartiles* by margin, so the
left-hand quartile is the quarter of episodes whose candidates are hardest to
tell apart.

**Reconstruction error.** Mean squared error between the output and the true
target, over hidden coordinates, divided by the error of ignoring the input and
drawing the average training item. So 1.0 is the do-nothing score, and the
margin is quoted in the same units.

**The error vector.** Write `e` for `output - target` and `Delta` for
`nearest rival - target`, both restricted to hidden coordinates. These are the
only two quantities identification depends on, which the next section derives.

**rho.** The component of `e` along `Delta`, divided by the length of `Delta`.
The single number that decides whether the nearest rival beats the target.

**Alignment.** A cosine between `e` and some direction: 0 means unrelated, 1
means exactly along it. Reported against `Delta` and against the direction from
the target to the *context centroid*, which is the plain average of the sixteen
context items.

**Shrinkage.** How far `e` travels from the target toward that centroid, as a
fraction of the whole distance. 0 is no pull; 1 lands on the average of the
context.

**Retrieval weights.** Defined in the addressing section below, where they are
needed.

## Why only one direction matters

Identification compares two squared distances. Expanding the second around the
first, with `e` and `Delta` as above:

    ||output - rival||^2 - ||output - target||^2  =  ||Delta||^2 - 2 e.Delta

The rival wins exactly when that difference is negative, which rearranges to a
condition on one number:

    rho  =  e.Delta / ||Delta||^2  >  1/2

Read the identity rather than the algebra. Every part of the error that points
across the target-to-rival line cancels out of both distances. Only the part
that points *along* it survives. The size of the error appears nowhere.

This is why an error of the network's own magnitude, pointed at random, is
harmless. The hidden coordinates number a few hundred, and a random direction
puts about one over the square root of that on any particular axis. The project
had already recorded this without drawing the conclusion: `nearest_distractor`
in `lib/splitfig.py` carries a note that an earlier isotropic-noise sweep left
identification at 1.000 out to sigma 1.1, and abandoned the measurement as
uninformative. It was not uninformative. It was the answer.

![Identification under three predictions of the same episodes](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r21_isotropic_v1.svg)

The middle bar is the control. It has the network's error magnitude, quartile by
quartile, and none of the network's error direction.

## Where the error actually points

If not at random, then where. Two candidate directions are worth measuring: the
nearest rival, which is what an interference story predicts, and the centroid of
the context, which is what a hedging story predicts.

![Alignment of the error with two directions](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r21_direction_v1.svg)

| margin quartile | 0.01 | 0.13 | 0.60 | 1.26 |
|---|---|---|---|---|
| identification | 0.367 | 0.664 | 0.992 | 1.000 |
| reconstruction error | 0.023 | 0.077 | 0.080 | 0.079 |
| rho | +0.468 | +0.432 | +0.084 | +0.030 |
| shrinkage toward centroid | 0.426 | 0.407 | 0.079 | 0.059 |
| alignment with centroid | +0.558 | +0.595 | +0.337 | +0.285 |
| alignment with nearest rival | +0.376 | +0.534 | +0.356 | +0.197 |

Read the shrinkage row first. It falls from 0.426 to
0.059 across the quartiles, tracking identification almost
exactly: where the output sits nearly half way to the average of its sixteen
candidates, it cannot name any of them; where it stays on the target, it can.

The two alignments are closer together, and the centroid direction wins by less
than the shrinkage row suggests. It leads in the two quartiles that fail
(0.56 against 0.38, and
0.59 against 0.53) and the two are level
in the third (0.34 against 0.36), where
identification is already 0.992 and the question is moot. So the
centroid is the better description of the error where the error matters, and the
two directions are not cleanly separable elsewhere.

The shrinkage row is not what confusion between two similar items looks like.
Pairwise confusion would leave the output on the line between two candidates
without pulling it toward the average of all sixteen, and would not switch off
as the margin grows. This is regression to the mean of the context — and it is
what squared error asks for. A network uncertain about
which of sixteen items it is looking at minimises expected squared error by
outputting their average, weighted by how likely each is. Identification then
takes a hard nearest-neighbour decision over that average and is punished for
exactly the hedge the loss paid for.

Note the row that report 20 rested on. `rho` in the hardest quartile is
+0.468, and the threshold is 0.500. The network sits on the
decision boundary. Identification there is close to a coin toss, which makes it
a poor dependent variable for comparing architectures — small changes in the
network move it by large, noisy amounts, or not at all.

## Undoing the shrinkage without touching a weight

Shrinkage is a one-parameter defect, so it has a one-parameter correction. Take
the finished output, and push it away from the context centroid:

    corrected  =  centroid + gain * (output - centroid)

No retraining. No gradient. The weights are frozen and the network is not
consulted a second time.

![Identification against de-shrink gain, hardest quartile](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r21_deshrink_v1.svg)

| network | as trained | best gain | at that gain | gain predicted by the shrinkage |
|---|---|---|---|---|
| d256 4x64, 4.06M | 0.203 | x3.00 | 0.305 | x3.04 |
| d512 8x64, 14.95M | 0.367 | x1.50 | 0.523 | x1.74 |
| d512 4x128, 14.94M | 0.367 | x1.50 | 0.508 | x1.81 |

The last column is the check that makes this an explanation rather than a fitted
curve. If the error is a pull of size `s` toward the centroid, the correction
that undoes it is a gain of `1/(1-s)`, computed from the shrinkage column of the
previous table and never from identification. The predicted and the measured
optimum agree on all three networks.

Over all episodes rather than the hardest quartile, exp49 goes from
0.756 to 0.805 at gain x1.50.

Set that beside what it is competing with. The parameter increase report 20
built its conclusion on — 4.06M to 14.95M, roughly four times the parameters and
twice the training cost — bought +0.145 on this metric. A single scalar applied
after the fact buys +0.049 overall and
+0.156 on the quartile the whole diagnosis rested on.

It is a partial correction, not a fix: 0.523 against a ceiling of
0.999. Some of the deficit is real error. Roughly a third of it
was never a deficit at all.

## The memory is not where it is lost

Report 20's other claim was that the memory is not the bottleneck, argued from a
capacity sweep. The claim is right and the argument was indirect. The memory can
be read.

Recall here runs on a matrix-valued state written by the delta rule. Each context
token writes `S <- S + e_i k_i^T`, where `k_i` is that item's *key* and `e_i` is
the correction the write applies, and between writes the state decays by a
learned per-channel factor. After all sixteen writes the state is a sum of those
outer products, so the query's read is

    output  =  sum_i  e_i * w_i,      w_i = (k_i * A_i) . q / sqrt(dk)

with `A_i` the decay that survives from write `i` to the end and `q` the query's
own key. Those `w_i` are the *retrieval weights*: this architecture's version of
an attention distribution over the sixteen items, recovered from the state rather
than inferred. Whether `w_i` is largest at the target separates *addressing* the
right slot from *reconstructing* what is in it.

![Addressing against naming, hardest quartile](https://media.tanh.xyz/seewhy/26-09-04/recall-gen_r21_addressing_v1.svg)

On the hardest quartile, in exp49's best layer (layer 3), at least
one head puts its largest retrieval weight on the correct item in
**0.711** of episodes. The finished output names that item
in **0.367**.

Addressing runs ahead of naming, on every network tested. Whatever is lost is
lost after the state has already found the item. That is a direct measurement of
the thing the capacity sweep could only probe, and it settles it in the same
direction.

One further observation the scalar metric hides. In d512 4x128 the addressing
concentrates into a single sharp layer — layer
0 reaches a
separation of +1.34 standard
deviations between the target's weight and the rival's — while another layer
goes blind. In d512 8x64 the same work is spread evenly across all four
layers. Same parameter count, same final score, different machine. Doubling the
head dimension reorganised the computation rather than adding to it.

## What this changes

Report 20's headline claim about the memory stands. Its explanation of the
shortfall does not, and the recommendation that followed from it should be
withdrawn.

The sentence "this is a precision problem, and precision tracks capacity" is
wrong on the first clause, which makes the second irrelevant. Identification does
not depend on how close the output lands. It depends on which way the output
leans, and it leans toward the average of the context because squared error pays
for leaning that way.

Three things follow.

Identification should be quoted with `rho` beside it, or with the isotropic
control beside it. A number that can be moved by +0.049
with a scalar is not measuring what it appears to measure.

Low-margin episodes are a step function evaluated at its own threshold. Sweeping
architectures against identification on the hardest quartile measures noise
around a decision boundary. Either report the margin-resolved table, or use a
metric that degrades smoothly.

The next experiment on this prior is an objective, not a size. Squared error over
a set of candidates selects for the hedge this report measures. A discrimination
term, or a whitened output space, addresses the cause; more parameters buy a
smaller hedge only incidentally.

## What this does not establish

That capacity is irrelevant. The correction is partial, the largest network is
still the best at every gain, and d256 4x64 responds to it far less
(0.305 at its own optimum) because more of its error is genuinely
misdirected rather than merely shrunk.

That the fix transfers. The gain is one number chosen on the same 512 episodes
it is scored on. The peak is broad and it is predicted independently by the
shrinkage, so it is not a fitted artefact, but it has not been validated on a
held-out draw.

That this is the whole error. Alignment with the centroid is
0.56, not 1.00. A substantial part of the error points
somewhere neither direction tested here describes.

Anything about the absent-answer task. Every number here is from
`A_seen_present`: worlds seen in training, answer present in the context. This
report is about recall, not about completion.

## Sources

`results.jsonl` rows `exp45`, `exp49` and `exp54` — recall training on the
synthetic prior at d_model 256/512 and two head/dk splits, the same three
checkpoints report 20 used. Every number is recomputed from those checkpoints by
`scripts/diag_identification.py`, which also carries the derivation above in its
module docstring. Episodes are the project's standard eval draw at M=16,
condition `A_seen_present`, context type `class` — single-world episodes, the
type these networks were trained on. Identification is scored on the first query
of each episode, matching `lib/splitfig.nearest_distractor`; published numbers
average four queries, which is why exp49 reads 0.756 here against
0.759 there. Figures generated by `scripts/gen_report_21.py`. No training was run
for this report.
