#import "/template.typ": *

// OBLIGATIONS
//  - Mechanism, not restatement of §6.
//  - What would falsify the explanation, and whether that test was run.

= Analysis

== What retrieval training actually learns, and how far it travels

The delta-rule state is a content-addressed table. A context image is written
under a key computed from its own pixels, and a query reads with a key computed
from its visible pixels. Nothing in that circuit refers to *which* image is being
stored — only to what it looks like, so there is no slot in it for a memorised
identity, and it applies unchanged to images the model has never met. The digit
split confirms it: identification accuracy is 1.000 on digit classes the model
never saw in training. Whatever the keys encode, it is not digit identity.

It is tempting to stop there and conclude that the mechanism is
distribution-free. It is not, and the difference matters enough to state
carefully.

A key is a *learned* map from 784 pixels to a 64-number address. What makes two
images land at different addresses is a similarity metric, and that metric was
fitted to MNIST. Fashion-MNIST — still grayscale, still centred, still
28 #sym.times 28 — drops identification to 0.651. MNIST under a single fixed
pixel permutation drops it to 0.116, against chance 0.063, even though the
permuted images have the same pixels, the same marginal statistics and the same
pairwise distances, so a nearest-neighbour matcher on raw pixels would score
identically on both pools. The entire gap between 1.000 and 0.116 is the learned
encoder's dependence on MNIST's spatial layout.

So *neither* of the two things a model can learn here is free of its training
data. They differ in **granularity**:

#align(center, table(
  columns: 3, stroke: none, align: (left, left, left), inset: 5pt,
  table.hline(stroke: rule),
  [*what is stored in the weights*], [*granularity*], [*how far it travels*],
  table.hline(stroke: rule),
  [individual images], [one entry per training image],
  [nowhere: 0.134 on training-pool images against 0.561 on novel ones],
  [a similarity metric], [one function for the distribution],
  [within the distribution: 1.000 across digit classes, 0.651 across datasets,
   0.116 once spatial layout is destroyed],
  table.hline(stroke: rule),
))

"Retrieval generalises" therefore means the coarse-grained thing transfers within
the distribution it was fitted to. That is a real and useful property — it is
what lets the mechanism work on unseen classes — but it is not freedom from the
training data, and it cannot be assumed to survive a change of domain.

=== The permutation result is a diagnostic, not a deficiency

It would be easy to read 0.116 as a failure to be fixed, and to treat
permutation-robustness as a target. That reading is wrong, and it is worth being
explicit about why.

There is no free lunch: a retriever that makes *no* assumption about its inputs
cannot outperform one that makes a correct one, on any distribution where a
correct assumption exists. Spatial structure in images is such an assumption, and
it is *true* — natural images are not pixel-permuted, and a model that discarded
layout in order to score well on permuted inputs would have thrown away real
information in exchange for robustness against a distribution that does not
occur.

So the permutation test should be read as a probe of *which* assumption the
training data taught the model to rely on, not as a benchmark. The answer it
returns — that spatial layout is load-bearing for the learned keys — is evidence
that the mechanism found and exploited a genuine regularity of its data. The
score of 0.116 is informative precisely because it is low.

The same distinction sorts the other two pools. Fashion-MNIST is natural data
that shares the spatial-structure assumption, so 0.651 is a meaningful measure of
transfer and a meaningful thing to try to improve. Random fields are neither
natural nor structured, so 0.222 bounds what the encoder does with inputs that
violate its assumptions and is not a target at all. The general point: arbitrary
or adversarial data is a good instrument for *discovering which assumptions a
model has made*, and a bad objective to optimise against, because the world it
describes is not the world the model will see.

== Why completion decays as retrieval sharpens

Early in training the read from the state is diffuse: keys are near-random, so
$S q$ is close to an unweighted average of the stored values, and the decoder
turns it into something near the average digit. That is why the absent-target
conditions *start* near 0.635 — the model is accidentally implementing the
uniform-average strategy, which the reference table shows is worth about 1.0 in
the limit and rather better than that early, when the output is also partly
driven by the query's own visible half.

Recall training then sharpens the keys, because that is what reduces its loss:
the state must separate the $M$ context items well enough to return one of them.
As the read becomes selective, the accidental averaging disappears. On an
episode where nothing in the state matches, a selective read returns an
incoherent mixture rather than a mean, and the decoder produces the fragmentary
output visible in the completion grids. The measured trace of this is that the
distance from the model's output to the mean image *rises* over training on the
absent-target conditions, from 0.005 at initialisation to 0.046 at convergence:
the model is not falling back on the prior, it is moving away from it.

So the deficit is not "the state is full of images and has no room for
knowledge". exp3 rules that reading out directly — the same state does both at
once when the objective asks for both, at no cost to the completion ceiling. The
deficit is that a recall objective supplies no gradient toward completion, and
the diffuse-read behaviour that incidentally provided some is actively in the way
of the objective it does have.

== Why generalisation appears at large context

Two explanations are available for the $M$-sweep, and they make opposite
predictions.

/ Information: a larger context contains more digits, so there is more to
  generalise from, and the model learns to use it.
/ Capacity: a larger context cannot be stored, so retrieval stops paying, and
  gradient descent falls into the only other minimum — put the digit prior in the
  weights.

The results select the second, on three independent grounds. The model is *below*
the soft-look-up ceiling at every $M$ and further below it as $M$ grows, so its
answers are not extraction from the context. At $M = 256$ it scores 0.556 with
the answer present and 0.561 without, so removing the answer from the context
costs it nothing. And its condition C beats its condition D fourfold, so its
advantage tracks whether an image was in the *training pool*, not what is in the
context.

The convergence is the strongest evidence: at $M = 256$ a model trained purely on
retrieval and a model trained purely on completion arrive at the same place — the
same ceiling, the same coincidence of their answer-present and answer-absent
numbers, the same C/D asymmetry. They are running the
same algorithm. One of them was asked to retrieve and could not.

*What would falsify this.* If the effect is capacity, it must reproduce when the
memory shrinks at fixed context: the context is then identical episode for
episode, and only the ability to hold it changes. If the effect is information,
shrinking the state should not produce it. That is exp15–exp17, and it is the
reason the paper contains a state-size sweep at all.

*The verdict.* It reproduces. Holding the context at 16 images and shrinking the
memory from 16 384 numbers to 2 048, the answer-present error rises 0.017
#sym.arrow 0.019 #sym.arrow 0.024 #sym.arrow 0.035 while the answer-absent error
improves 0.852 #sym.arrow 0.819 #sym.arrow 0.755 #sym.arrow 0.681, on evaluation
episodes that are identical across all four runs. The information account predicts no
movement here and gets the sign of the effect wrong; the capacity account
predicts exactly this.

It reproduces in direction rather than in magnitude. Even at the smallest state,
retrieval has not collapsed — identification accuracy is still 1.000, and the two
conditions are still 0.035 and 0.681 rather than the 0.556 and 0.561 seen at
$M = 256$. Compression ratio alone is not
sufficient either: at a ratio near 3, the small-state run reaches D = 0.755 and
the large-context run 0.658, so how many items must be told apart matters
alongside how many numbers are available to tell them apart with. Both are
properties of the memory rather than of the context, which is what the argument
needs, but the two knobs are not interchangeable and we do not claim they are.

== Why a bigger context at inference changes nothing

If large contexts carried usable signal, a model trained at $M = 16$ should
extract some of it when handed 256 images. It extracts none: its completion error
*rises* to 0.942 while a model trained at 256 reaches 0.561 on the same task. And
the reverse fails too — the model trained at 256 scores 0.552 and 0.541 on the
two conditions when handed a context of 4, where the state has sixteen-fold spare
capacity: still the same number, still not reading anything.

Both follow from the two-route account. The short-trained model took route A and
never built route B, so a larger context has nothing to unlock; all it does is
overload a memory that was tuned for sixteen items, which is why the error climbs
towards the mean-image line rather than staying flat. The long-trained model took
route B and never built route A, so a smaller context does not give it anything
to read — its error is flat to within 0.02 across a 64-fold change in context
size, which is what a model that ignores its input entirely looks like.

The two are separate attractors. Training selects one; inference-time context
size does not move between them. Any account in which context length is the
operative variable *at inference* predicts otherwise and is ruled out.

== Why nothing crosses the digit split

Retrieval crosses it perfectly and completion does not cross it at all: the
recall-trained model completes unseen classes at the mean-image level, and the
completion-trained model does *worse* than the mean image on them. Even ridge
regression, fitted on 0–4, only reaches 0.851.

The asymmetry follows from what each ability has to represent. Retrieval needs a
map from image to key that keeps distinct images distinct — a generic property of
pixels, indifferent to which digits exist. Completion needs the conditional
distribution of the bottom half given the top, which is *specific to the shapes
in the training set*. A prior for 0–4 applied to a 7 is not merely uninformative;
it is wrong, which is how the completion-trained model ends up above 1.0.

This is the sharpest form of the paper's point. On the same images, in the same
episodes, the retrieval machinery transfers completely and the generalisation
machinery does not transfer at all. They are not two grades of the same ability.
