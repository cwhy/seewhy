#import "/template.typ": *

// OBLIGATIONS
//  - Mechanism, not restatement of §6.
//  - What would falsify the explanation, and whether that test was run.

= Analysis

== Why retrieval transfers for free

Retrieval generalising to unseen images is the least surprising result here, and
it is worth saying why so that it is not mistaken for a strong claim.

The delta-rule state is a content-addressed table. A context image is written
under a key computed from its own pixels, and a query reads with a key computed
from its visible pixels. Nothing in that circuit refers to *which* image is being
stored — only to what it looks like. A mechanism built out of "compute a key from
the input, match, read" has no place to put a memorised identity, so it applies
unchanged to inputs it has never met.

The digit split makes the point sharply: identification accuracy is 1.000 on
digit classes the model never saw in training. Whatever the keys encode, it is
not digit identity. This is generalisation in the sense that the *mechanism*
transfers — and it is exactly the sense in which retrieval is cheap.

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
answers are not extraction from the context. Its gain is 0.004 at $M = 256$, so
removing the answer from the context costs it nothing. And its condition C beats
its condition D fourfold, so its advantage tracks whether an image was in the
*training pool*, not what is in the context.

The convergence is the strongest evidence: at $M = 256$ a model trained purely on
retrieval and a model trained purely on completion arrive at the same place — the
same ceiling, the same zero gain, the same C/D asymmetry. They are running the
same algorithm. One of them was asked to retrieve and could not.

*What would falsify this.* If the effect is capacity, it must reproduce when the
memory shrinks at fixed context: the context is then identical episode for
episode, and only the ability to hold it changes. If the effect is information,
shrinking the state should not produce it. That is exp15–exp17, and it is the
reason the paper contains a state-size sweep at all.

*The verdict.* It reproduces. Holding the context at 16 images and shrinking the
memory from 16 384 numbers to 2 048, gain falls monotonically 0.835 #sym.arrow
0.800 #sym.arrow 0.730 #sym.arrow 0.646 and completion improves 0.852
#sym.arrow 0.819 #sym.arrow 0.755 #sym.arrow 0.681, on evaluation episodes that
are identical across all four runs. The information account predicts no
movement here and gets the sign of the effect wrong; the capacity account
predicts exactly this.

It reproduces in direction rather than in magnitude. Even at the smallest state,
retrieval has not collapsed — identification accuracy is still 1.000 and gain is
still 0.646, against 0.004 at $M = 256$. Compression ratio alone is not
sufficient either: at a ratio near 3, the small-state run reaches D = 0.755 and
the large-context run 0.658, so how many items must be told apart matters
alongside how many numbers are available to tell them apart with. Both are
properties of the memory rather than of the context, which is what the argument
needs, but the two knobs are not interchangeable and we do not claim they are.

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
