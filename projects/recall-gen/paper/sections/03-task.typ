#import "/template.typ": *

// OBLIGATIONS
//  - Exact shapes, splits, sizes, preprocessing.
//  - Chance level DERIVED, not asserted.
//  - What makes the task hard, and the shortcut ruled out.

= Task and data

== Images as tokens

The models in this paper read a sequence of items, one at a time. Ordinarily
those items are words. Here each item is an entire handwritten digit.

We use MNIST @lecun1998mnist: 70 000 greyscale images of handwritten digits,
28 #sym.times 28 pixels each, split by the dataset's authors into 60 000
training and 10 000 test images. Every pixel is an integer in $[0, 255]$, which
we divide by 255 so that 0 is black and 1 is white. Each image is then flattened
into a vector of $28 times 28 = 784$ numbers. That vector is one #gloss[token][
the smallest unit a sequence model reads or writes] — so where a language model
sees a sequence of words, our models see a sequence of pictures.

== An episode

Each training example is a self-contained *episode*:

$ underbrace(x_1 space.quad x_2 space.quad dots.h space.quad x_M, "context")
  quad quad
  underbrace(tilde(x)_(t_1) space.quad dots.h space.quad tilde(x)_(t_Q), "queries") $

The first $M$ items are complete images drawn at random from a pool. The last
$Q$ items are *masked*: the bottom $R$ rows of the image have been set to zero.
Alongside each masked image the model is given a binary vector of the same
length marking which pixels were removed, so it always knows where the hole is
rather than having to infer it from the black pixels — MNIST is mostly black
background, and that ambiguity would otherwise be a confound.

The model's job is to fill in the hole: for each query it must output the
$28 R$ pixel values that were removed. Nothing else is scored.

Unless stated otherwise $M = 16$, $Q = 4$, and $R = 14$, so exactly the bottom
half of each query image is hidden and $28 times 14 = 392$ of the 784 pixels
must be predicted.

== The four conditions

Everything in this paper turns on two independent binary properties of an
evaluation episode.

/ Is the target present?: Whether the true, complete version of the query image
  is one of the $M$ context images. When it is, the answer is *available* in the
  episode and the task is retrieval. When it is not, no amount of looking will
  produce the answer and the model must fall back on what it knows about digits
  in general.

/ Is the context novel?: Whether the $M$ context images are drawn from the pool
  the model trained on, or from MNIST's test split, which it has never seen in
  any role.

Crossing the two gives four conditions:

#align(center, table(
  columns: 3, stroke: none, align: left,
  table.hline(stroke: rule),
  [], [*target in context*], [*target absent*],
  table.hline(stroke: rule),
  [context from training pool], [*A*], [*C*],
  [context never seen], [*B*], [*D*],
  table.hline(stroke: rule),
))

Condition A is the only one any recall-trained model in this paper is trained
on. B asks whether retrieval survives on images the model has never encountered.
C and D ask what the model does when there is nothing to retrieve.

Within an episode, image indices are drawn *without replacement*, so in C and D
the target is guaranteed absent rather than merely unlikely to be present.

== What counts as good, and what chance is

The natural error measure for filling in pixels is mean squared error over the
hidden pixels. Raw MSE is hard to read, so every number in this paper is
*normalised MSE*: the model's MSE divided by the MSE of the single most trivial
strategy, predicting the average of all training images at every hidden pixel.

$ "nMSE" = (sum_(i in "hidden") (hat(y)_i - y_i)^2) / (sum_(i in "hidden") (macron(x)_i - y_i)^2) $

The denominator is the analogue of chance level for a regression task: 1.0 means
*exactly as good as ignoring the input entirely*, below 1.0 is better than the
trivial prior, above 1.0 is worse than it. Measured on our evaluation episodes
the denominator is 0.0711–0.0743 depending on condition; because it is computed
per condition, nMSE is comparable across conditions even though the underlying
images differ.

We also report *identification accuracy* on conditions A and B. Take the model's
output, and ask which of the $M$ context images it most resembles — measuring
the distance on the *hidden* pixels only, so a model that merely copies the
visible half of the query cannot score. Identification accuracy is the fraction
of queries where the answer is the correct context image. Chance is
$1 slash M = 1 slash 16 = 0.0625$, since one of $M$ equally plausible context
images is correct.

== What makes it hard, and the shortcut we had to rule out

Retrieval here is not free. The context is $M times 784 = 12 thin 544$ numbers,
and the model's memory of it (§4) is fixed at 16 384 numbers regardless of $M$ —
so at $M = 64$ and $M = 256$ the context cannot be stored at all, only
summarised.

The shortcut worth ruling out is the one that makes a retrieval result
meaningless: a model could ignore the context and simply learn to complete
digits, since MNIST digits are highly predictable from their top halves. Two
things rule it out. First, identification accuracy is computed on hidden pixels
against the context set, so it only rises if the output matches the *specific*
context image rather than a plausible digit. Second, condition C exists: a model
completing from a general prior scores the same in A and C, whereas a model
retrieving scores far better in A. Both diagnostics are reported for every run.
