#import "/template.typ": *

= Task and data

The task is MNIST digit classification @lecun1998mnist: given a 28x28 greyscale
photograph of a handwritten digit, name the digit. The dataset ships as 60,000
training images and 10,000 test images, each an array of 784 bytes in
$[0, 255]$ with an integer label in $0..9$. Both splits are used in full.

Each image is flattened to a vector of 784 floats and standardised,

$ x = (p slash 255 - 0.1307) slash 0.3081, $

with $p$ the raw byte value. The two constants are the mean and standard
deviation of the MNIST training pixels; they are the values the reference
implementation of PC-ALM uses, and using anything else would put our
accuracies on a different footing from the numbers we are checking. Labels are
one-hot encoded: the label $c$ becomes a vector $y in RR^10$ with $y_c = 1$ and
every other entry zero. No augmentation, no cropping, no deskewing.

Where a smaller subset is used (only in smoke tests), it is drawn
class-balanced rather than by a plain shuffle.

Forward-Forward additionally needs the label inside the input, because its
layers are trained to discriminate correctly-labelled data from
incorrectly-labelled data. Following @hinton2022ff, the first ten pixels of the
image are overwritten by a one-hot label. This is done in the raw pixel domain,
so in standardised units the hot pixel is $(1 - 0.1307) slash 0.3081 = 2.8215$
and the other nine are $(0 - 0.1307) slash 0.3081 = -0.4242$. Those ten pixels
sit in the top-left corner and are blank in essentially every MNIST image, so
the overwriting destroys no digit information.

== Chance level

There are ten classes. A network that ignores its input and emits a fixed
class is right whenever the test image happens to be of that class; the test
split has 1,000 images of each digit, so that is $1000 slash 10000 = 0.100$.
A network that guesses uniformly at random is right with probability
$sum_(c=1)^(10) P("guess" = c) P("true" = c) = 10 times (1 slash 10)(1 slash 10)
= 0.100$. Chance is 10%, and every accuracy in this paper is to be read against
it.

== What makes this hard, and what would make it easy

MNIST with backpropagation is not hard: a two-layer network of this size, given
a few epochs and a reasonable learning rate, passes 97%. Three things make the
numbers in this paper much lower than that, and they are all deliberate.

The first is the one-epoch budget. Every headline number is a single pass over
the training set — 937 weight updates at batch 64. That is the budget the
comparison being replicated used, and it is the budget the PC-ALM paper uses for
its width-depth grid, because the local methods are expensive enough that
longer runs across a grid are impractical.

The second is the architecture. The residual network under test is 32 layers
deep and 32 units wide. Depth-to-width ratios like this are unusual in practice
and are chosen by the PC-ALM paper precisely because they are the regime where
standard predictive coding fails, so they are where the comparison between local
rules has any content.

The third is that no method is allowed to see a different dataset, a different
preprocessing, or more than one epoch than another. The obvious shortcut in a
comparison like this — giving one method a learning rate tuned on the test set
and the others a default — is ruled out by sweeping the learning rate for every
method under the same protocol (section 5).
