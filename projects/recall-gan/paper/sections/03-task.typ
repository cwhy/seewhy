#import "/template.typ": *

= Task and data

== Data

MNIST @lecun1998mnist: 28 × 28 greyscale images of handwritten digits,
flattened to 784 numbers in $[0, 1]$. Training uses the 60 000-image training
split. *Every* evaluation uses the 10 000-image test split. So every context
evaluated in this paper is made of images neither network trained on.

== Episodes

All training and testing happens in *episodes*. An episode draws $M = 16$
images uniformly at random from the pool: the *context*. Duplicates are
possible but rare (16 draws from 60 000).

During training each episode asks the discriminator about $8$ query images:

#figure(
  table(
    columns: 3, align: (left, left, left), stroke: none,
    table.hline(),
    [*query*], [*source*], [*target*],
    table.hline(stroke: 0.5pt),
    [4 real], [copies of context images, chosen at random], [1 ("in context")],
    [4 fake], [generator outputs for this episode], [0 ("not in context")],
    table.hline(),
  ),
  caption: [The queries of one training episode. Nothing else is ever queried
  during training; in particular, real images that are *not* in the context
  never appear.],
) <tab:queries>

64 episodes are processed per training step.

== Chance levels

The training queries are balanced, 4 against 4. A discriminator that outputs a
constant is right on half of them, so chance accuracy is $4 / 8 = 0.5$, and
chance #gloss[AUROC][the probability that a randomly chosen positive gets a
higher score than a randomly chosen negative] is $0.5$.

The digit classifier of @sec:probes chooses among 10 digits, so its chance
accuracy is $1 / 10 = 0.1$ when classes are balanced, as they are in the MNIST
test split to within a few percent.

== What makes the task hard

The discriminator never sees an example of "real, but not in the context".
Nothing in training asks it to reject an unseen real digit. It only has to
reject the generator's images.

There is therefore an easy solution: recognise generated images by their look
and ignore the context. That solution needs no memory, and it stays available
for as long as the generator's images look different from real ones. Whether
the discriminator uses its context at all is the central measurement of this
paper, not an assumption. @sec:probes defines two tests for it.
