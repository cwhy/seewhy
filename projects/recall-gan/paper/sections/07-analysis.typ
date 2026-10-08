#import "/template.typ": *

= Analysis

== Why the generator's access to the context matters

The discriminator is never told that unseen real images exist. Its only
negatives are the generator's images. So the information it extracts from its
state is exactly the information it needs to reject *those* images.

A generator whose images can be recognised by their look, such as a collapsed
one, asks for no memory. The discriminator rejects them on sight, and its
state goes unused: AUROC 1.000 with the right context and with the wrong one.

A generator that draws digits of the classes in its context asks for more.
Its images look like digits, and like the right digits. To reject them the
discriminator must ask whether *this* image is one of the 16 it stored. That
is what the swap test measures, and with the context-aware generator the
answer depends on the stored context (AUROC 0.970 with it, 0.570 without).

The training curves suggest an order, though not a cause. In all 12 runs the
discriminator began to rely on its state before the generator began to use
the context, by a median of 1250 steps. One reading is that the discriminator's
use of its state is what makes the context worth reading for the generator,
which then pushes the discriminator further. These runs do not test that
reading; they show only the order.

== What the state stores

The probes point to a stored summary that is coarser than the images
themselves.

- *It is about classes more than items.* Unseen images of a digit present in
  the context score higher than images of absent digits (0.227 against 0.061).
  With a single-digit context, the state works as a digit classifier
  (@sec:classifier), which pure item-level storage would not.
- *It is more than pixels.* With the same 16 examples per digit, it classifies
  better than nearest-neighbour or RBF density in pixel space, even at the RBF
  bandwidth that suits this test best. Where pixels are least reliable (twos,
  eights, fives) the gap is largest.
- *It is not a calibrated density.* Scores saturate at a ceiling, so the top of
  the ranking is a block of equally-scored images of one digit
  (@fig:strips). Some unseen images of a class present in the context get the
  lowest score of all. Context images themselves are sometimes scored below 0.5
  (@fig:episode).

== What would falsify this account

If the discriminator's use of its state came from the architecture alone,
the blind runs would show it too; they do not. That comparison is weakened by
the blind generators having collapsed (@sec:limits). A cleaner test is a
blind generator that does not collapse, under the same recipe. That run has
not been done.
