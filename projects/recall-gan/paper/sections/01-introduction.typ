#import "/template.typ": *

= Introduction

Suppose a network is shown sixteen images, and then a seventeenth, and must
say whether the seventeenth was among the sixteen. Suppose also that the
network may not keep the sixteen images. It must squeeze them into a memory of
fixed size and answer from that memory alone. What should such a memory hold,
and how could it be trained?

The obvious training signal is labelled pairs: "this image was in the set",
"this one was not". This paper uses a different one. It trains the memory as
the discriminator of a #gloss[GAN][a pair of networks, a generator that makes
images and a discriminator that judges them, trained against each other]. In
each training episode, the "real" images are the sixteen context images. The
"fake" images come from a generator. The discriminator reads the context into
a fixed-size linear-attention memory and must accept the context images and
reject the fakes.

Nothing in training shows the discriminator a real image that is *not* in the
context. It is only ever asked to reject the generator's images. So whether it
learns anything about "this particular set" depends entirely on the generator.
If the generator's images can be told apart by their look, the discriminator
needs no memory at all. If they cannot, it has to consult what it stored.

The generator comes in two versions: one that sees the context and one that
does not. The question is whether the game, on its own, turns the
discriminator's memory into a test of "is this image in my context?".

== Contributions

- *A GAN whose discriminator is an in-context set-membership test.* The
  discriminator sees its context only through a fixed-size KDA state of
  16 384 numbers, and is trained only against a generator (@fig:method).
- *The context-aware generator is what makes the memory matter.* With it, the
  discriminator relies on its state, and separates context images from unseen
  real images it was never trained to reject. With a blind generator it does
  not (@sec:results).
- *The memory captures digit identity without labels.* Given 16 examples of
  one digit as context, the trained discriminator classifies test digits better
  than pixel-space nearest-neighbour and RBF density estimates that see the
  same 16 examples (@sec:classifier).
- *A reliable recipe.* With Adam $beta_1 = 0, beta_2 = 0.99$, none of 12
  runs failed (@sec:stability).

Section 2 defines GANs, delta-rule linear attention and in-context learning.
Sections 3 and 4 give the task and the method in full. Sections 5–7 report
the runs and what they show. Section 8 states the limits.
