#import "/template.typ": *

= Background

This section defines the three ingredients the method combines. Nothing here
is new; it is what the rest of the paper relies on.

== Generative adversarial networks

A #gloss[generative adversarial network (GAN)][two neural networks trained
against each other @goodfellow2014gan] has a *generator* $G$ and a
*discriminator* $D$. The generator turns random noise $z$ into an image. The
discriminator receives an image and outputs a probability that it is *real*,
meaning drawn from the training data, rather than *fake*, meaning made by $G$.

The two are trained in alternation. $D$ is trained to output 1 on real images
and 0 on fakes. $G$ is trained to make $D$ output 1 on its fakes. Each network
is a moving target for the other: as $G$ improves, $D$ must find finer
differences, and as $D$ improves, $G$ must fix them.

In an ordinary GAN, the generator is the product and the discriminator is
discarded after training. In this paper it is the other way round.

Two failure modes matter here. In #gloss[mode collapse][the generator
produces one or a few images whatever the noise], $G$ stops exploring. When
$D$ becomes too confident, the signal it passes back to $G$ becomes flat, and
$G$ can no longer tell which direction is "more real".

A #gloss[gradient penalty][an extra loss term that keeps the discriminator's
output from changing too steeply as its input changes] is the standard remedy
for the second @mescheder2018r1. The form used here penalises the squared size
of $D$'s input gradient at real images:

$ cal(L)_"GP" = gamma / 2 dot EE_(x " real") norm(nabla_x D(x))^2 . $

== Linear attention with a delta-rule memory

#gloss[Attention][a way for a network to read a set of items, here images,
and combine information from them] normally keeps every item it has seen.
*Linear attention* instead compresses the items into one fixed-size matrix,
the #gloss[state][a $d_k times d_k$ matrix that summarises everything written
so far], and reads from that matrix alone @schlag2021fastweights.

Each item $t$ is turned into a *key* $k_t$ (where to store) and a *value*
$v_t$ (what to store). The *delta rule* writes an item by correcting what the
state currently returns at that key:

$ S <- S + beta_t (v_t - S k_t) k_t^top . $

Here $beta_t in (0, 1)$ is a learned write strength. If the state already
returns $v_t$ at $k_t$, nothing changes. Reading with a *query* vector $q$
returns $o = S q$.

*Kimi Delta Attention (KDA)* @kimi2025linear adds a learned per-channel
forgetting factor $alpha_t$ to the delta rule, as in gated delta networks
@yang2025gated:

$ S <- S dot "Diag"(alpha_t), quad S <- S + beta_t (v_t - S k_t) k_t^top . $

Whatever the number of items written, the state keeps the same size. Here it
is $4 times 64 times 64 = 16 space 384$ numbers.

== In-context learning

A network learns #gloss[in context][from examples given as input at test time,
with no change to its weights] when its behaviour on a query depends on a set
of examples it reads alongside it. The weights are fixed after training; only
the input changes. In this paper the "examples" are a set of images, and the
question asked of them is the simplest one possible: is this image one of them?
