# Pretraining the discriminator does not survive the GAN

**Finding.** A discriminator pretrained to tell context images from other real
images becomes an excellent membership test on its own: AUROC 0.995 for
context images against unseen real images, before any generator exists. Within
the first 500 steps of GAN training that skill is gone (AUROC about 0.5 in all
6 runs). Afterwards 4 of 6 context-aware generators collapse, where none did
when the same GAN starts from scratch, and those discriminators end at chance.
The 2 runs that recover end where training from scratch ends (digit classifier
0.815 and 0.829, against a median of 0.843 from scratch). Pretraining helped in
no run.

## The setting

Each training episode draws 16 random MNIST images, the *context*. The
discriminator writes them, one at a time, into a fixed-size memory: one KDA
(delta-rule linear attention) state of 4 × 64 × 64 = 16,384 numbers. It then
answers, for each query image, "was this image in my context?", reading only
from that state.

The GAN: the context images are the *real* images, and the generator's images
are the *fakes*. Each episode queries 4 context images ("in") and 4 generated
images ("not in"). The generator either reads the same context through its own
KDA state (context-aware) or gets only noise (blind). Training uses a gradient
penalty of weight 10 on the real queries and Adam (learning rate 2e-4,
β1 = 0, β2 = 0.99), for 20,000 steps. All figures use 6 runs (seeds 50–55)
and held-out test images.

From scratch, this works with the context-aware generator: the discriminator
ends up separating context images from unseen real images it was never trained
to reject (AUROC 0.954), and, given 16 examples per digit, classifies digits at
0.843 (chance 0.1). With the blind generator it fails: the generator collapses
to one image and the discriminator never uses its memory.

## The idea

Yours: before the GAN, train the discriminator alone on data from the dataset
that is *not* in the context, so it starts the game already knowing what
membership means.

Choices of mine: 5,000 pretraining steps; each episode asks about 4 context
images ("in") and 4 other real images ("not in"); plain cross-entropy, no
gradient penalty (there is no generator to stabilise); the same Adam setting.
The GAN then runs exactly as above, starting from the pretrained discriminator
with a fresh optimiser. Non-member images are not used again.

## Pretraining alone works

![Pretraining](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r05_pretrain_v1.png)

*During pretraining, before any generator (median of 12 runs: the same
pretrained discriminator starts both the context-aware and the blind runs).*

Context against unseen real images reaches AUROC 0.995. The digit classifier
starts at 0.77 and drifts down to 0.71 as membership sharpens: the pretrained
memory tells *which images* were stored better than *which digits*.

## Then the GAN erases it

![Curves](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r05_curves_v1.png)

*GAN phase, median of 6 runs. Step 0 is the pretrained discriminator.*

| median [min, max] of 6 runs | from scratch, context-aware G | pretrained D, context-aware G | from scratch, blind G | pretrained D, blind G |
|---|---|---|---|---|
| after pretraining only: context vs unseen real | — | 0.995 [0.99, 1.00] | — | 0.995 [0.99, 1.00] |
| after pretraining only: digit classifier | — | 0.710 [0.69, 0.73] | — | 0.710 [0.69, 0.73] |
| end: context vs unseen real (AUROC) | 0.954 [0.94, 0.98] | 0.501 [0.50, 0.97] | 0.501 [0.50, 0.51] | 0.500 [0.50, 0.50] |
| end: digit classifier (16 per digit) | 0.843 [0.83, 0.86] | 0.117 [0.09, 0.83] | 0.101 [0.09, 0.12] | 0.090 [0.09, 0.12] |
| end: context vs generated (AUROC) | 0.971 [0.96, 0.98] | 1.000 [0.98, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| end: swap test | 0.584 [0.50, 0.66] | 1.000 [0.64, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| end: share of fakes rejected | 0.824 [0.63, 0.88] | 1.000 [0.73, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| end: generator variety | 0.177 [0.17, 0.18] | 0.001 [0.00, 0.18] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| end: generator collapsed (variety < 0.06) | 0 of 6 | 4 of 6 | 6 of 6 | 6 of 6 |

In every pretrained run, context against unseen real falls from 0.99 to about
0.5 within the first 500 GAN steps, and the digit classifier from about 0.72 to
0.08–0.19. At that point no generator has collapsed yet; the generators are
still producing their first blurred images. Collapse follows later, between
steps 2,500 and 6,000, in 4 of 6 context-aware runs. In the 2 that survive, the
discriminator rebuilds the skill from zero: the digit classifier is back above
0.7 by step 3,000–4,000, about 1,000 steps behind training from scratch.

![Classifier by context size](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r05_context_use_v1.png)

*Digit classifier with 1 to 16 images per digit, end of training.*

## Generated images

![Samples](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r05_samples_v1.png)

*The median run of each setup (by digit classifier), three held-out contexts,
same noise. The pretrained context-aware median run (seed 54) is one of the
four that collapsed.*

## What the discriminator ends up scoring

![Density](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r05_density_v1.png)

*One held-out context. Unseen test images scored highest, at the median and
lowest, by the median run of each setup. The pretrained run shown collapsed:
its discriminator gives every real image 0.96–0.99, whatever the context.*

## What this changes

- Starting from a discriminator that already knows membership does not carry
  that knowledge into the GAN. It is lost in the first few hundred steps, and
  the run is then less stable than starting from scratch (4 of 6 collapses
  against 0 of 6).
- Two candidate causes, neither tested:
  1. The gradient penalty switches on at the start of the GAN. The pretrained
     discriminator was trained without it, so it is sharp — its score changes
     steeply between stored and unstored images — and the penalty punishes
     exactly that.
  2. The first fakes are easy to reject by their look, and learning to reject
     them overwrites the membership features.
- Tests that would separate them, your call: pretrain *with* the same
  gradient penalty; or keep showing non-members during the GAN as well
  (pretraining as a curriculum rather than a one-off). The second changes the
  game the GAN plays.
