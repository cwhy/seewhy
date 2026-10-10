# Trained on synthetic worlds, the discriminator learns the world, not the item

**Finding.** Trained on a synthetic prior and tested on MNIST, the method
transfers nothing: the discriminator separates context images from unseen MNIST
images at AUROC 0.51–0.52 (chance 0.5; 0.954 when trained on MNIST) and
classifies digits at 0.12–0.15 (chance 0.1; 0.843). It fails on its own kind of
data too. On held-out synthetic worlds it scores the context items and unseen
items *of the same world* identically (AUROC 0.502), while separating them from
items of a *different* world (median 0.81–0.87). Every training context comes
from one world, so the game taught it "does this belong to this world?", never
"was this item stored?". On MNIST, every digit belongs to the same world.

## The setting

Each training episode has 16 *context* items. The discriminator writes them,
one at a time, into one fixed-size memory: a KDA (delta-rule linear attention)
state of 4 × 64 × 64 = 16,384 numbers. For each query it answers "was this in
my context?" from that state alone. In the GAN, the context items are the
*real* data and the generator's outputs are the *fakes*: 4 context items
("in") and 4 generated items ("not in") per episode. The generator reads the
same context through its own KDA state. Gradient penalty 10 on the real
queries, Adam (learning rate 2e-4, β1 = 0, β2 = 0.99), 20,000 steps. Trained
on MNIST, this gives a discriminator that rejects unseen real digits it was
never trained on, and works as a digit classifier from 16 examples per digit.

## What was tried

Yours: train on synthetic data, test on MNIST. The synthetic data is
recall-gen's prior, unchanged except for width (784, MNIST's). A *world* is a
random latent-factor model: k latent dimensions, a random mixing matrix, one of
four nonlinearities, noise, rescaled to [0, 1]; 1,024 training worlds and 128
held-out worlds, 48 items each. Choices of mine: each episode's 16 context items
come from one world, as in recall-gen; two versions of the prior, k ≥ 1
(recall-gen's default) and k ≥ 8 (recall-gen found smaller k makes items nearly
identical); the networks never see an MNIST image in training. 6 runs each
(seeds 50–55), against the MNIST-trained runs on the same seeds.

![Worlds, k >= 1](https://media.tanh.xyz/seewhy/26-10-11/recall-gan_r06_worlds_synth_v1.png)

*Four held-out worlds of the k ≥ 1 prior, 16 items each, drawn as 28 × 28.*

![Worlds, k >= 8](https://media.tanh.xyz/seewhy/26-10-11/recall-gan_r06_worlds_synth_k8_v1.png)

*The same for k ≥ 8. Within a world the items share a texture; nothing in them
looks like a digit.*

## Results

| median [min, max] of 6 runs, end of training | trained on MNIST | trained on synthetic, k >= 1 | trained on synthetic, k >= 8 |
|---|---|---|---|
| MNIST: context vs unseen real (AUROC) | 0.954 [0.94, 0.98] | 0.512 [0.50, 0.52] | 0.519 [0.51, 0.53] |
| MNIST: digit classifier, 16 per digit | 0.843 [0.83, 0.86] | 0.123 [0.11, 0.14] | 0.147 [0.10, 0.18] |
| MNIST: context vs generated (AUROC) | 0.971 [0.96, 0.98] | 0.669 [0.50, 0.72] | 0.973 [0.81, 1.00] |
| MNIST: swap test (lower = relies on state) | 0.584 [0.50, 0.66] | 0.665 [0.51, 0.72] | 0.967 [0.84, 1.00] |
| MNIST contexts: generator variety | 0.177 [0.17, 0.18] | 0.159 [0.11, 0.17] | 0.145 [0.10, 0.20] |
| held-out worlds: context vs unseen items | — | 0.503 [0.50, 0.51] | 0.502 [0.50, 0.51] |
| held-out worlds: context vs generated | — | 0.647 [0.57, 0.82] | 0.751 [0.59, 0.84] |
| held-out worlds: generator variety | — | 0.207 [0.18, 0.21] | 0.219 [0.20, 0.23] |
| MNIST: digit classifier, 1 per digit (probe) | 0.478 [0.41, 0.51] | 0.120 [0.07, 0.15] | 0.094 [0.07, 0.14] |
| MNIST: digit classifier, 16 per digit (probe) | 0.843 [0.83, 0.86] | 0.123 [0.11, 0.14] | 0.147 [0.10, 0.18] |

![Curves](https://media.tanh.xyz/seewhy/26-10-11/recall-gan_r06_curves_v1.png)

*Median of 6 runs. The synthetic-trained discriminators never separate context
items from unseen items, on MNIST or on held-out synthetic worlds, at any point
in training.*

![Classifier by context size](https://media.tanh.xyz/seewhy/26-10-11/recall-gan_r06_context_use_v1.png)

## What the discriminator learned instead

Scored on 512 episodes from held-out worlds, the context items against:

| AUROC, median of 6 runs (range) | k ≥ 1 | k ≥ 8 |
|---|---|---|
| unseen items of the same world | 0.502 (0.501–0.506) | 0.503 (0.500–0.508) |
| items of a different world | 0.813 (0.57–0.92) | 0.867 (0.78–0.90) |
| MNIST images | 0.997 (0.91–1.00) | 0.917 (0.29–1.00) |

It gives context items and unseen same-world items the same probability, 0.50,
and gives other worlds' items less. That is a test of world membership. The
generator explains why it is enough: it imitates the context's world (below),
so "belongs to this world" separates its fakes no better than chance either,
and nothing in the game rewards telling individual items of a world apart.

## Generated items

![Synthetic samples](https://media.tanh.xyz/seewhy/26-10-11/recall-gan_r06_samples_synth_v1.png)

*Each synthetic-trained generator (median run) on three held-out worlds:
textures in the world's style.*

![MNIST samples](https://media.tanh.xyz/seewhy/26-10-11/recall-gan_r06_samples_v1.png)

*The same generators on three MNIST contexts, with the MNIST-trained generator
for comparison. The synthetic-trained ones produce dim noise.*

![Density on MNIST](https://media.tanh.xyz/seewhy/26-10-11/recall-gan_r06_density_v1.png)

*One MNIST context: unseen test images scored highest, at the median and lowest
by each median run.*

## What this changes

- With one world per episode, the synthetic GAN teaches world membership, which
  is useless on MNIST, where all digits share one world.
- Item-level membership would need episodes where "belongs to this world" does
  not decide the answer, for example contexts mixing items from several worlds,
  so that unseen items of the same worlds are the hard negatives the generator
  has to imitate. That changes the episode design; your call.
- An open difference: MNIST contexts are also "one world", yet trained on
  MNIST the discriminator learned item- and class-level membership. MNIST has
  structure inside its world (ten digits, many writing styles); these synthetic
  worlds have none. Untested.
