# A blind generator collapses under prefix training too

**Finding.** With a generator that cannot see the context, prefix training
fails the same way plain training does. In all 12 runs (6 per query mix) the
generator collapsed to one image within 500–2,000 steps. The discriminator
then rejects that image on sight, ignores its KDA state, and learns nothing
about the context: context images against unseen real images at AUROC 0.500,
digit classifier at chance (0.104 and 0.087 against 0.10). The blind generator
cannot copy, but it puts no pressure on the discriminator at all.

Same seeds (50–55), 20k steps, the method's recipe otherwise.

## What was run

Asked by you after report 3: prefix training exactly as in reports 2 and 3, but
with the blind generator, which receives only noise. Two query mixes: the
newest image plus one random earlier image as real queries (as report 2), and
two random context images (as report 3). Compared with the same setups using
the context-aware generator, and with plain training using either generator.

## Generated images

![Samples](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r04_samples_v1.png)

*The median run of each blind setup, on three held-out contexts with the same
noise. Every sample of a run is the same image, for every context.*

![Twins](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r04_twins_prefix_blind_v1.png)

*Prefix training, newest image as a real query, blind generator (seed 53). Top:
the image written at each step. Bottom: the fake queried at that step. The
discriminator gives every real image 1.00 and every fake 0.00.*

The collapse is the same one the blind generator showed under plain training:
one image, regardless of context or noise.

## What the discriminator learned

| median of 6 runs | plain, context-aware G | plain, blind G | prefix (newest real), context-aware G | prefix (newest real), blind G | prefix (random reals), context-aware G | prefix (random reals), blind G |
|---|---|---|---|---|---|---|
| digit classifier, 1 per digit (chance 0.1) | 0.478 | 0.105 | 0.084 | 0.108 | 0.122 | 0.089 |
| digit classifier, 16 per digit | 0.843 | 0.101 | 0.529 | 0.104 | 0.418 | 0.087 |
| generated digit = last context image's | 0.137 | 0.094 | 0.513 | 0.104 | 0.882 | 0.090 |
| generated digit = an earlier one's | 0.147 | 0.089 | 0.130 | 0.108 | 0.119 | 0.102 |
| distinct digits among 8 samples | 4.1 | 1.0 | 2.9 | 1.0 | 1.5 | 1.0 |
| context vs unseen real (AUROC) | 0.954 | 0.501 | 0.872 | 0.500 | 0.795 | 0.500 |
| context vs generated (AUROC) | 0.971 | 1.000 | 0.891 | 1.000 | 0.960 | 1.000 |
| share of fakes rejected | 0.824 | 1.000 | 0.948 | 1.000 | 0.963 | 1.000 |
| twin AUROC | — | — | 1.000 | 1.000 | 0.436 | 1.000 |

*Medians of 6 runs. Plain training with a blind generator uses the paper's
control runs (exp18/19) on seeds 50–55.*

![Over training](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r04_curves_v1.png)

*Median of 6 runs; dashed: blind generator. The three blind setups lie on top
of each other at chance.*

In every blind run, generator variety falls below 0.06 between step 500 and
step 2,000, and stays there.

![Classifier by context size](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r04_context_use_v1.png)

## What this changes

- The blind generator cannot test "prefix training without copying". It
  collapses before prefix training can matter, exactly as under plain training.
- So in every setup tried, the discriminator only learns about its context when
  the generator reads the context. Under plain training that works (digit
  classifier 0.843). Under prefix training, the same reading lets the generator
  copy the image just written (report 3).
- Separating "prefix training" from "copying" would need a generator that reads
  the context but cannot copy from it. That is a design change and has not been
  tried.
