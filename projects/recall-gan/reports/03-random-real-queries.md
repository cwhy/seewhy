# Random real queries do not stop the copying

**Finding.** Making the real queries at each prefix random did not stop the
generator from copying the newest context image. It copied more: its image
matches the last context image's digit 88% of the time, against 51% when the
newest image was always a real query, and about 15% by chance. The pictures
show why the change could not help. The generator does not just pick the newest
image's digit; it reproduces the image itself, almost pixel for pixel. Such a
copy *is* a context image in all but name, yet it is labelled "not in context".
The discriminator cannot be right about it, and is not: it ranks the newest
real image above the copy only 44% of the time, below chance. Every
discriminator measure ends lower than with plain training.

Same seeds (50–55), 20k steps, the method's recipe otherwise.

## What was changed

Agreed with you after report 2: at each prefix, the two real queries are two
random images from the context so far, instead of the image just written plus
one random earlier image. Nothing else changed. Both versions were run, with
and without the nearest-image context.

## The generator copies the image just written

![Twins, random reals](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r03_twins_prefix_rand_v1.png)

*Prefix training with random real queries, the run with the median twin score
(seed 52), one held-out sequence. Top: the image written at each step. Bottom:
the fake queried at that step, which the generator drew right after the
previous image was written. Read it diagonally: the fake in each column
reproduces the image one column to the left — the same 8, 3, 5, slanted 7.
Numbers: the discriminator's P(in context).*

The discriminator cannot tell them apart: the copies score 0.34–0.58, the real
images 0.30–0.83, overlapping throughout.

![Twins, nearest image + random reals](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r03_twins_prefix_nn_rand_v1.png)

*With the nearest-image context as well (seed 50). As in report 2, after the 4
random warm-start images the generator steers the context to one digit, here a
7, and the discriminator gives up (0.44–0.50 for real and copy alike).*

## Copying, all five setups

![Copying](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r03_copy_match_v1.png)

*Digit match between the generator's image and the context, after all 16
context images; 512 held-out contexts, 8 images each. Labels from a small MNIST
classifier (98.1% test accuracy).*

Plain training shows no preference for the last image. Every prefix setup
does, and random real queries raise it. In each of the two new setups, one run
of six collapsed to a single digit for every context (the whiskers reaching
down to 0.09); it is counted in the medians.

## What the discriminator learned

| median of 6 runs | plain training | prefix, newest real | prefix, random reals | prefix + nearest, newest real | prefix + nearest, random reals |
|---|---|---|---|---|---|
| digit classifier, 1 per digit (chance 0.1) | 0.478 | 0.084 | 0.122 | 0.330 | 0.256 |
| digit classifier, 16 per digit | 0.843 | 0.529 | 0.418 | 0.429 | 0.310 |
| generated digit = last context image's | 0.137 | 0.513 | 0.882 | 0.907 | 0.832 |
| generated digit = an earlier one's | 0.147 | 0.130 | 0.119 | 0.102 | 0.099 |
| distinct digits among 8 samples | 4.1 | 2.9 | 1.5 | 1.3 | 1.5 |
| context vs unseen real (AUROC) | 0.954 | 0.872 | 0.795 | 0.798 | 0.710 |
| context vs generated (AUROC) | 0.971 | 0.891 | 0.960 | 0.716 | 0.938 |
| share of fakes rejected | 0.824 | 0.948 | 0.963 | 0.116 | 0.519 |
| twin AUROC | — | 1.000 | 0.436 | 0.778 | 0.602 |

*Medians of 6 runs. Twin AUROC: the newest context image against the fake
queried with it (chance 0.5). The last four rows use random 16-image contexts.*

![Classifier by context size](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r03_context_use_v1.png)

*The digit classifier with 1 to 16 images per digit (chance 0.1). Dashed:
random real queries.*

![Score by position](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r03_position_v1.png)

*Each of the 16 context images used as a query after all 16 writes. All are in
the context.*

With random real queries the discriminator now distrusts the most recent
images: the last context image scores 0.35 in the prefix-only setup, against
0.71 for the first. Those are the images the generator copies, so a recent
image and a fake look alike to it.

## Over training

![Curves](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r03_curves_v1.png)

*Median of 6 runs. Dashed: random real queries.*

With random real queries, the twin score starts near 1 and falls through
training (below 0.5 after about 13k steps). The copying is learned, step by
step. The digit classifier rises to 0.74 around step 3k and then declines with
it.

## Generated images

![Samples](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r03_samples_v1.png)

*The median run (by digit classifier) of plain training and of the two new
setups, same contexts and noise.*

Both new setups draw one digit per context.

## What this changes

- The copying in report 2 was not caused by always querying the newest image.
  Under prefix training the generator copies the image just written, at the
  pixel level, whichever real queries are used. Plain training, where the
  generator only acts after the whole context, does not copy.
- A copied context image labelled "not in context" is a label the
  discriminator cannot satisfy. That is what drives its scores down, and it is
  the situation the very first design discussion of this project was about.
- Options, all yours to choose:
  1. Count a near-copy of a context image as "in context" (judged by distance
     to the nearest context image). That changes the labels.
  2. Drop prefix training and return to plain training, which works (digit
     classifier 0.843) and does not copy.
- Any other fix inside prefix training has to deal with the same fact: every
  context image the generator can read is also one it can copy.
