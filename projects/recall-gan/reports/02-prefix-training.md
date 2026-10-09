# Under prefix training the generator copies the last image

**Finding.** Training on every prefix of the context made every measure worse
than plain training, and the main reason is copying. After all 16 context
images, the prefix-trained generator draws the same digit as the *last* context
image 51% of the time (91% with the nearest-image context), against 13–15% for
any earlier image or by chance. That teaches the discriminator that "the same
digit as the newest image" means fake. With one example per digit it then
classifies *below* chance (0.084 against 0.10), the opposite of what the digit
classifier rewards, and with 16 examples it reaches 0.529 against 0.843 for
plain training. With the nearest-image context the generator goes further and
fills the context with its own digit; the discriminator rejects only 12% of its
fakes.

Same seeds (50–55), 20k steps, the method's recipe otherwise. Plain-training
numbers are the paper's runs on the same seeds.

## What was tried

Agreed with you: train on every prefix ("teacher forcing"), and build the
context from the real image nearest to the generator's image, after a random
warm start. Choices of mine that you approved: a warm start of 4 random images,
and per prefix 2 real queries (the image just written, one random earlier one)
and 2 fakes (the "twin" — the generator image matched to the image just written
— and a fresh one). The ablation trains on every prefix with a fully random
context.

## The context the generator builds

![Twins, nearest-image context](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r02_twins_prefix_nn_v2.png)

*Nearest-image arm, the run with the median twin score (seed 50), one held-out
sequence. Top: the image written at each step. Bottom: its twin, the generator
image whose nearest real image it is. Numbers: the discriminator's
P(in context) at that step.*

After the warm start the generator proposes a 3, then 1s. Each proposal pulls a
real 1 into the context. From step 7 on, every written image is a 1 and every
twin is a 1. The discriminator gives up on them: 0.48–0.49 for real and fake
alike.

![Twins, random context](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r02_twins_prefix_v2.png)

*Prefix-only arm (seed 52), same view. The context is random; the "twin" is just
the previous step's fake, with no pixel relation to the image above it.*

Here the discriminator keeps the two apart easily (0.73–0.97 for written
images, 0.03–0.44 for twins).

## Generated images

![Samples](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r02_samples_v2.png)

*The median run of each arm on the same three held-out contexts and the same
noise.*

Both prefix arms draw essentially one digit per context: all zeros, all nines,
all eights. Plain training is more varied but noisier. My variety measure did
not catch this: it pools samples from many contexts, so one digit per context
still looks varied overall.

## Over training

![Curves](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r02_curves_v2.png)

*Median of 6 runs. Grey: plain training. Blue: prefix only. Orange: prefix +
nearest-image context. The twin score exists only for the two prefix arms.*

| median of 6 runs, end of training | plain training | prefix only | prefix + nearest image |
|---|---|---|---|
| twin AUROC (newest image vs its twin; chance 0.5) | — | 1.000 (twin unrelated) | 0.778 |
| context vs unseen real images (AUROC; chance 0.5) | 0.954 | 0.872 | 0.798 |
| digit classifier (chance 0.1) | 0.843 | 0.529 | 0.429 |
| share of fakes rejected (chance 0.5) | 0.824 | 0.948 | 0.116 |

The share of fakes rejected falls steadily in the nearest-image arm, from 1.0
to 0.12, as the generator learns to fill the context with its own digit.

Caveat: the last three rows use random 16-image contexts. The nearest-image arm
never trained on those, so they are partly off-distribution for it. The twin
AUROC is its in-distribution measure, and it ends at 0.778.

## Is the generator copying the last image?

Yes, at the level of the digit. A small digit classifier, trained here on
MNIST's training split (98.1% test accuracy), labels the generated
images. The context is 16 random held-out images; each run draws 8 images for
each of 512 contexts.

![Digit match](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r02_copy_match_v1.png)

| median of 6 runs | plain training | prefix only | prefix + nearest image |
|---|---|---|---|
| generated digit = the **last** context image's digit | 0.137 | 0.513 | 0.907 |
| generated digit = an **earlier** context image's digit | 0.147 | 0.130 | 0.102 |
| base rate (share of the context with that digit) | 0.135 | 0.155 | 0.155 |
| distinct digits among 8 images from one context | 4.1 | 2.9 | 1.3 |

Plain training shows no preference for the last image: its rate equals the
base rate. Both prefix arms strongly prefer it. Copying also explains the
narrow samples: with 8 draws from one context, the nearest-image generator
produces 1.3 different digits on average, against 4.1 for plain training.

## Is that why the digit classifier drops?

The digit classifier gives the discriminator one context per digit and asks
which context scores a test image highest. If the discriminator used only the
last image it wrote, adding more images per digit would not help. It helps in
all three arms up to 8 images per digit; in the two prefix arms it then stops
rising or drops, while plain training keeps improving to 16:

![Accuracy by context size](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r02_context_use_v1.png)

| images per digit | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| plain training | 0.478 | 0.537 | 0.662 | 0.752 | 0.843 |
| prefix only | 0.084 | 0.209 | 0.447 | 0.548 | 0.529 |
| prefix + nearest image | 0.330 | 0.363 | 0.463 | 0.509 | 0.429 |

So the discriminator does read more than the last image. The telling number is
the first column. Given a single image of a digit, the prefix-only
discriminator classifies *below chance*: it scores other images of that digit
*lower* than images of other digits. That is what the copying teaches. At every
prefix, the fakes are the generator's copies of the newest image's digit, while
the real queries are the newest image itself and an earlier one. "Same digit
as the newest image, but not that image" is therefore the mark of a fake. The
digit classifier rewards the opposite.

The same pattern shows in how the discriminator scores the context images
themselves:

![Score by position](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r02_recency_v1.png)

*Each of the 16 context images used as a query after all 16 writes. All are in
the context.*

Plain training scores every position alike (about 0.75). The prefix-only
discriminator scores the last image 0.90 and every earlier one below 0.5. The
lowest is the image written just before the last (0.40), the one whose digit
the final fake copies.

Across runs the picture is less clean:

![Per run](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_r02_copy_vs_classifier_v1.png)

The three arms separate clearly, but within the 12 prefix runs the amount of
copying predicts the classifier only weakly (correlation -0.40).
So copying separates prefix training from plain training, but how much a
given run copies does not set its exact score. None of this is a controlled
test. That would mean retraining with copying of the last image prevented,
which has not been run.

## What this changes

- Training on every prefix gives the generator a simple winning move: copy the
  newest image's digit. That move teaches the discriminator to reject the
  newest image's digit, which works against using the context as a whole.
  Plain training, where queries come only after the full context, has no
  "newest image" to copy, and shows no copying.
- The nearest-image context makes it worse: copying the last digit also decides
  the next image written, so the context fills with one digit.
- Generator variety must be measured within a context. Pooled across contexts
  it hid all of this.
- If prefix training is worth keeping, the fix would have to remove the reward
  for copying the newest image, for example by drawing the real queries at each
  prefix from all earlier images rather than always including the newest. That
  is a design change; I have not run it.
