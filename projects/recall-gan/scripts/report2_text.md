# Prefix training lets the generator take over the context

**Finding.** Training on every prefix of the context made every measure worse
than plain training. With the nearest-image context, the generator learned to
steer what gets written. Once 4 random images are in, it keeps drawing one
digit, the nearest real image is that same digit, and the context fills up with
it. The discriminator then cannot tell its fakes from the real copies: it scores
both about 0.49. Across 6 runs it rejects only 12% of fakes, against 82% with
plain training.

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

![Twins, nearest-image context]({twins_nn})

*Nearest-image arm, the run with the median twin score (seed 50), one held-out
sequence. Top: the image written at each step. Bottom: its twin, the generator
image whose nearest real image it is. Numbers: the discriminator's
P(in context) at that step.*

After the warm start the generator proposes a 3, then 1s. Each proposal pulls a
real 1 into the context. From step 7 on, every written image is a 1 and every
twin is a 1. The discriminator gives up on them: 0.48–0.49 for real and fake
alike.

![Twins, random context]({twins_prefix})

*Prefix-only arm (seed 52), same view. The context is random; the "twin" is just
the previous step's fake, with no pixel relation to the image above it.*

Here the discriminator keeps the two apart easily (0.73–0.97 for written
images, 0.03–0.44 for twins).

## Generated images

![Samples]({samples})

*The median run of each arm on the same three held-out contexts and the same
noise.*

Both prefix arms draw essentially one digit per context: all zeros, all nines,
all eights. Plain training is more varied but noisier. My variety measure did
not catch this: it pools samples from many contexts, so one digit per context
still looks varied overall.

## Over training

![Curves]({curves})

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

## What this changes

- As built, the nearest-image context hands the generator control over the
  context, and it uses that control to make one digit dominate. That defeats
  the purpose: the twin pairs become indistinguishable copies of one class
  rather than a test of telling individual images apart.
- Prefix training on its own also lowers every measure and narrows the
  generator to one digit per context.
- Generator variety must be measured within a context, not pooled across
  contexts. I will add that measure before any further run.
- Questions for you: keep the nearest-image idea but stop the generator from
  steering (for example, choose the next image near the twin but never the
  same digit class twice, or limit how many steps it controls), or drop it and
  go back to the long-context runs?
