# The discriminator wins; the generator learns one trick

**Status after eight experiments.** A discriminator that sees the context only through one KDA memory
learned to tell stored images from unstored ones almost perfectly. Among images from the *same* generator,
it separates stored from fresh at AUROC **0.998** (chance 0.5). On real images it scores
**0.996**. The generator took six fixes to get playing at all. Once playing, the context-aware one
uses its context: **0.67** of its samples sit nearer their own context than an unrelated set,
against **0.51** for the context-blind one and chance **0.49**. But what it learned is narrow.
In two of three seeds it is one rule: *if the context holds a 1, draw a 1.* The discriminator still
rejects 91% of its images.

All numbers: MNIST, 16-image contexts, held-out test-split episodes, 20k training steps unless stated.
"Seeds" means three independent runs (42, 43, 44).

## Where each piece came from

Several pieces of the setup were added by me (Claude) while running on my own. Marked here so
nothing reads as agreed when it was not.

| piece | source |
|---|---|
| discriminator = one KDA memory; its final state decides "was this in the context?" | **yours** |
| generator without the context; also a version with it | **yours** |
| the discriminator should win; it should be a proper density estimator over all images | **yours** |
| exp1: generated images always "not in context", plus the two diagnostics | agreed in discussion |
| plain-GAN check (exp2) after you doubted the failure was the idea's | agreed in discussion |
| **gradient penalty**: extra loss that keeps the discriminator's score from changing too sharply near stored images | introduced by me; then discussed (you asked what it is and whether modern GANs use it) |
| putting generated images into the context | mine — proposed, never agreed |
| **frozen generator**: a second generator, trained once as a plain GAN (exp3) and then locked; it only supplies fake images for the context | mine, not discussed |
| **warm start**: the generator being trained begins as a copy of the frozen generator, not from random weights | mine, not discussed |
| penalty sweep, 3 seeds, the "add a 1" test; MNIST, 16 images, network sizes | mine, not discussed |

## 1. The game, in one picture

The discriminator reads 16 images into its memory, then answers "was this image one of them?" for each
query. The generator makes images and is rewarded when the discriminator wrongly says yes.

The final setup (exp8) adds two things of mine. First, 4 of the 16 context images are fakes from the
**frozen generator** — a separate generator trained earlier and then locked. Fakes are now sometimes in
the context and sometimes not, so "looks fake" no longer tells the discriminator the answer. Second, the
generator being trained starts as a copy of that frozen generator (**warm start**).

![One episode](https://media.tanh.xyz/seewhy/26-10-08/recall-gan_r01_episode_v2.png)

*One held-out episode (first draw of a fixed random seed, not chosen), scored by the exp8 context-aware run,
seed 44. Bold number = the discriminator's probability that the query was in the context; green when on the
right side of 0.5. The two "frozen-G" images come from the same frozen generator; only one of them was
stored. The last query is the trained generator's attempt.*

## 2. What each experiment changed

| exp | what changed | what happened |
|---|---|---|
| 1 | the game as first proposed: generated images always labelled "not in context" | membership perfect (AUROC 1.000 without a generator); both generators collapsed to one noisy blob |
| 2 | **control:** same generator in a plain real-vs-fake GAN | collapsed too → exp1's failure was my GAN setup, not the idea |
| 3 | plain GAN + one stabiliser at a time | only the **gradient penalty** (weight 10) worked: varied, readable digits |
| 4 | exp1 + gradient penalty | generator alive, but the discriminator won by spotting "looks generated" and called 69% of unstored real images "stored" |
| 5 | 4 of 16 context slots filled with the generator's *own* images | every generator collapsed: one image, planted in every context, can never be rejected |
| 6 | penalty weight 0 / 1 / 10, no generator | the penalty costs membership: 99.7% → 94.6% → 77.5% of unstored images rejected |
| 7 | generated context images from a *frozen* generator | discriminator: AUROC 1.000 on stored vs fresh generated images; trained generator still collapsed (11 of 12 runs) |
| 8 | exp7 + the trained generator starts as a copy of the frozen one | **penalty 1: generator alive, context-aware one uses its context.** Penalty 0: still collapses |

## 3. Getting the generator to play

The generator's variety over training shows the story. A line at zero is a generator drawing one image.

![Generator variety over training](https://media.tanh.xyz/seewhy/26-10-08/recall-gan_r01_diversity_v2.png)

*Seed 42 for every line. Real MNIST's own variety is the dotted line.*

Three different things killed it, in order:

1. **An unstable GAN.** Without the gradient penalty, the discriminator became a cliff: certain on real
   images, certain on everything else, with no slope for the generator to follow. Even a plain GAN collapsed
   (exp2). The penalty keeps the discriminator's score smooth near real images (exp3).
2. **A shortcut.** With generated images always "not in context", the discriminator could win by
   recognising generated images, without reading its memory (exp4).
3. **A loophole I built.** Putting the generator's own images in the context let a collapsed generator plant
   its single image everywhere, so it could never be rejected (exp5). A frozen, separate generator for the
   context images closed it (exp7). Starting the trained generator as a copy of that frozen one got it
   playing (exp8).

What the generators draw:

![Samples](https://media.tanh.xyz/seewhy/26-10-08/recall-gan_r01_samples_v2.png)

*Same four held-out contexts and the same noise for every panel; rows = contexts, which were drawn at
random and all happen to contain a 1. Left: exp1's collapse. Middle: the frozen generator. Right: exp8's
context-aware generator, drawing 1s. Section 5 shows what it draws when the context has no 1.*

## 4. The discriminator is a real membership test — with a price

On the final setup the discriminator is near-perfect on every kind of query:

| exp8, penalty 1, 3 seeds | context-aware generator | context-blind generator | chance |
|---|---|---|---|
| stored vs unstored **real** images (AUROC) | 0.996 (seeds 1.00–1.00) | 0.999 (seeds 1.00–1.00) | 0.5 |
| stored vs fresh **frozen-generator** images (AUROC) | 0.998 (seeds 1.00–1.00) | 0.999 (seeds 1.00–1.00) | 0.5 |
| trained generator's images rejected | 0.921 (seeds 0.90–0.95) | 0.977 (seeds 0.97–0.98) | 0.5 |

The generator needs the gradient penalty, and the penalty works against membership. Membership needs the
discriminator to react sharply to small differences; the penalty smooths exactly that away.

![Penalty cost](https://media.tanh.xyz/seewhy/26-10-08/recall-gan_r01_penalty_v2.png)

## 5. The generator reads its context — for one thing

Context use grows through training for every context-aware seed, and stays at chance for every blind one:

![Context use over training](https://media.tanh.xyz/seewhy/26-10-08/recall-gan_r01_closer_v2.png)

What it reads, shown directly. Each episode: a context with no 1, then the same context with one image
swapped for a 1, and **the same noise** fed to the generator both times. Only the context changed.

![Same noise, add a 1](https://media.tanh.xyz/seewhy/26-10-08/recall-gan_r01_ones_v2.png)

*exp8 context-aware generator, seed 44. Three held-out episodes, first draws of a fixed seed. Green frame =
the swapped-in 1.*

Without a 1 in the context, this generator draws noisy blobs, not digits. It has not learned to draw
digits in general; it has learned to draw a 1 when that is likely to pay. The discriminator rejects the
blobs, which costs the generator little, because they are only a share of its attempts.

Across all six generators, on 512 held-out contexts:

![Share of 1s](https://media.tanh.xyz/seewhy/26-10-08/recall-gan_r01_ones_bars_v2.png)

| exp8, penalty 1 | share of samples that are a 1: context without a 1 → with one |
|---|---|
| context-blind, seed 42 | 0.029 → 0.029 |
| context-blind, seed 43 | 0.007 → 0.007 |
| context-blind, seed 44 | 0.099 → 0.099 |
| context-aware, seed 42 | 0.107 → 0.101 |
| context-aware, seed 43 | 0.065 → 0.666 |
| context-aware, seed 44 | 0.009 → 0.708 |

Seeds 43 and 44 learned "a 1 in the context → draw a 1". Seed 42 does not react to a 1. It still sits
above chance on the training mix (0.61), so it reads
something else, not yet identified. Using the context pays: the context-aware generator fools the
discriminator on 9.0% of its images, the blind one on 3.0%.

## 6. Pending conclusions

- **The discriminator side of the idea works.** A single KDA memory of 16k numbers supports a near-perfect
  "was this stored?" test for 16 images, including among images from one generator (AUROC 0.998).
- **"Discriminator wins" is the outcome, and by a wide margin.** Even the best generator is rejected
  91% of the time.
- **A generator can learn to read the context to attack it.** Not by copying images, but by matching a
  class — in two of three seeds.
- **Every earlier "the generator can't" was a setup problem.** Each failure in exp1–exp7 traced to the GAN
  recipe or to my episode design, not to the idea.

## 7. Open questions

1. **Why 1s?** Guess: 1s look alike, so an unstored 1 is the easiest image to mistake for a stored one —
   the discriminator's blind spot. Testable now: its error rate on unstored 1s vs other digits.
2. **What does seed 42 read?** It uses the context, but not via 1s.
3. **Can the generator do more than one trick?** It found one class and stopped. Longer training, or a
   reward for variety, may push it to other weak spots.
4. **Can we keep the generator alive without the penalty**, so membership stays at full strength?
5. **Larger contexts.** 16 images is far below the memory's size. Where does membership start to fail, and
   does the generator find it first?

## What this changes

- The next run should test question 1 directly. If 1s are the blind spot, the game is doing what it was
  built for: the generator finds where the discriminator is weakest.
- The standard setup from here on: frozen-generator context images, warm start, gradient penalty 1.

*Source: `projects/recall-gan/experiments1-8.py`, `results.jsonl`, `concepts.md`. Generator:
`scripts/gen_report_1.py`.*
