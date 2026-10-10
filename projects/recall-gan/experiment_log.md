# Recall-GAN — Experiment log

One entry per experiment, in run order: what was run, what came out, and what
it decided. Definitions of the current method and metrics are in
[concepts.md](concepts.md); this file is the history.

| exp | what | status |
|---|---|---|
| 1 | first version: extra "unstored real" negatives, Adam 2e-4/β1 0.5 | GAN broken (collapse) |
| 2 | control: same generator in a plain GAN | collapse → bug in recipe |
| 3 | plain GAN + stabilisers | gradient penalty fixes it |
| 4 | exp1 + gradient penalty | runs; extra negatives hurt membership |
| 5, 7, 8 | fakes in the context, frozen generator, warm start | **outside the design** (Claude's additions) |
| 6 | gradient-penalty weight, no generator | penalty costs membership |
| 9, 10 | **the game as proposed**: real = context images, fake = generated | context-aware works in 2/3 |
| 11 | generator rate 1e-4 | worse (1/3) |
| 12, 13 | exp10 rerun + generator context-use logging | 6/6; GPU non-determinism found |
| 14, 15 | recipe tricks: fake-side penalty, Adam β1 0/β2 0.99 | **Adam (C): 0/12 failures** |
| 16, 17 | relativistic loss, with/without C | no gain over C |
| 18, 19 | final recipe, blind generator (control) | all 12 collapse; D ignores its state |
| 20 | the method, 60k steps (3× longer) | G converges ~45k; D's generalisation declines after ~20–30k |
| 21, 22 | the method, M = 64 and 256 | item membership falls with M; class probe holds; M=256 still improving at 20k |
| 23, 24 | prefix training ("teacher forcing"), ± nearest-image context | both weaker on every end-of-context measure; twin AUROC 0.778 with nearest-image context |
| 25, 26 | as 23/24 with random real queries | copying stronger (0.88); generator copies the newest image pixel by pixel; twin AUROC 0.436 |
| 27, 28 | prefix training with the blind generator (both query mixes) | 12/12 collapse within 500–2,000 steps; D ignores its state |
| 29, 30 | discriminator pretrained on non-members, then the GAN | skill (AUROC 0.995) erased in 500 GAN steps; 4/6 context-aware collapse |

---

## Episode design used by exp1–8

One episode: M context images (M=16 in exp1) and Q queries.

| query | source | label |
|---|---|---|
| member | a copy of a random context image | 1 |
| real non-member | a random pool image | 1 iff its pool index is in the context (exact) |
| generated | G's output | 0 — the **plain rule**, under test in exp1 |

Training mix per episode: 4 members, 2 real non-members, 2 generated. The
`none` arm fills the generated slots with real non-members. Balanced 4/4, so
chance accuracy is 0.5.

Real non-members are always present, so D cannot win by real-vs-fake
detection alone; it still *can* use that shortcut on the generated slots.

All of exp1–8 used these extra "unstored real" negatives, which the design never
had (Claude's addition). From exp9 on, queries are only context images
("in context") and generated images ("not").

### The margin question (exp1's framing, later dropped)

## The margin question (what exp1 tests)

With the plain rule, a generator that learns to produce images close to
context images forces D to call near-copies "not in context". Once they are
closer than D's state can resolve, that is impossible and `acc_gen` falls to
chance. Whether a context-aware generator learns to move toward the context at
all is open: recall-gen found models often do not use their context. If it
does, labels switch to a distance rule (within ε of a context image = member).


---

## Entries

**exp1 (M=16, seed 42, 20k steps) — membership is easy; the generator failed,
cause unknown.** One untuned setting, no generator positive control: the
collapse below cannot yet be told apart from a bug or a tuning failure.

- `none`: `auroc_real` 1.000 on held-out episodes (0.994 at step 5k).
  Accuracy 0.999 on members, 0.997 on non-members. A 16k-float KDA state
  holds 16 MNIST images with room to spare.
- `blind` and `ctx` reach the same `auroc_real` 1.000, so the generator neither
  helped nor hurt membership.
- Both generators collapsed to one noisy blob, the same for every z and every
  context: `gen_std` 0.000 by step 10k, `p_gen` 0.000, `acc_gen` 1.000 from
  step 1k. G loss climbed to 11–16.
- `closer_gen` 0.521 (`ctx`) and 0.497 (`blind`), against `closer_real` 0.514.
  This does **not** answer Q1: a generator that ignores its own noise input was
  never in a position to use the context. Candidate causes, none yet ruled
  out: saturated output sigmoid (the blob's pixels sit at 0 and 1), D/G
  imbalance, a bug in the G update, or the real-vs-fake shortcut. Next: the
  same G and loop against a plain unconditional real-vs-fake D.

**exp2 — the plain GAN fails too, so exp1 was a broken GAN.** Same G and loop,
context-free real-vs-fake D, three learning-rate settings. All collapse:
`gen_std` ≤ 0.003 and `sat` 0.91–1.00 at 10k steps. Lowering D's rate alone
does not help.

**exp3 — R1 fixes it.** One stabiliser at a time in the plain game. LeakyReLU
in D: collapses (`sat` 0.999). LayerNorm in G: collapses (`gen_std` 0.008). R1
(weight 10): D loss 0.687, `p_gen` 0.493, `gen_std` 0.181, and the samples are
varied, recognisable digits. All three together: balanced too, but blurrier
samples. R1 is the operative fix.

**exp4 — exp1 with R1. The generator now plays, and two things show up.**
(One seed, one R1 weight.)

- *The real-vs-fake shortcut is real.* `blind`: G stays alive (`gen_std`
  0.170, digit-like but speckled samples), and D spends itself on spotting
  generated images: `auroc_gen` 0.980, but `auroc_real` only 0.656 and
  `acc_non` 0.309 — it calls most real non-members "in context".
- *R1 costs membership by itself.* `none` with R1: `auroc_real` 0.981 and
  `acc_non` 0.786, against 1.000 and 0.997 without. Plausible reason, untested:
  membership needs D to be sensitive to small input differences, and R1
  penalises D's input gradient on exactly the members.
- `ctx`: G alive to ~10k steps (`gen_std` 0.159 at 5k), collapsed by 20k
  (`gen_std` 0.000, `sat` 1.000). `closer_gen` 0.492–0.518 throughout, against
  `closer_real` 0.514: no sign of the generator using its context while alive.

**exp6 — the gradient penalty's cost to membership, 3 seeds, no generator.**
Penalty weight 0 / 1 / 10: `auroc_real` 1.000 / 0.999 / 0.979, `acc_non`
0.997 / 0.946 / 0.775. A smooth cost; seeds agree to ±0.01.

**exp5 — mixed context (4 of 16 slots generated), penalty 0 / 1 / 10, both
generators, 3 seeds. Every generator collapsed, and the design rewards it.**

- `gen_std` ≤ 0.013 and `sat` ≥ 0.94 in all 18 runs. They never developed
  diversity: `gen_std` stayed below 0.11 at every evaluation.
- `p_gen` sits at 0.50 from the first evaluation, and `auroc_genmem` is 0.50.
- Mechanism (a flaw in the exp5 design, not in the idea): the generated
  context items are the generator's own outputs. A generator that always
  emits one image x* puts x* in every context, so a fresh x* is pixel-identical
  to a stored member yet labelled "not in context". D cannot do better than
  0.5 on it, which is exactly what G is trained to achieve. Collapse is the
  generator's winning move, and the blind generator finds it as easily as the
  context-aware one.
- Membership on real images is unaffected at penalty 0 and 1 (`auroc_real`
  1.000 / 0.999), so removing the shortcut did work for D.

**exp7 — generated context items from a frozen generator (exp3's working one).
The discriminator is a real membership test; the trained generator still
collapses.** Penalty 0 / 1, both generators, 3 seeds.

- `auroc_genmem` 1.000 (penalty 0) and 0.999 (penalty 1): D tells stored
  frozen-generator images from fresh ones from the same generator — membership
  with no realness signal at all. `auroc_real` 1.000 / 0.998. Seeds agree.
- The trained generator collapses in 11 of 12 runs (`gen_std` 0.000, `p_gen`
  0.000); only `ctx` at penalty 1 keeps some variety (`gen_std` 0.087, range
  0.01–0.14). Early on it lives (`gen_std` ~0.1 at step 2k with penalty 1),
  then dies.
- Reading: the trained generator's images are the only ones never in a
  context, so D can still reject them by their look ("not the frozen
  generator's style"). Without a strong penalty D's slope is too steep for G to
  follow toward that style, the exp2 failure again. The penalty that keeps G
  alive (10) costs membership (exp6). Untested.

**exp8 — warm start (the trained generator starts as the frozen one). With
penalty 1 the generator stays alive and the context-aware one uses its
context.** 3 seeds.

- Penalty 0: generators still die (`gen_std` 0.047 blind, 0.004 ctx).
- Penalty 1: alive. `gen_std` 0.192 (blind), 0.156 (ctx). D still wins:
  rejects 97.7% (blind) and 92.1% (ctx) of generated images; `auroc_real`
  0.999 / 0.996, `auroc_genmem` 0.999 / 0.998.
- `closer_gen`: blind 0.506 [0.50, 0.51]; **ctx 0.674 [0.61, 0.71]**, chance
  0.498. The ctx generator fools D three times as often (`p_gen` 0.090 vs
  0.030). `closer_gen` rises steadily through training (seed 42: 0.51 at 2k
  steps → 0.61 at 20k).
- What it reads (scripts/tmp/diag_ones.py, 512 held-out all-real contexts):
  the samples are mostly "1"s. Contexts with no "1" vs exactly one "1":

  | run | P(sample is a 1) | closer |
  |---|---|---|
  | ctx s43 | 0.064 → 0.671 | 0.479 → 0.704 |
  | ctx s44 | 0.011 → 0.691 | 0.521 → 0.736 |
  | ctx s42 | 0.098 → 0.102 | 0.488 → 0.512 |
  | blind (all) | unchanged | ≈ 0.5 both |

  Two of three context-aware generators learned "if the context holds a 1,
  draw a 1". The third uses the context some other way, or not at all on
  all-real contexts. Open: whether "1" is where D's membership test is weakest
  (thin, look-alike digits), i.e. whether G found D's blind spot.

---

**Note on exp4–8.** Every run from exp1 to exp8 trained with extra unstored
real images labelled "not in context", which the user's design never had
(Claude's addition). exp5, 7 and 8 also added mechanisms of Claude's (fakes in
the context, frozen generator, warm start). They are kept as a record but are
outside the project's design. exp9/10 are the first runs of the game as
proposed.

**exp9/10 — the game as proposed: real = the 16 context images, fake =
generated.** Queries: 4 context images ("in context"), 4 generated ("not").
Gradient penalty 10 on the context-image queries. 3 seeds, 20k steps.
`*_swap` = same queries scored against another episode's context (evaluation
only). `auroc_real` = context images vs unseen real images (evaluation only;
never trained on).

| run | gen variety | context vs generated (AUROC) | same, wrong context | context vs unseen real | gen nearer own context (chance) |
|---|---|---|---|---|---|
| blind s42 | 0.000 (collapsed) | 1.000 | 1.000 | 0.500 | 0.490 (0.514) |
| blind s43 | 0.163 | 0.989 | 0.982 | 0.551 | 0.488 (0.529) |
| blind s44 | 0.000 (collapsed) | 1.000 | 1.000 | 0.500 | 0.508 (0.500) |
| ctx s42 | 0.164 | 0.976 | 0.949 | 0.575 | 0.511 (0.514) |
| ctx s43 | 0.183 | 0.945 | **0.593** | **0.905** | **0.667** (0.529) |
| ctx s44 | 0.178 | 0.968 | **0.659** | **0.891** | **0.605** (0.500) |

- Blind generator: alive in 1 of 3 seeds. There, D separates fakes almost as
  well with the wrong context (0.982 vs 0.989): it barely uses its memory.
- Context-aware generator: alive in 3 of 3. In seeds 43 and 44, D depends on
  its memory — with the wrong context it rejects its own context images
  (accept rate 0.32 / 0.38 vs 0.90 / 0.92) — and, without ever being trained
  on unseen real images, separates context images from them at AUROC
  0.905 / 0.891. In those same two seeds the generator's samples sit nearer
  their own context; its samples are recognisable digits of the classes present
  in the context. Seed 42 shows neither.
- 3 seeds per arm: a pattern, not yet a firm result.

**Evaluation-only probes of exp9/10 (scripts/tmp/density_probe.py,
class_contexts.py).**

- Density over ~10,000 real test images, 20 contexts: the two discriminators
  that rely on their KDA state (ctx s43/s44) score context images at median
  0.73–0.75 and unseen images at 0.23–0.32 (AUROC ~0.90–0.92); unseen images
  of a class present in the context average 0.34–0.38 vs 0.16–0.21 otherwise.
  The other two (ctx s42, blind s43) score every real image ~0.5–0.55
  regardless of context.
- **Single-digit contexts as a classifier** (16 random test images of one
  digit; 10-way = digit whose context scores highest; 5 draws): ctx s44
  0.839 (0.834–0.847), ctx s43 0.807 (0.789–0.829), vs nearest pixel neighbour
  among the same 16 images 0.783 (0.775–0.795); ctx s42 0.315, blind s43 0.379;
  chance 0.10. Per-digit AUROC 0.93–0.99 for s43/s44. Never trained with labels
  or single-digit contexts. Reading (user): where training works, the KDA state
  captures the in-context factor better than pixel L2.

**Metric decision (user, 2026-10-08): pixel L2 is not a reference for
similarity in this project.** It is one arbitrary measure, not the truth.
Use the binary truth (context image or not) and labels (single-digit
contexts) instead. `closer_gen`, the score-vs-distance curve and the
pixel-nearest-neighbour baseline rest on L2 and are superseded.

**exp11 — exp10 with the generator's learning rate halved (1e-4). Worse, not
more stable.** Only seed 43 works (class probe 0.790, swap 0.625, variety
0.190). Seeds 42 and 44 end collapsed (variety 0.002 / 0.015) with the
discriminator ignoring its KDA state (swap 1.000 / 0.993) and the class probe
at chance (0.115 / 0.105). Same seeds as exp10, but trajectories are not
paired: a different rate changes everything downstream. So the guess "one big
generator update causes the collapse" is not supported. In seed 43 the class
probe rises 0.2 → 0.79 between ~12k and 20k steps, together with the swap
score falling — it tracks the discriminator coming to rely on its KDA state.

**exp12/13 — exp10 rerun unchanged (seeds 42–47) with the generator's context
use logged (`g_ctx_dep`, evaluation only). All 6 worked.**

- The same seed does not reproduce: exp10 and exp12 agree at step 0 and
  differ by step 500 for every seed. No training code changed; the cause is
  GPU arithmetic that is not bit-for-bit repeatable, amplified by the GAN.
  exp10's seed-42 collapse was one unlucky run, not a property of the seed.
- At exp10's settings: **8 of 9 runs worked** (exp10 2/3, exp12/13 6/6).
  Generator rate 1e-4 (exp11): 1/3.
- Final, 6 runs: class probe 0.61–0.87 (chance 0.10); swap score 0.55–0.60;
  generator context use 0.82–0.95 (0 = ignores context); variety 0.17–0.19.
- Order of events in all 6: neither uses the context for ~5–7k steps; then
  the discriminator starts relying on its KDA state (swap falls, class probe
  rises); the generator's context use follows 1–3k steps later (0.2 → 0.8+).
  In the one failure (exp10 s42) the first happened and the second did not,
  then the generator collapsed — one case, not confirmed.

**exp14/15 — stabilising: GAN-recipe tricks, 12 runs each (seeds 50–61).**
Failure (fixed beforehand): class probe ends < 0.5 or variety ends < 0.06.

| recipe | failed | collapse events | class probe min / median / max | gen context use (median) |
|---|---|---|---|---|
| A original | 1/12 | 3 | 0.11 / 0.82 / 0.87 | 0.85 |
| B + penalty on generated images (R2) | 6/12 | 4 | 0.08 / 0.52 / 0.74 | 0.57 |
| C + Adam beta1 0, beta2 0.99 | **0/12** | **0** | **0.82 / 0.85 / 0.86** | 0.99 |
| D = B + C | 2/12 | 2 | 0.09 / 0.67 / 0.78 | 0.92 |

- C: no failure and no collapse event in 12 runs, and the tightest class-probe
  spread. Original recipe across all runs: 2 failures in 21 (exp10/12/13/14).
  With a ~10% base rate, 0/12 alone has a ~28% chance even if C changed
  nothing; the absence of collapse events (0 vs 3) and the tight spread are
  the stronger evidence. Not yet conclusive.
- B hurts: in its failures the generator keeps its variety but the
  discriminator rejects every fake without its KDA state (swap 1.00, fakes
  rejected 1.00).

**exp16/17 — relativistic loss (pairs: context image i vs generated image i),
12 runs each, seeds 50–61, gradient penalty on context images only.**

| recipe | failed | collapse events | class probe min / median / max |
|---|---|---|---|
| A original (exp14) | 1/12 | 3 | 0.11 / 0.82 / 0.87 |
| C Adam beta1 0, beta2 0.99 (exp15) | 0/12 | 0 | 0.82 / 0.85 / 0.86 |
| relativistic, original Adam (exp16) | 0/12 | 2 | 0.51 / 0.83 / 0.84 |
| relativistic + C (exp17) | 1/12 | 1 | 0.12 / 0.85 / 0.86 |

Relativistic alone reduces failures but not collapse events (2, both
recovered; one run ended at 0.51, just above the failure line). On top of C
it adds nothing and had one failure. Medians of the other measures are close
across C, relativistic and both (context vs generated AUROC 0.969–0.971,
context vs unseen real 0.951–0.959). **Decision: recipe C (Adam only) is the
default — fewest failures and collapse events, and the smallest change.**

**exp18/19 — final recipe (Adam β1 0 / β2 0.99), blind generator, seeds
50–61: the control for the method paper.** All 12 generators collapsed
(variety 0.000); D separates context from generated at AUROC 1.000 with the
right or the wrong context; digit classifier at chance (median 0.093).

**eval_report / report_summary_v1, v2 — probes of the final checkpoints for
the paper** (`scripts/eval_report.py`, `scripts/summarise_report.py`).
Context-aware, 12 runs: digit classifier 0.839 (0.819–0.849, 5 context draws)
vs pixel nearest-neighbour 0.783, pixel RBF density 0.624 (median bandwidth)
and 0.788 (best bandwidth, 0.15 × median); all 12 beat the best RBF. Context
vs unseen real AUROC 0.961; unseen digits present in the context score 0.227 vs
0.061 absent; 19% of unseen images score > 0.5. Order of events in all 12: the
discriminator relies on its state first (swap < 0.8 at median step 2750), the
generator follows (context use > 0.5 at median 4000). v2 only adds the
per-digit list that v1 lost to a key collision.
Paper: https://media.tanh.xyz/seewhy/paper/recall-gan_paper.html

**exp20 — the method at M = 16, trained 60k steps (3× the paper), seeds
50–55.** 0/6 failed, no collapse events. The generator's context use keeps
rising to ~45k and then levels (median 0.71 at 6k, 1.02 at 21k, 1.13 at 45k,
1.13 at 60k). The discriminator does not improve with it: digit classifier
0.855 at 6k → 0.851 at 21k → 0.729 at 60k; context vs unseen real peaks at
0.964 (30k) and ends at 0.929; context vs generated 0.934 at 60k against 0.971
for the same seeds at 20k (exp15). Swap test unchanged (0.549). So longer
training finishes the generator but erodes what the discriminator's state
generalises to. Cause not tested.

**exp21/22 — the method with contexts of 64 and 256 images, 20k steps,
seeds 50–55.** 0/6 failed in each. Medians at 20k (paper, M = 16, same seeds:
context vs unseen real 0.954, vs generated 0.971, generator context use 0.995):

| | M = 64 | M = 256 |
|---|---|---|
| context vs unseen real (AUROC) | 0.860 | 0.704 |
| context vs generated (AUROC) | 0.925 | 0.808 |
| swap test (wrong context) | 0.602 | 0.629 |
| digit classifier, 16 per digit | 0.857 | 0.853 |
| digit classifier, M per digit (`class_acc_m`) | 0.917 | 0.924 |
| generator context use | 0.625 | 0.290 |
| time per run | 435 s | 1612 s |

- Item-level membership (context vs unseen real) falls as the context grows
  past the fixed 16,384-number state; class-level use holds and improves with
  more examples per digit (0.92).
- The discriminator still relies on its state at every M (swap 0.60–0.63).
- M = 256 is still improving at 20k (context vs unseen real 0.684 at 15k,
  0.704 at 20k; generator context use 0.247 → 0.290), so its numbers are a
  lower bound. M = 64 has mostly levelled off (0.834 → 0.860).
- One M = 64 run (seed 50) has swap 0.934 and context-vs-unseen 0.636: its
  discriminator mostly ignores its state.

**exp23/24 — training on every prefix ("teacher forcing", agreed with the
user), with and without the nearest-image context, M = 16, 20k steps, seeds
50–55.** Design: after each write, 2 real queries (newest image, a random
earlier one) and 2 fakes (the "twin" from the previous prefix, a fresh one).
exp23: first 4 context images random, each later one the real training image
nearest (pixel L2) to the generator's image at the current prefix. exp24: all
random. `twin_auroc` = newest context image vs its twin, held-out.

| median [range], end of training | paper (exp15) | prefix only (exp24) | prefix + nearest (exp23) |
|---|---|---|---|
| digit classifier | 0.843 [0.826, 0.864] | 0.529 [0.401, 0.627] | 0.429 [0.170, 0.560] |
| context vs unseen real | 0.954 [0.936, 0.979] | 0.872 [0.854, 0.897] | 0.798 [0.697, 0.832] |
| context vs generated | 0.971 [0.958, 0.981] | 0.891 [0.780, 0.955] | 0.716 [0.608, 0.934] |
| swap test | 0.584 | 0.542 | 0.483 |
| fakes rejected (P < 0.5) | 0.824 | 0.948 | 0.116 |
| twin AUROC | — | 1.000 (twin unrelated) | 0.778 [0.742, 0.810] |
| generator context use | 0.995 | 1.409 | 3.539 |
| generator variety | 0.177 | 0.188 | 0.193 |
| time per run | 153 s | 355 s | 464 s |

- No generator collapsed in either arm. "Failed" by the old definition
  (classifier < 0.5): 2/6 and 4/6 — a definition made for the plain game.
- Caveat: the end-of-context measures use random 16-image contexts. For exp23
  that is not the context distribution it trained on; `twin_auroc` is its
  in-distribution measure.
- exp23's discriminator separates each newly written image from its pixel-
  nearest fake at AUROC 0.778 (0.80 at 2k, 0.69 at 10k, 0.78 at 20k): above
  chance, far from the paper's 0.97 on the easier context-vs-generated task.
  Its generator's output depends on the context 3.5× more than on its noise.
- Prefix training alone (exp24) already lowers every end-of-context measure;
  the nearest-image context lowers them further.

**exp25/26 — as exp23/24, with the two real queries at each prefix drawn at
random from the context so far (`prefix_real="random"`), seeds 50–55.** Report
3. Copying of the last context image's digit rises to 0.882 (prefix only) and
0.832 (with nearest image), against 0.513 / 0.907 with the newest image always
queried; plain training 0.137 (chance ~0.15). The twin pictures show the
generator reproducing the image just written almost pixel for pixel, so a fake
is a near-copy of a context image. Twin AUROC 0.436 (prefix only, falls below
0.5 after ~13k steps) and 0.602. Digit classifier with 16 per digit 0.418 /
0.310 (plain 0.843). One run per setup collapsed to one digit (prefix_rand s50,
prefix_nn_rand s53). The discriminator now scores the most recent context
images lowest (last 0.35 vs first 0.71, prefix only).

**exp27/28 — prefix training with the blind generator, newest-image (27) and
random (28) real queries, seeds 50–55.** Report 4. All 12 generators collapse
to one image within 500–2,000 steps (variety < 0.06), as under plain training
(exp18/19). The discriminator rejects every fake (1.000), ignores its state
(context vs unseen real 0.500) and the digit classifier sits at chance (0.104 /
0.087). No copying (the generator cannot read the context).

**copy_target — why the prefix-trained generator copies only the newest image
(scripts/analyse_copy_target.py, evaluation only, seeds 50–55).** Medians:

| | plain | prefix, newest real | prefix, random reals |
|---|---|---|---|
| generated image best matches position 16 (uniform 0.0625) | 0.070 | 0.469 | 0.911 |
| generator's memory read returns position 16 | 0.067 | 0.382 | 0.404 |
| cosine between queries of different noise samples | 0.002 | 0.001 | 0.001 |
| cosine between keys of different context images | 0.942 | 0.965 | 0.874 |

The copy is of the newest image itself (pixel correlation), not just its
digit. The generator's queries differ with the noise (cosine ~0), so the
"query ignores noise" explanation is out. The keys of different images in the
generator's KDA state are nearly parallel (cosine 0.87–0.97) in every arm,
including plain training: under the delta rule each write then mostly
overwrites the previous one along that shared direction, and the read returns
the newest image ~6× more often than chance in the prefix arms. Plain training
has the same key geometry but no copying, so this explains why the target is
the newest image, not why prefix training brings copying on.

**exp29/30 — discriminator pretrained 5k steps on context vs real non-members
(no gradient penalty), then the method's GAN, context-aware (29) and blind (30)
generators, seeds 50–55.** Report 5. After pretraining: context vs unseen real
0.995, digit classifier 0.710. Within 500 GAN steps both fall to chance-level
(~0.5 / 0.08–0.19) in every run, before any collapse. Context-aware: 4/6
collapse (steps ~3k–6k); the 2 survivors end at classifier 0.815 / 0.829
(scratch median 0.843). Blind: 6/6 collapse, as from scratch. Untested causes:
the gradient penalty switching on, or the easy early fakes overwriting the
membership features.
