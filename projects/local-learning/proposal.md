# Proposal — replicating a "quick MNIST test"

## Premise

On 2026-09-14 @mayfer posted a four-column table comparing backprop, PC-ALM and
Forward-Forward on MNIST at ~56K parameters, captioned "quick MNIST test",
quote-tweeting Sakana AI's PC-ALM announcement. Forward-Forward comes out ahead
on accuracy; PC-ALM is forty times slower to train than backprop and slightly
worse; Forward-Forward's inference costs three times backprop's.

The parameter counts pin down the architectures, and this is the crux:

* **56,128** is exactly a residual MLP of width 32 and depth 32 — the headline
  cell of the PC-ALM paper (arXiv:2605.31022).
* **56,232** is exactly a plain 784-66-66 MLP with biases and no output layer —
  the shape Forward-Forward needs, since goodness replaces the classifier head.

So the three rows are matched on parameter count and on nothing else. Two of
them are 32 layers deep and 32 units wide; the third is two layers of 66. A
comparison of *learning rules* that varies architecture between rows cannot
separate the rule from the shape.

## Claims under test

| # | Claim | Where it comes from | Test |
|---|---|---|---|
| C1 | The three accuracies are ~87.4 / ~86.3 / ~88.8 in one epoch | the tweet | exp1, exp2 with five seeds |
| C2 | PC-ALM costs ~40x backprop's training time, at near-equal accuracy | the tweet | exp1 timings |
| C3 | Forward-Forward's inference costs ~3x backprop's | the tweet | exp2 timings |
| C4 | Forward-Forward's win is architectural, not a property of the learning rule | ours | exp2: backprop on 784-66-66, Forward-Forward on the deep-narrow stack |
| C5 | The ordering is an artifact of a one-epoch budget; at equal wall clock it changes | ours | exp4 |
| C6 | PC-ALM beats PC in the deep-narrow regime and matches it when shallow | PC-ALM paper | exp5 |
| C7 | No method is being held back by an unfair learning rate | ours | exp3 |

## Staged experiments

1. **exp1** — residual MLP L=32 N=32, one epoch, {bp, pc, pcalm} x 5 seeds.
   Expect backprop and PC-ALM within ~1.5 points of each other, PC clearly
   behind, PC-ALM's training time 30-60x backprop's.
2. **exp2** — Forward-Forward on 784-66-66 x 5 seeds; backprop on the same stack;
   Forward-Forward on a 31x32 deep stack. Expect FF ~88-89% on its own shape,
   backprop clearly above it on that same shape, and FF to degrade when deep.
3. **exp3** — learning-rate grid, 5 rates x 5 methods x 2 seeds. A fairness
   check, not a headline.
4. **exp4** — ten epochs per method, accuracy against cumulative training
   seconds. Expect backprop to reach PC-ALM's one-epoch accuracy in well under
   a second and to be far ahead at PC-ALM's 13-second mark.
5. **exp5** — depth sweep L in {2,4,8,16,32,64}. Expect the PC / PC-ALM gap to
   open with depth and close at L=2, and Forward-Forward to fall off with depth.

## Deviations from the sources

* We implement PC / PC-ALM from the paper's equations in JAX, then check the
  three constants the text leaves implicit against the authors' released code:
  the batch rescale of the activity step, the `pre_dual_energy` timing of the
  weight update, and the fixed layer scales. All three are documented in
  `lib/methods.py`.
* We measure `eta_h = 1/lambda_max` by power iteration instead of reading the
  authors' per-cell table, so the depth sweep needs no lookup.
* Forward-Forward normalises each layer's input to unit mean square rather than
  unit L2 norm. Same direction, different constant; the constant matters only
  at initialisation, and unit mean square is the one that leaves goodness near
  the threshold at the start of a one-epoch run.
* Negative samples are a uniformly random wrong label, not the hard negatives
  Hinton mines from the network's own predictions.
* All wall-clock numbers are ours, on one RTX 4090, and exclude compilation.
  They are not comparable in absolute terms to the tweet's; the ratios are.
