# A new dataset costs retrieval less than a new class

A network was trained on Fashion-MNIST and nothing else. Then it was shown
sixteen MNIST digits and asked to reproduce one of them from a half-erased copy.

It picked the right digit **97.3%** of the time.
Chance is 6.3%. It has never seen a digit.

Report 13 ran the easier-sounding test on the same architecture: train on
Fashion-MNIST classes 0 to 4, retrieve from classes 5 to 9. Same dataset, same
preprocessing, five held-out classes instead of a whole new world. That scored
**81.9%**.

So a network handles an entirely different dataset better than it handles five
held-out classes of its own. Whatever governs retrieval here, it is not novelty.

## The control

The obvious objection is that Fashion-MNIST is simply the richer thing to have
trained on, and a network that learned it can handle anything.

Run the crossing the other way and that dies. Train on MNIST, retrieve from
Fashion-MNIST: **64.7%**.

| trained on | retrieving | seen it? | identification |
|---|---|---|---|
| Fashion-MNIST | MNIST | no | **0.973** |
| Fashion 0-4 | Fashion 5-9 | no | 0.819 |
| MNIST | Fashion-MNIST | no | **0.647** |
| Fashion-MNIST | Fashion-MNIST | yes | 1.000 |
| MNIST | MNIST | yes | 1.000 |

Read the middle column, not the left one. Retrieving an MNIST digit scores high
whether or not the network trained on MNIST. Retrieving a Fashion image scores
low whether or not the network trained on Fashion.

The difficulty belongs to the pool being searched, not to the network's history
with it.

## Why

Identification asks which of the sixteen context images the output most
resembles, measured on hidden pixels only. Two separate things decide whether it
succeeds: how well the network reconstructs, and how far the target sits from
its nearest rival in that context. Only the first is about the network.

The second can be measured with no network at all.

![How far each target sits from its nearest rival](https://media.tanh.xyz/seewhy/26-08-27/recall-gen_r15_margin_v2.svg)

The pools differ, and in the order the identification numbers need. A target in
the MNIST pool sits a median **0.70** from its
closest rival. In Fashion classes 5 to 9 that is
**0.54**, and across all Fashion classes
**0.47** — in the same units, on the same
axis as the model errors marked on the plot. A t-shirt, a pullover and a coat
have nearly the same lower half. Nothing else looks like a 4.

What makes this more than a correlation is that the two runs being compared
reconstruct about equally well. The network is slightly **worse** at rebuilding
an MNIST digit than report 13's network was at rebuilding a held-out garment —
0.215 against
0.251 — and identifies it far better anyway.
Reconstruction quality is held roughly fixed; the candidate geometry is what
moves, and identification moves with it.

Two honest limits on that. The reverse control has both a smaller margin **and**
a much worse reconstruction (0.726), so it is
consistent with the story but does not independently test it. And the margin
does not predict the error rate quantitatively — flipping needs the error to
point at the rival, not merely to be large enough — so this ranks pools rather
than forecasting accuracies.

An earlier version of this figure swept isotropic Gaussian noise on a perfect
reconstruction and asked where identification broke. It found nothing: all four
pools sat at 1.000 out to a noise scale of 1.1 per pixel. Independent noise over
392 pixels adds nearly the same offset to every candidate's distance, so it
barely disturbs the ranking. The measurement above replaced it.

## The task and the three networks

Each episode is sixteen complete 28x28 images, one per token, then a seventeenth
with its bottom fourteen rows erased. The network produces the missing 392
pixels.

The novel pool is a dataset rather than a class. The six conditions absorb that
without changing. The novel dataset's ten labels are shifted up by ten and the
two test splits concatenated, so `held_same` comes out as Fashion-MNIST's own
test split and `held` as MNIST's.

    A/C   Fashion-MNIST train split    images seen in training
    E/F   Fashion-MNIST test split     new images, same world
    B/D   MNIST test split             a different world

E/F is the control that separates image novelty from distribution shift. It
costs nothing: 0.030 against
0.031, identification
1.000 against
1.000.

Three networks, identical in size and shape, differ only in training episodes.
**Recall-trained** always had its answer in the context. **Completion-trained**
never did. **Frozen** is the recall network with its four mixing layers held at
their random initialisation.

![Completions across three levels of novelty](https://media.tanh.xyz/seewhy/26-08-27/recall-gen_r15_f2m_grid_v2.png)

## Prediction crosses badly, as usual

With the answer absent, the recall-trained network scores
**0.877** on MNIST, against
0.458 on new Fashion images.

The reference that matters is the linear map: fit visible pixels to hidden
pixels on Fashion, ignore the context, apply to MNIST. It scores
0.870. The network scores
0.877. Sixteen images to look at are worth
essentially nothing.

Two cautions about that number, because the normaliser is doing work here.

1.0 means "no better than drawing the average Fashion image". On MNIST queries
that is a weak constant, so 1.0 is easier to beat than it looks. Predicting pure
black, which is a real strategy on MNIST, scores
1.092 — worse than the Fashion mean, because the
bottom half of a digit does carry ink.

And the context did contain usable information the network ignored. The soft
look-up ceiling — the best a similarity-weighted blend of the sixteen can do,
which is the shape of computation linear attention performs — is
0.639 on MNIST, well below the network's
0.877.

![The same six blocks as numbers](https://media.tanh.xyz/seewhy/26-08-27/recall-gen_r15_f2m_bars_v2.svg)

## The other two arms

The completion-trained network scores 0.076 with
the answer present and 0.076 with it absent. Same
number: it does not read its context. That signature has now appeared in all
four domains this project has run.

On MNIST it scores **1.212**, worse than the
do-nothing line, and its identification drops to
0.169.

The frozen network gives up retrieval — 0.459 on
MNIST against the recall network's 0.973 — and
buys prediction with it: **0.684** against
0.877, which does beat the linear map's
0.870.

That is the clearest present/absent trade this project has measured. The frozen
network is worse at finding and better at guessing, on the same images, in the
same run.

## A nearest-neighbour context

Replacing the sixteen unrelated images with the query's own sixteen nearest
neighbours makes the context informative. No network was trained that way.

![The same three networks on nearest-neighbour contexts](https://media.tanh.xyz/seewhy/26-08-27/recall-gen_r15_f2m_knn_v2.png)

The soft look-up ceiling on MNIST moves from
0.639 to
0.358. Compare the red numbers here
against the first figure to see how much each network collects.

## Which context matters: the one it trained on, or the one it is given

The figure above holds the network fixed and improves its context. That leaves
the other half unasked: what if the network had been *trained* on informative
contexts to begin with?

exp40 answers it. It is exp36 with one change — during training its sixteen
context images were the query's own nearest neighbours rather than sixteen
unrelated pictures. Same data, same objective, same Q.

![Both networks on unrelated contexts](https://media.tanh.xyz/seewhy/26-08-27/recall-gen_r15_ctx_iid_v1.png)

![Both networks on nearest-neighbour contexts](https://media.tanh.xyz/seewhy/26-08-27/recall-gen_r15_ctx_knn_v1.png)

Completion on MNIST, with the answer absent:

| | tested on unrelated | tested on neighbours |
|---|---|---|
| **trained on unrelated** | 0.878 | 0.535 |
| **trained on neighbours** | 0.860 | 0.513 |

Read across a row and the score moves by about
0.34.
Read down a column and it moves by about
0.02.

Completion is governed by the context the network is handed, not by the context
it grew up on. Training on informative contexts for twelve thousand steps buys
almost nothing that being handed one at test time does not already give.

Identification behaves in the opposite way.

| | tested on unrelated | tested on neighbours |
|---|---|---|
| **trained on unrelated** | 0.970 | 0.763 |
| **trained on neighbours** | 0.993 | 0.914 |

Here training does the work. The knn-trained network is the better retriever in
both columns, including the unrelated contexts it never trained on, where it
reaches 0.993 against
0.970. Being made to tell
near-identical images apart during training produces a sharper matcher, and that
sharpness transfers to the easy case.

It also reconstructs better with the answer present:
0.130 against
0.221.

## The same confound, found a second way

Look down the columns of the identification table rather than across. Both
networks identify worse on neighbour contexts than on unrelated ones —
0.970 to
0.763 for one,
0.993 to
0.914 for the other — even
though a neighbour context makes every other number better.

That is the nearest-rival effect again. A knn context is assembled *from* the
query's closest matches, so it is by construction a low-margin context. Measured
the same way as the figure above, the median distance from a target to its
nearest rival falls from **0.68** on unrelated contexts to
**0.39** on neighbour contexts.

This matters because it is an independent test. The earlier figure compared
different *pools* and found identification tracking the margin. This compares
different *contexts drawn from the same pool*, with the network held fixed, and
finds the same thing. Two unrelated manipulations, one mechanism.

It also means a neighbour context is not simply better. It makes the answer
easier to construct and harder to name.

## What this changes

Report 13 concluded that retrieval is not class-agnostic, because it degraded
from 1.000 on MNIST digits to
0.819 on held-out Fashion classes. That
conclusion was right about the number and wrong about the cause.

The degradation was not the network failing to generalise. It was Fashion-MNIST
being a harder pool to tell apart.

Two facts make that the better reading. Retrieving Fashion images stays hard for
a network that trained on MNIST — 0.647, worse
than report 13's 0.819, not better. And
retrieving MNIST digits is easy for a network that has never seen one:
0.973. In both crossings the score follows the
pool being searched, not what the network was trained on.

Report 12's original claim, that retrieval is content-addressed and largely
indifferent to what it has seen, survives this better than report 13's revision
of it. What has to be added is that identification accuracy is not a pure
measure of the network. It is confounded by how confusable the candidates are.
That confound is worth more than the entire class-novelty effect report 13
attributed to the network.

Two things follow. Any identification number in this project should be quoted
with its pool's nearest-rival distribution, or at least compared only within a
pool. `lib/splitfig.nearest_distractor` computes it and costs no training.

And the class-split design is a weaker instrument than it appears. Holding out
classes changes both what the network has seen and what it is searching among,
and the second effect is the larger one. Crossing datasets in both directions,
as here, separates them.

## What this does not establish

Two datasets, one direction each way. Both are 28x28 grey and centred, which is
what makes a single network scorable on both; a wider shift would need a
different instrument.

The tolerance curve uses isotropic noise, which is not what a network's error
looks like. It ranks the pools correctly but should not be read as predicting
any particular accuracy.

The frozen network's present/absent trade is one run at one size. It is the most
interesting thing here that has not been replicated.

## Sources

`results.jsonl` rows `exp36`, `exp37` and `exp38` — recall, completion
and frozen-layer training on Fashion-MNIST with MNIST as the novel pool.
`exp39` is the reverse-direction control and `exp30` the
within-Fashion class split from report 13. `baselines_fashion_to_mnist_M16_r14_split`, `baselines_fashion_to_mnist_M16_r14_split_knn_Q1` and `baselines_mnist_to_fashion_M16_r14_split` hold
the linear, look-up, predict-black and average-image references. Nearest-rival
distances and identification ceilings are computed in this script by
`lib/splitfig` and involve no trained network. Figures generated by `scripts/gen_report_15.py` from `lib/splitfig.py`.
No training was run for this report.
