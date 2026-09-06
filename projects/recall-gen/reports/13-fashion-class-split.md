# On Fashion-MNIST retrieval breaks and prediction survives

Report 12 ran a class split on MNIST. Train on digits 0 to 4, test on 5 to 9.
Retrieval crossed that boundary untouched — identification accuracy **1.000** on
digits never seen, against chance of 0.063. Prediction collapsed to **0.999**,
where 1.0 is the score for drawing the average digit and ignoring the input
entirely.

The same experiment on Fashion-MNIST comes out the other way round.

Retrieval no longer crosses intact. Identification accuracy on unseen classes is
**0.819**, against **0.998**
on unseen images of *seen* classes and a ceiling of 1.000.
Chance is 0.063.

Prediction does not collapse. On unseen classes with the answer absent the same
network scores **0.668**, where MNIST scored 0.999. It
beats a linear map that ignores the context entirely, which scores
0.785. On MNIST that
comparison ran the other way: 0.999 against the linear map's 0.851.

So the capability that survives a class change is not a property of the
mechanism. It is a property of the dataset.

## What was run

Three networks, identical in size and shape. They differ only in the episodes
they were trained on. All three saw Fashion-MNIST classes 0 to 4 and nothing
else — t-shirt, trouser, pullover, dress, coat.

The **recall-trained** network always had its answer sitting in the context
during training. Copying was always valid for it.

The **completion-trained** network never had its answer in the context. Copying
was never available, so it could only predict.

The **frozen** network was trained like the recall one, with its four mixing
layers held at their random initialisation. Only the embedding and output head
learn — 0.60M of its 4.03M numbers.

Each episode shows sixteen complete 28x28 images, one per token, then a
seventeenth with its bottom fourteen rows erased. The network produces the
missing 392 pixels. Error is squared error over those pixels, divided by the
error of always drawing the average training image, so 1.0 is the do-nothing
score.

The class split is wider here than on MNIST. Four of the five held-out classes —
sandal, sneaker, bag, ankle boot — are not garments at all. Only shirt is a
near neighbour of anything in training.

![Completions across three levels of novelty](https://media.tanh.xyz/seewhy/26-08-25/recall-gen_r13_fashion_grid_v5.png)

The red number beside each method is that network's score over all 512 episodes
of that block. The numbers under the tiles are the five episodes shown. The
tiles are examples; the red number is the result.

## Retrieval is the thing that breaks

Read down the left column of the figure, where the answer is present.

The top two rows are the same. Seen images score
**0.021**, unseen images of seen classes score
**0.027**. Identification is
1.000 and 0.998
against a ceiling of 1.000. A new picture of a coat costs
nothing.

The bottom row is not the same. Unseen classes score
**0.251** — an order of magnitude worse than the row
above it — and identification falls to
0.819.

That is the finding. The image is *sitting in the context*. A pure look-up
scores 0.000 on these same
episodes, because the answer is exactly there to be found. The network misses it
roughly one time in six.

On MNIST the same measurement was 1.000, matched to the ceiling. Report 12
concluded that retrieval matches pixels and needs to know nothing about what it
is looking at. Fashion-MNIST says that conclusion was too strong. Matching a
sandal against fifteen other sandals is harder than matching a 9 against fifteen
other digits, and this mechanism is not class-agnostic enough to do it.

![The same six blocks as numbers](https://media.tanh.xyz/seewhy/26-08-25/recall-gen_r13_fashion_bars_v5.svg)

## Prediction is the thing that holds up

Now the right column, where the answer is absent.

The recall-trained network scores **0.425** on seen
images, **0.448** on new images of seen classes, and
**0.668** on unseen classes. Every one of those is
below 1.0, the do-nothing score.

MNIST's numbers in the same three places were 0.696, 0.684 and 0.999. The first
two match; the third does not. On MNIST the class boundary erased prediction
entirely. Here it costs about half the gap to the do-nothing score, and no more.

The reference that settles it is the linear map. Fit visible pixels to hidden
pixels on classes 0 to 4, ignore the context, apply to unseen classes: it scores
0.785. The network scores
0.668. Having sixteen images to look at is worth
something here. On MNIST it was worth less than nothing.

The reason is visible in the tiles. Fashion-MNIST images are centred silhouettes
against black. A sneaker's bottom half is not a coat's bottom half, but both are
a filled shape with a flat base, and the prior for "filled shape with flat base"
was learned from coats. A 9's bottom half shares no such structure with a 3's.

## The completion arm behaves identically to MNIST

The completion-trained network scores **0.031** with
the answer present and **0.031** with it absent. Those
are the same number.

That signature appeared on MNIST too, at 0.015 and 0.015. It means the network
does not read its context. It was never rewarded for reading it, so it does not,
and whether the answer is there makes no difference to what it draws.

It is also the worst of the three at retrieval on unseen classes:
identification 0.250, against the recall
network's 0.819 and chance of 0.063.

And it is the worst at prediction on unseen classes as well:
**0.916**, above the do-nothing score of 1.0 and
well above the linear map's
0.785. Training only to
predict does not produce a prior that crosses to shoes.

## Freezing does not help here

On MNIST the frozen network was the better generaliser: 0.888 on unseen classes
with the answer absent, against the recall network's 0.999.

On Fashion-MNIST there is nothing for it to recover. The recall network already
scores 0.668; the frozen one scores
**0.671**. They are the same.

Freezing costs retrieval instead. Identification on unseen classes is
0.342, against
0.819 trained. On seen classes it is
0.938 against
0.998. A random mixer can retrieve a coat it has
been tuned around, and cannot retrieve a sandal.

## A better context rescues prediction, not retrieval

Everything above uses sixteen unrelated images. Such a context says little about
an absent answer. Replace it: give each query its own sixteen nearest
neighbours, ranked by how similar their visible halves are. For an unseen
sneaker, those neighbours are other unseen sneakers.

No network was trained this way. Same weights, new kind of context.

![The same three networks on nearest-neighbour contexts](https://media.tanh.xyz/seewhy/26-08-25/recall-gen_r13_fashion_knn_v5.png)

The soft look-up ceiling on unseen classes with the answer absent moves from
0.621 to
0.310. The context really
did become informative.

Read the red numbers in the figure against the ones in the first figure to see
how much of that each network collects.

What does not move is the class gap in retrieval. A helpful context supplies
missing knowledge about what a sneaker looks like. It does not make the
sixteen-way match easier — if anything it makes it harder, because sixteen
near-identical sneakers are harder to tell apart than sixteen unrelated images.

## What this changes

Report 12's headline was "perfect retrieval and chance-level prediction, on the
same images, in the same run". That sentence is true of MNIST and false of
Fashion-MNIST, where the two halves both move and both move the other way.

Three things follow for what is already published.

The claim that retrieval is content-addressed and therefore class-agnostic needs
the qualifier that MNIST classes are unusually easy to tell apart at the pixel
level. `concepts.md` finding 1 should carry that.

The claim that a recall-trained model's completion is worse than ignoring the
context is MNIST-specific. Here it is better than ignoring the context, by
0.668 against
0.785.

The frozen-layer result does not replicate. It was the best generaliser on MNIST
and is indistinguishable from the recall network here.

For the next run: the two datasets differ in two ways at once, class similarity
and low-level structure. A rotation over which five Fashion-MNIST classes are
held out would separate them. Holding out only shirt — the one held-out class
that *is* a garment — should restore MNIST-like retrieval if class similarity is
what drives it.

Report 14 asks the same question on chess positions, where the completion task
has real structure and the answer comes out different again.

## Sources

`results.jsonl` rows `exp30`, `exp31` and `exp32` — recall,
completion and frozen-layer training on Fashion-MNIST classes 0-4. `baselines_fashion_mnist_M16_r14_split` and
`baselines_fashion_mnist_M16_r14_split_knn_Q1` hold the linear, look-up and average-image references. Every MNIST
number quoted in comparison comes from `exp8_sharedq`, `exp9_sharedq` and
`exp29`. Identification ceilings are computed in this script by scoring the
true target against its own context. Figures generated by
`scripts/gen_report_13.py` from `lib/splitfig.py`. No training was run for this
report.
