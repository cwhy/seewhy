# Recall and completion on a synthetic prior, tested on chess

Three networks were trained on a synthetic prior and never shown a real chess
position. They are identical in size and shape and differ in one thing: what their
training episodes looked like. One always had its answer sitting in the context.
One never did. One was the first with most of its weights frozen at random.

Then all three were asked to do both jobs on real chess: find an item that is
present, and predict one that is absent.

| | finding, on chess | predicting, on chess |
|---|---|---|
| recall-trained | **0.945** | 1.861 |
| completion-trained | 0.077 | **2.380** |
| frozen layers | 0.057 | 2.057 |
| chance / do-nothing | 0.063 | 1.000 |

## The terms used here

**An episode.** The network is shown sixteen complete items, one per token, then
a seventeenth with part of it erased. It has to produce the erased part. For a
chess item the erased part is the queenside, files a to d.

**The context** is the sixteen items. **The query** is the seventeenth. Two kinds
of episode matter throughout, and they are the two columns of every figure below.
In one the query is a copy of one of the sixteen, so the answer is *present* and
the network can succeed by finding it. In the other the query is not among them,
so the answer is *absent* and the missing part has to be worked out.

**The three networks** are identical in size and shape — 14.95M numbers, four
layers, d_model 512 — and differ only in the episodes they were trained on.

- **Recall-trained**: its answer was always one of the sixteen. Copying always
  worked, so it never had to learn to predict anything.
- **Completion-trained**: its answer was never among the sixteen. Copying was
  never available, so it could only ever predict.
- **Frozen layers**: trained like the recall network, but its four mixing layers
  keep their random starting values forever. Only the input embedding and the
  output head learn — a small fraction of the total.

**The synthetic prior** is what all three were trained on instead of real data.
Each episode samples a fresh **world** — a random low-dimensional generative
model — and draws its sixteen items and its query from that one world. The
network therefore cannot memorise any particular world; it has to work out what
this world is from the sixteen items in front of it. That is the whole idea, and
it is what TabPFN does with tables.

A world is drawn in one of two modes. A **continuous** world produces items whose
coordinates take any value in [0, 1]. A **simplex** world puts exactly one active
coordinate in each group of thirteen, so its items are strictly binary. Forty per
cent of worlds are simplex. That number matters later, because a chess board in
this project *is* sixty-four groups of thirteen with one active coordinate each.

**Three bands** appear in every figure, increasing in novelty downward: items from
worlds seen during training, items from worlds never seen, and real chess — which
the network has never been shown in any form.

The two synthetic bands are scored on **single-world episodes**, matching how
these networks were trained: all sixteen context items come from one world, which
is what makes "infer this world" a question at all. The real chess band is scored
on unrelated items, because a real dataset has no world structure to respect.
Scoring the synthetic bands the second way asks a question the networks were
never trained for and reads 1.35 where their own task reads 0.41; an earlier
version of these reports did exactly that.

**Normalised error** is squared error over the erased coordinates, divided by the
error of a fixed reference so that 1.0 means "no better than that reference".
Which reference is stated on each figure. For chess the reference used here is the
average real chess item, whose raw error is 0.0344.

**Identification accuracy** asks which of the sixteen context items the network's
output most resembles, measured on the erased coordinates only — so a network
that merely copies the visible part cannot score. It is 1/16 = 0.063 at chance.
Its ceiling is 1.000 unless two context items share an erased half, in which case
even a perfect answer can lose the tie-break; the ceiling is reported where it
is not 1.000.

**Two references involve no trained network at all.** *Ridge* is a linear map from
the visible coordinates to the erased ones, fitted on the training pool and
applied blind — it never looks at the context. *Soft look-up* is a
similarity-weighted blend of the sixteen context items, which is the shape of
computation linear attention can actually perform, and is therefore the bar for
whether the context is being used at all.

## Finding, and predicting

![The three arms on chess](https://media.tanh.xyz/seewhy/26-09-03/recall-gen_r16_bars_v5.svg)

The left panel puts real chess beside the control that decides how to read
everything else: fresh worlds drawn from the prior the networks were actually
trained on.

The recall-trained network scores 0.634 there, against
chance of 0.063. It did learn to retrieve from a world it had never seen, given
sixteen examples of it. That is the thing the prior was built to teach, and at
this width it works.

At half this width it did not. The same arm, same prior, same budget, at
d_model=256 scored 0.142 on that control. The section below is
about that, and it is the reason the arms here are the wider ones.

On real chess the recall arm scores 0.945. The
completion-trained network scores 0.077 and the frozen
one 0.057 — both at chance. Across three independent
draws of 512 episodes the spread is about 0.015, drawn as
the error bar on each bar.

The right panel is prediction, with the two references that bound it. Ridge never
looks at the context and scores 2.093. The soft look-up uses
nothing but the context and scores 1.033. The best of the three
networks is **recall-trained** at 1.861.

## The recall/completion split

Across reports 12 to 15 this comparison had a consistent shape with two parts.
One part reproduces here and the other does not.

The part that reproduces: **a completion-trained network does not read its
context.** On chess it scores 2.380 when the answer is
present and 2.380 when it is absent — the same number, to three
decimals. Whether the answer is sitting in front of it makes no difference to
what it draws, because its training never rewarded looking. Its identification
is 0.077 against chance of 0.063. That signature has
now appeared in every domain this project has run, real or synthetic, and four
times the capacity does not change it.

The part that does not: **completion training buys no prediction here.** On real
data that was the trade — the completion arm gave up finding and got predicting
in return. On this prior it gives up finding and gets nothing: it predicts at
2.380 against the recall arm's 1.861, and the recall
arm is the one that can also find things.

So on a synthetic prior the recall objective dominates. It is better at finding
by 0.869 and no worse at
predicting. There is no trade to make.

The frozen arm is at chance too, at 0.057. Freezing the
mixing layers has been survivable on real data; on a prior that has to be
inferred from the context every episode, it is not.

![What the three networks produce](https://media.tanh.xyz/seewhy/26-09-03/recall-gen_r16_grid_v5.png)

The bottom band is real chess, and it is the one to look at. The prior's own
items, two bands above, are what the networks were fed.

## Is it trained enough?

The honest answer to why training stopped where it did is that the budget was
set to 48 000 steps, not that anything showed it was enough. It was not enough,
and the reason is worth stating because it changes how every number above should
be read.

![Identification against training compute](https://media.tanh.xyz/seewhy/26-09-03/recall-gen_r16_scaling_v4.svg)



Four times the steps takes identification on the prior's own training worlds from 0.614 to **0.650**, and on worlds it has never seen from 0.542 to **0.592**.

Twice the width, at the original budget, reaches **0.759** and **0.644** — roughly four times the parameters and twice the recurrent state, 32 768 floats against 16 384.

Of the two, **more width** buys more on worlds the network has never seen, which is the number that matters: 0.592 for the longer run against 0.644 for the wider one.

Three things say the same thing. The training loss is still falling at the end.
The gap between worlds seen in training and worlds never seen is small —
0.614 against 0.542 for the narrow 48 000-step run — so the network is underfitting rather than memorising, and a
network that is underfitting has room that more compute can buy. And the budget
has already been quadrupled once, from 12 000 steps to 48 000, which moved
identification on training worlds from 0.488 to 0.614.

The flattening at the end of any single run is a property of the optimiser, not
of the task. The learning rate follows a cosine decay to a tenth of its peak over
whatever `steps` is set to, so every run in this project goes flat over its last
decile whatever it has converged to. The 12 000-step run went flat at 0.49 and
the 48 000-step run went flat at 0.62. Same shape, different values. Reading
either as saturation would be reading the schedule.

## Does the prior's simplex mode carry anything?

This is a question about the prior rather than about the three networks, and it
is worth asking because one dataset behaves very differently from the other two.

Chess is sixty-four groups of thirteen with one active coordinate each, which is
exactly what a simplex world draws. A recall network trained on the prior *with*
simplex worlds retrieves chess at 0.942; trained on an otherwise identical prior
*without* them, at 0.321. That is not generalisation to chess. It is the prior
generating chess.

![Resemblance against transfer](https://media.tanh.xyz/seewhy/26-09-03/recall-gen_r16_ablation_v4.svg)

The same axis orders chess. It has **1.000** of its coordinates
within 0.1 of 0 or 1, against 0.914 for MNIST and 1.000 for chess.
Removing the simplex mode helps retrieval on chess by
0.621, against a draw-to-draw spread of
0.015.

Ordering the three datasets by how binary they are orders them by how much the
simplex mode is worth: chess +0.62, MNIST +0.12,
Fashion-MNIST -0.03. Three points is an ordering and not a
fit, and it is measured on one axis only.

## What this establishes

That a network trained only on this synthetic prior learns to retrieve from a
world it has never seen — 0.634, against chance of
0.063 — and carries part of that to real chess, at
0.945. It needed 15M parameters to do it; at 4M it did
not.

That the recall objective dominates on this prior. It is the only one of the
three that retrieves at all, and it predicts as well as either of the others.
The trade the completion objective makes on real data is not available here.

And that prediction is where the prior gives nothing. No arm, at either width,
at either setting of the simplex mode, predicts a real chess item better than
drawing the average real chess item.

## What this does not establish

Where the ceiling is. Nothing here is converged in the sense that more compute
would not help: the largest model tried is the best model on every measure, and
the step sweep was still climbing. The scaling section gives the two points that
bound what has actually been tested, and both directions were still paying.

Whether prediction would ever come. It did not move with four times the steps or
four times the parameters, which is evidence that it is not simply a compute
problem — but a prior with fewer latent dimensions, or more than sixteen items
per episode, would be a different experiment. Sixteen examples of an
800-dimensional world is very little to infer a world from, and that is a
property of the task design rather than of the network.

## Sources

`results.jsonl` rows `exp49` (recall), `exp50` (completion) and
`exp51` (frozen layers), all trained on the same prior with the same sampler,
seed and step budget; `exp43` and `exp44` for the simplex ablation, both
recall-trained. Their `_stdeval_*` rows come from `scripts/standard_eval.py` and
the `baselines_synth*_to_chess` rows carry the ridge, soft look-up and average-item
references. The eval-draw spread is measured over seeds 20260825, 4242 and
909090. Reports 17 and 18 share one generator, `scripts/gen_report_17.py`,
selected by `--target`; figures come from `lib/synthfig.py`. No training was run
for this report.
