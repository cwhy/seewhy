# Retrieval crosses to new digits. Prediction does not.

A network was trained on MNIST digits 0 to 4 only. It never saw a 5, 6, 7, 8 or
9 during training. Not as an example, not as a question, not as an answer.

Then it was tested on those five unseen digits.

It finds them perfectly. Asked to reproduce an image sitting in its context, it
picks the right one every time, even when that image is a 9. Its error is
**0.043**, where 0.0 is perfect and 1.0 means no better than drawing the average
digit.

It cannot predict them at all. Asked to fill in the missing half of an unseen
digit when the answer is *not* in its context, it scores **0.999**. That is the
average-digit score. It has learned nothing about what a 9 looks like.

Both numbers come from the same network, on the same unseen images, in the same
evaluation run. Only the question differs.

Two networks appear throughout this report. They are identical in size and shape.
They differ only in the episodes they were trained on.

The **recall-trained** network always had its answer sitting in the context during
training. Copying was always a valid strategy for it, and it is the network the
two numbers above describe.

The **completion-trained** network never had its answer in the context during
training. Copying was never available to it. It could only ever predict.

Both saw digits 0 to 4 and nothing else.

![Completions across three levels of novelty](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r12_digit_split_grid.png)

## The task

Each example is a small episode. The network is shown sixteen complete MNIST
images, one per token. Then it is shown a seventeenth image with its bottom half
erased. It has to produce the missing 392 pixels.

Error is mean squared error over those hidden pixels. It is divided by the error
of always drawing the average training image. So 1.0 is the do-nothing score.
Below 1.0 is better than nothing. Above 1.0 is worse.

Two kinds of episode matter, and they are the two columns of the figure above.

In the first kind, the answer is already there. The seventeenth image is a copy
of one of the sixteen. The network can succeed by finding it and copying it.

In the second kind, the answer is absent. None of the sixteen is the query. The
missing half has to be worked out from what the visible half suggests.

## Why split the digits

Held-out images are a weak test. A held-out 3 is a new image, but the network has
seen thousands of 3s. Whatever it knows about the shape of a 3 still applies.

Splitting by digit class removes that. Train on 0 to 4, test on 5 to 9, and the
test images are not merely new. Their whole category is new.

The figure has three rows for this reason. The top row is images seen in
training. The middle row is new images of digits 0 to 4 — new pictures, familiar
shapes. The bottom row is digits 5 to 9, which are new in every sense.

Reading down a column shows how much each kind of novelty costs.

## What the pictures show

Look at the left column of the figure, where the answer is present. All three
rows look the same. The recall-trained network reproduces the query almost
exactly, whether it is a familiar 3 or an unseen 9.

This is not a small effect. Identification accuracy — does the output most
resemble the correct one of the sixteen context images — is **1.000** on unseen
digits. Chance is 1/16, or 0.063.

Now look at the right column, where the answer is absent. The top two rows are
recognisable attempts. The bottom row is not. On unseen digits the network
produces a shape that belongs to no particular digit.

The middle row is the control that makes this readable. Those are new images too.
If new images alone broke the network, that row would fail as well. It does not:
**0.684** against the top row's 0.696. Novel images cost essentially nothing.

Novel classes cost everything: **0.999**.

One more thing is visible in the figure, in the completion-trained row. Its two
columns match, panel for panel, to within rounding. Same picture, same number,
whether the answer is present or absent.

That network ignores its context. It was never rewarded for reading it, so it
does not. Whatever it draws, it draws from the visible half alone.

![The same six blocks as numbers](https://media.tanh.xyz/seewhy/26-08-19/recall-gen_r12_digit_split_bars.svg)

## Worse than ignoring the context

A linear model was fitted to predict hidden pixels from visible ones. It sees no
context at all. It is just a fixed map, trained on digits 0 to 4.

On unseen digits with the answer absent it scores **0.851**. The trained network
scores 0.999. The network with sixteen images to look at does worse than a linear
map with none.

So the network has not merely failed to learn a general prior. It has learned a
prior that applies to five digits and does not extend.

## The other training signal fails differently

The completion-trained network was never allowed to copy. If predicting is the
skill that transfers, it is the network that should show it.

It handles new images of familiar digits reasonably: **0.642**. On
unseen digits it scores **1.221** — worse than drawing the average digit.

It also loses the ability to find things. Identification accuracy on unseen
digits drops to **0.370**, against the recall-trained network's 1.000.

So neither objective produces knowledge that crosses a class boundary. One keeps
its finding ability and loses its predicting ability. The other loses both.

## What this means

Finding an image and understanding an image are different capabilities here, and
only one of them transfers.

Finding works by matching pixels. A visible top half is compared against sixteen
candidates, and the closest one wins. Nothing in that operation needs to know
what digit it is looking at. It works on a 9 for the same reason it works on a 3.

Predicting needs something else. To finish an unseen bottom half the network
needs to know how that kind of shape continues. That knowledge was built from
digits 0 to 4. It does not stretch to 5 to 9.

The clean version of the result: **perfect retrieval and chance-level prediction,
on the same images, in the same run.**

## What this does not establish

The split is one particular split. Digits 0 to 4 may be unusually poor
preparation for 5 to 9. A rotation of which digits are held out would say whether
the effect is about class novelty in general.

Ten classes is a small vocabulary. A dataset with more classes would show whether
prediction transfers once training covers enough of the space.

Both networks here were trained on unrelated context images. Reports on the
nearest-neighbour version of this task show the context can be made informative,
and that is untested under a class split.

## Sources

`results.jsonl` rows `exp8_sharedq` and `exp9_sharedq` (recall-trained and
completion-trained, digits 0-4, shared-query evaluation), and
`baselines_M16_r14_split` for the linear and average-image references. Figures
generated by `scripts/gen_report_12.py`. No training was run for this report.
