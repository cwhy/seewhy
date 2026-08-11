# Report 3 — perfect retrieval and chance-level completion, on the same digits

exp8, exp9 (digit split), exp13, exp14 (fine-tuning probe), exp15–exp17
(state-size control), and `baselines_M16_r14_split`.

Normalised MSE: **1.0 = no better than predicting the average training image.**

## The digit split

Training pool: MNIST digits **0–4** only. The model never sees a 5, 6, 7, 8 or 9
in any role — not as context, not as a query, not as a target. Six conditions
instead of four, so that novel *images* and novel *classes* can be told apart:

| | pool | what is novel |
|---|---|---|
| A / C | train | nothing |
| E / F | test, digits 0–4 | the images |
| B / D | test, digits 5–9 | the images **and** the classes |

### Retrieval crosses the split intact

exp8, recall-trained:

| condition | nMSE | identification accuracy |
|---|---|---|
| A — nothing novel | 0.014 | 1.000 |
| E — novel images, seen classes | 0.020 | 1.000 |
| B — **novel classes** | 0.043 | **1.000** |

Chance for identification is 1/16 = 0.063. Reconstruction is three times worse on
unseen classes than on training images, but identification is *perfect*: shown
sixteen digits it has never encountered and the top half of one of them, the
model picks out the right one every time.

### Completion does not cross it at all

Same run, and its completion-trained counterpart, on the absent-target
conditions:

| | C (seen) | F (novel img, seen class) | **D (novel class)** |
|---|---|---|---|
| recall-trained (exp8) | 0.712 | 0.705 | **1.006** |
| completion-trained (exp9) | 0.015 | 0.648 | **1.224** |
| ridge, fitted on 0–4 | 0.589 | 0.582 | 0.851 |
| best soft look-up | 0.941 | 0.938 | 0.933 |

**1.006.** On digit classes it has never seen, the recall-trained model's
completion is worth exactly as much as predicting the average image — to within
noise, precisely nothing.

And the completion-trained model is *worse than nothing* on them: 1.224. It
learned a real prior — 0.648 on novel images of the classes it trained on — and
that prior is actively wrong when applied to a 7. Only ridge regression, at
0.851, beats the trivial baseline at all.

So on the *same* images, in the *same* episodes: retrieval transfers completely,
completion transfers not at all. This is the cleanest evidence in the project
that the two are not two grades of one ability. Retrieval needs a map from image
to key that keeps distinct images distinct — a generic property of pixels,
indifferent to which digits exist. Completion needs the conditional distribution
of the bottom half given the top, which is specific to the shapes in the training
set.

## The state-size control

The M-sweep in Report 2 changes two things at once: it overruns the memory *and*
enlarges the context. This sweep changes one. M stays at 16, model width stays at
256, the parameter count is unchanged (the projections are 256×256 however they
are sliced into heads), and only the memory's shape moves. **The evaluation
episodes are identical across all four runs.**

| state | content/state | A | id. acc. | gain | D |
|---|---|---|---|---|---|
| 16 384 (exp1) | 0.8 | 0.015 | 1.000 | 0.835 | 0.852 |
| 8 192 (exp15) | 1.5 | 0.017 | 1.000 | 0.800 | 0.819 |
| 4 096 (exp16) | 3.1 | 0.022 | 1.000 | 0.730 | 0.755 |
| 2 048 (exp17) | 6.1 | 0.031 | 1.000 | 0.646 | 0.681 |

Monotone in both directions, on episodes that never change. The information
account — "larger contexts generalise better because they contain more digits" —
predicts nothing should happen here, and gets the sign wrong. The capacity
account predicts exactly this. **The control passes.**

Two honest qualifications:

* It reproduces in *direction*, not magnitude. Even at the smallest state,
  retrieval has not collapsed (id. acc. 1.000, gain 0.646) — nowhere near the
  0.004 at M=256.
* Compression ratio alone does not predict the outcome: at a ratio near 3,
  exp16 reaches D 0.755 while exp4 (M=64, full state) reaches 0.658. How many
  items must be told apart matters alongside how many numbers there are to tell
  them apart with. Both are properties of the memory, which is what the argument
  needs, but they are not one knob.
* exp17 reaches 2 048 by shrinking the per-head key to 8 dimensions, so it
  confounds memory size with key dimensionality.

## Is the recall solution worth anything?

2 000 steps of completion training, from exp1's recall-trained weights (exp13)
against random initialisation (exp14). Same budget, schedule, data, seed — the
only difference is where the weights started.

| initialised from | D | B | gain |
|---|---|---|---|
| exp1's recall weights | 0.439 | 0.435 | 0.004 |
| random noise | 0.454 | 0.453 | 0.001 |

The head start is worth **0.015 nMSE, about 3% relative**. Two thousand steps
from scratch gets to the same place. And the fine-tuned model has spent its
retrieval getting there: gain 0.004, down from 0.835.

Whatever a recall-trained model has learned, it is not a representation that a
generalising model wants. It is machinery for matching and copying, and that
machinery is worth almost nothing to a model that has to actually know something
about digits.

## Caveats

* exp8, exp9, exp13, exp14, exp15, exp16 and exp17 are single runs. The seed
  spread on the M=16 recall configuration (Report 2) is ±0.014 on gain, which is
  the scale to read these differences against — the 0.015 fine-tuning gap is
  right at it.
* The digit split makes the training pool smaller (~30 000 images), so exp8/exp9
  are not perfectly comparable to the ten-digit runs. The comparisons drawn here
  are all *within* the split.
* MSE rewards hedging; see Report 1.

## Figures

* exp8 completions, all six conditions — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp8_grid.png>
* exp9 completions — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp9_grid.png>
* exp17 (smallest state) — <https://media.tanh.xyz/seewhy/26-08-12/recallgen_exp17_curves.svg>

## Where this leaves the project

The paper is written up at <https://media.tanh.xyz/seewhy/paper/recall-gen_paper.html>.
The one-line version: **within this setting, retrieval training buys no
generalisation, and what looks like generalisation emerging at large context is
the model abandoning the context.**
