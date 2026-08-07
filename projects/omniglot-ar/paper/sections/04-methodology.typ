#import "/template.typ": *

= Methodology <sec:method>

== Data

Omniglot via `dpdl-benchmark/omniglot`, loaded by
`shared_lib.datasets.load_omniglot()`. The two HuggingFace splits are the
originals: `train` is the background set and `test` the evaluation set.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [*split*], [*characters*], [*alphabets*], [*images*],
    [background (`bg`)], [964], [30], [19 280],
    [evaluation (`ev`)], [659], [20], [13 180],
  ),
  caption: [The two splits. Their character inventories are disjoint, and the
    loader raises if they are ever found to overlap.],
)

Images are resized to $28 times 28$ (bilinear) and *inverted*, so ink is high
and "ink" means $v > 0$ — matching MNIST, so the pixel-bin vocabulary means the
same thing on both. Ink covers 18.7% of a drawing.

== Model

A pre-norm transformer over the token bag, deliberately identical in depth,
width, head size and embedding scheme to `universal-ar/experiments39.py`. This
project varies the data and the token layout, not the architecture, so outcomes
remain comparable to the prior negative result.

$ e = "pos"[p] + "val"[v] + "ref"[r] (+ "lab"[ell]) $

with the label-field term present only in experiments 3 and 4. Then $L$ blocks
of multi-head self-attention and an MLP, each pre-normed and residual, and a
final projection to the value vocabulary. No causal mask.

#kv(
  ("layers / width / head dim", "4 / 256 / 32"),
  ("parameters", "3.38 M (3.38 M with the label field)"),
  ("optimiser", "adamw, lr 3e-4, weight decay 1e-4"),
  ("schedule", "warmup 200 then cosine decay"),
  ("gradient clip", "global norm 1.0"),
  ("effective batch", "16 episodes"),
  ("steps", "12000"),
)

Three implementation choices carry over from the prior work, each for a measured
reason: embeddings are looked up by *one-hot matmul* rather than a gather,
because a gather's backward scatter contends on the handful of rows that
mostly-background pixel values occupy and serialises (345× slower on MNIST);
`jax.checkpoint` wraps each block, because episodes run to a few thousand tokens
and the attention matrices dominate memory; and gradient accumulation runs
through `lax.scan` to keep the whole step inside one XLA computation.

== Loss and metrics

The loss is cross-entropy on the masked query-label tokens only, averaged over
scored tokens. Pixel completion is deliberately excluded: it is a separate claim
and belongs in a later experiment.

#figure(
  table(
    columns: (auto, 1fr),
    [*metric*], [*definition*],
    [$"acc"_"ev"$],
    [*The headline.* $N$-way accuracy on episodes built from evaluation
     characters, never seen in training. The argmax is restricted to the $N$
     label slots.],
    [$"acc"_"bg"$],
    [The same, on background characters (seen in training).
     $"acc"_"bg" - "acc"_"ev"$ is the memorisation gap.],
    [$"open"_"ev"$],
    [$"acc"_"ev"$ with the argmax over the *whole* value vocabulary, pixel bins
     included. It is lower than $"acc"_"ev"$ exactly when the model answers a
     label query with a pixel bin.],
    [$"train"$], [$N$-way accuracy on the training episodes themselves.],
    [$"nn"$], [Pixel 1-NN, see below.],
  ),
  caption: [Metrics. Restricting the argmax to label slots matters: an untrained
    head can answer a label query with a pixel bin, which scores zero and
    conflates "wrong class" with "emitted no class at all".],
)

Evaluation uses 64 fixed episodes per split, held constant across steps so the
curves are comparable. That is 320 scored queries at $N=5, Q=1$ and 128 at
$N=2$, giving standard errors of about 0.022 and 0.044 respectively.

== Baselines

Chance is $1/N$. Above it is necessary and nowhere near sufficient: on Omniglot,
raw-pixel nearest neighbour already gets a fair way there.

The 1-NN baseline is therefore computed *in-repo, on the same episodes, over the
same $C$ observed pixels the model receives*, with cosine distance. A baseline
measured on full images at a different resolution would be measuring a different
task and would not be a floor for anything reported here.

#callout(title: [Reading a result])[
  $"acc"_"ev" > "nn"$ is the bar for claiming in-context learning at all.
  $"acc"_"bg" - "acc"_"ev"$ is the memorisation gap. A large gap with
  $"acc"_"ev" approx$ chance would reproduce the prior failure on a new
  substrate; both at chance means nothing was learned to memorise.
]
