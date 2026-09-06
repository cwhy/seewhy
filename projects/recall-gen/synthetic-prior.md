# A synthetic prior that covers MNIST, Fashion-MNIST and chess

*Design note. Nothing here has been run.*

## Why this project is unusually well suited to it

TabPFN trains on datasets sampled from a prior and is then applied to real
tabular data it has never seen. It works because a tabular column has no fixed
meaning across datasets — the model cannot memorise "column 3 is age", so it is
forced to infer the mapping in context.

The recall-gen network is already in that position, and not by design. `W_pix`
is a dense matrix over a flat vector: **the architecture has no notion of
geometry**. A mask over image rows and a mask over board files are the same kind
of object to it. Nothing in it knows that coordinate 0 and coordinate 1 are
adjacent pixels, or that coordinates come in groups of 13.

So a synthetic prior does not need to imitate images, or boards. It needs to
produce, per episode:

1. **items with identity** — distinguishable enough that recall is well-posed;
2. **visible-to-hidden correlation** — so completion is possible but not free;
3. **a range of difficulty** wide enough to contain the real datasets.

That is a much smaller ask than "synthesise plausible images".

## The prior

Per episode, sample a world, then draw M+Q items from it.

    k     ~ loguniform(1, 64)                   latent dimension
    A     ~ random W x k, power-law spectrum    mixing, exponent sampled
    sigma ~ {identity, tanh, relu, sign}        pointwise nonlinearity
    out   ~ {continuous in [0,1], simplex}      dense like pixels, or one-hot like a board
    z_i   ~ N(0, I_k)                           one per item
    x_i   =  squash(sigma(A z_i + b) + noise)

`k` is the dial that matters. Small `k` means items lie near a low-dimensional
manifold: the visible half strongly predicts the hidden half, and items are also
mutually similar, so retrieval is *hard* and completion is *easy* — the
Fashion-MNIST corner. Large `k` means items are near-independent: retrieval easy,
completion near-impossible — closer to the MNIST corner. Sampling `k` per episode
spans the range rather than picking a point in it.

The `simplex` output mode exists so chess is inside the prior's support rather
than outside it: draw a one-hot per group of 13 coordinates.

The mask is a random subset of coordinates, size ~ U(0.3, 0.7) of the width.
Contiguity would be wasted — the network cannot see it.

## The width problem, and the one real decision

MNIST and Fashion-MNIST are 784 wide. Chess is 832. One network cannot currently
be scored on both.

The fix is to pad every domain to a common width (1024, say) and carry a validity
channel, which the mask machinery already supports. That is what makes "one
network, all three datasets" possible at all, and it is a change to
`lib/domains.py` rather than to the model.

The decision that actually shapes the result is different, and it is this:

**Is the mixing matrix `A` resampled per episode, or shared across many
episodes?**

*Per episode* is the TabPFN analogue. The model can never memorise coordinate
structure, and must infer the world from sixteen items every time. This is the
purer experiment, and it makes a sharp prediction: **retrieval should transfer to
all three datasets essentially intact, and completion should sit near the
do-nothing line on all three** — because sixteen items are nowhere near enough to
infer an 832-dimensional structure in context. If that is what happens, it is a
clean result, and it says the retrieval mechanism needs no data-specific prior at
all.

*Shared across episodes* lets a universal low-level structure be learned in the
weights, which is what would be needed for completion to transfer. It is the
version that could actually beat the ridge baseline on a real dataset, and the
weaker test of the in-context claim.

A third option is a mixture: a fixed component shared across all episodes plus a
per-episode component, with the ratio swept. That is the most informative and the
most expensive.

## How it would be evaluated

Unchanged — `scripts/standard_eval.py`, with the synthetic pool as `train` and
each real dataset as `held`. The three novelty bands become:

| band | pool |
|---|---|
| A / C | synthetic items seen in training |
| E / F | fresh synthetic items from fresh worlds |
| B / D | MNIST, or Fashion-MNIST, or chess |

E/F is the control that separates "a world it has not seen" from "a world unlike
anything the prior generates". If B collapses while E holds, the prior's support
does not cover real data, and that is a statement about the prior rather than
about the network.

The references are the ones that already exist: ridge fitted on the real training
pool is the bar completion has to clear, and the soft look-up ceiling is the bar
for using the context at all.
