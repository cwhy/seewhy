# Recall-GAN — Proposal

## Premise

[recall-gen](../recall-gen/proposal.md) trained KDA models to *complete* an
image from a context and found the objective buys a similarity metric and no
knowledge. This project asks the dual question with a discriminator: can a
KDA state be trained into a sharp **in-context membership test** — "was this
image in the context?" — by playing it against a generator that hunts for the
images it gets wrong?

The discriminator is the product. The generator is a source of hard
negatives. The game is set up so the discriminator should win; how finely it
wins is the measurement.

## Open questions, in the order they gate each other

| # | Question | How it is answered |
|---|---|---|
| Q1 | Does a context-aware generator move its outputs toward the context, under plain labels (generated = not in context)? | exp1: `closer_gen` vs its control `closer_real`, `ctx` arm vs `blind` arm |
| Q2 | If it does, does D's accuracy on generated images collapse — i.e. does the plain rule break? | exp1: `acc_gen` over training |
| Q3 | Do adversarial negatives make D a better membership test than real negatives alone? | `auroc_real` of the `blind`/`ctx` arms vs the `none` arm, plus a nearest-neighbour-negative baseline (later) |
| Q4 | At what resolution ε\* does D stop rejecting perturbed context images, against M and state size? | PGD attack on a trained D (later) |

Q1 decides the label rule for everything after it. If the answer is no, the
plain rule stands. If yes, labels switch to "within ε of a context image =
member", and Q4 sets ε.

## Predictions (labelled so they can be checked)

- `none` reaches `auroc_real` well above 0.5 at M=16: a 16k-float state has
  room for 16 items. Held weakly — the encoder has to spread same-class digits
  apart.
- `blind` collapses toward a few population-typical digits (`gen_std` falls)
  and stays at `closer_gen` ≈ 0.5.
- `ctx`: genuinely uncertain. Either it finds the copying route and
  `closer_gen` → 1 with `acc_gen` falling, or it settles where `blind` does.

## Known shortcut

On the generated slots, D can win by detecting "generated" without reading
the context. Real non-members are always in the mix, so this does not make
the membership task trivial, but `acc_gen` alone cannot tell the two routes
apart. If the plain rule survives exp1, a later run writes generated images
into the context too (a 2×2 of real/generated × member/non-member) so that
realness carries no information.
