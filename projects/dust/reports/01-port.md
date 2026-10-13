# The JAX port reproduces DUST's published numbers

We re-implemented DUST in JAX and checked it three ways. Backprop on our model reaches
**5.937** at 1M tokens (published 5.959) and **4.984** at 10M
(published 4.989). DUST at 256 draws reaches **6.074**
(published 6.067) and **5.939** at 1,024 draws (published 5.934),
which is level with our backprop. On identical weights, our gradient estimate and the authors'
PyTorch one agree layer by layer, and their gradient norms agree to 0.2%.
So differences between methods in the next reports are differences between methods, not between codebases.

## How DUST works, in one page

**The problem.** A language model has about 38 million parameters. Backprop gets the gradient
of the loss with respect to all of them from one backward pass. This project asks what you can
do with forward passes alone.

**The obvious way, and why it fails.** Perturb the weights at random, run a forward pass, and
see whether the loss went down. Do it for a *population* of perturbations and average. That is
evolution strategies, and EGGROLL is a fast version of it. But each forward pass returns *one
number* — the mean loss — and that number has to steer 38 million weights. The cosine between
the estimate and the true gradient grows only as √(P/d). With d = 3.8×10⁷ and P = 1,000 passes
that is 0.005. The estimate is almost all noise.

**DUST's move.** Perturb the *activations* instead, and do it separately at every token. A batch
is 16,384 tokens, and the loss is a sum of one term per token. If each token's loss mostly
responds to its own noise, then one forward pass is not one experiment. It is sixteen thousand.

![One forward pass, two ways](https://media.tanh.xyz/seewhy/26-10-07/dust_pass_value.svg)

*Left: EGGROLL. The pass is one population member and returns one number. Right: DUST. Every token
carries its own noise (coloured dots) and returns its own loss.*

**From rewards to a gradient.** For one linear layer, each draw adds noise `σ·aₜ` to the layer's
output at every token. A token's *reward* is how much lower its loss came out than the average
across the draws run together. Noise that helped, weighted by how much it helped and averaged over
draws, points *against* the gradient at that layer's output. That is an estimate of ∂L/∂y, the
quantity backprop gets from the chain rule. Multiply by the layer's cached input and sum over tokens,
and you have the weight gradient. It is the same outer product backprop forms.

![Reward to gradient](https://media.tanh.xyz/seewhy/26-10-07/dust_reward_to_gradient.svg)

**What makes it work on a real transformer.** Three things. *Interference:* jittering every layer at
once would let one layer's noise leak into another layer's reward, so DUST jitters different layers in
separate passes. *Attention internals:* queries, keys and values act through other tokens, so a token's
own loss barely sees them; DUST scores them against the estimated error at the attention output instead.
*The rest:* the output head is jittered directly on the cached logits, and the 16 residual-mixing scalars
use ordinary weight-space ES.

## 1. The losses land on the published ones

Test loss at the best-validation checkpoint, seeds 42–44. The published numbers are three-seed means.

| run | ours, mean ± sd | published | ours − published |
|---|---|---:|---:|
| backprop, 1M tokens | 5.937 ± 0.027 (n=3) | 5.959 | -0.022 |
| backprop, 10M tokens | 4.984 ± 0.008 (n=3) | 4.989 | -0.005 |
| DUST, 256 draws, 1M tokens | 6.074 ± 0.020 (n=3) | 6.067 | +0.007 |
| DUST, 1,024 draws, 1M tokens | 5.939 ± 0.010 (n=3) | 5.934 | +0.005 |

Every difference is smaller than the seed-to-seed spread. A model that knew only token frequencies
would score 6.994 on this test set, so the DUST and backprop numbers are well inside the range where
the network is using context.

## 2. The estimator matches the authors' code

Same random weights, same two-sequence batch, both implementations, cosine to the PyTorch autograd gradient.
First the control: the two frameworks' *autograd* gradients agree to a cosine of 1.00 on every layer
type, with equal norm. The models are the same function. Then the DUST estimates at 256 draws:

![Cosine to backprop, authors' code vs ours](https://media.tanh.xyz/seewhy/26-10-07/parity_cosines_h.svg)

| layer type | authors' PyTorch | this port |
|---|---:|---:|
| LM head | 0.765 | 0.757 |
| token embedding | 0.180 | 0.177 |
| attention output | 0.210 | 0.181 |
| MLP output | 0.121 | 0.135 |
| MLP input | 0.101 | 0.105 |
| total gradient norm ÷ backprop's | 3.328 | 3.323 |

The five large types agree to within 0.03. The remaining five (Q, K, V, value gate, value embedding) sit
within ±0.1 of zero in both. That is where the paper's own fitted law puts them at this population:
q 0.06, k 0.02, v 0.03, gate 0.05, value_emb 0.02. One reference run is 271 s on a 24-thread CPU.

## What this changes

- The port is faithful enough to carry the EGGROLL comparison. The next report puts the two methods
  on one ladder at 1M tokens.
- Backprop is reproduced at two budgets, so "DUST reaches backprop at 1k draws" is tested against our
  own baseline, not only the paper's. Here DUST (5.939 ± 0.010) and backprop
  (5.937 ± 0.027) are level. The paper has DUST slightly *below*
  (5.934 vs 5.959); our backprop is a little stronger than theirs.
- Not yet checked: DUST above 1,024 draws, and anything at 10M tokens. Seeds and data order
  differ from the release (a numpy permutation, not torch's), and weights are kept in f32 where the
  reference stores embeddings in bf16. Neither moved a number outside the seed spread.

Code: `lib/model.py`, `lib/dust.py`. Parity: `scripts/tmp/parity_export.py`, `parity_torch.py`.
