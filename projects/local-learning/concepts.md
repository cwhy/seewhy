# Local Learning vs Backprop — Concepts

What this project replicates: a table posted by [@mayfer on 2026-09-14](https://x.com/mayfer/status/2099541904241385615),
quote-tweeting Sakana AI's PC-ALM release. It is captioned "quick MNIST test":

| Method | Parameters | Test accuracy | Training | Inference: 10k images |
|---|---:|---:|---:|---:|
| Backprop | 56,128 | 87.39% | 0.324s | 0.0229s |
| PC-ALM | 56,128 | 86.28% | 13.485s | 0.0241s |
| Forward-Forward | 56,232 | 88.83% | 0.549s | 0.0781s |

The parameter counts identify the architectures exactly, and they are not the
same architecture (see "Model & loss").

## Task / data

MNIST, 60,000 training images and 10,000 test images, 28x28 greyscale, ten
classes. Images are flattened to 784 floats and standardised with the usual
MNIST constants, `(x/255 - 0.1307) / 0.3081`, matching the PC-ALM reference
implementation. Subsets, when used, are drawn class-balanced.

Chance is 10%: ten classes, and the test split has 1,000 images of each
(the balanced-subset draw keeps it that way for smaller n).

Forward-Forward needs the label inside the input. Following Hinton (2022), the
first ten pixels of the image are replaced by a one-hot label written in the raw
pixel domain, which after standardisation is 2.8215 for the hot pixel and
-0.4242 for the other nine.

## Model & loss

**Residual MLP** (backprop, PC, PC-ALM). The architecture of the PC-ALM paper,
width N, depth L, no biases, with z_0 = x:

    z_1    = (1/sqrt(784)) W_1 x
    z_i    = z_{i-1} + (1/sqrt(N L)) W_i relu(z_{i-1})     i = 2..L-1
    logits = (1/N) W_L relu(z_{L-1})

Weights are drawn from a unit normal; the scale factors sit outside them (the
paper's `gamma0 = 1` parameterisation). Loss is one-half squared error against
the one-hot label, summed over the ten outputs and averaged over the batch —
not cross-entropy, again following the paper.

At N=32, L=32 this is 784*32 + 30*32*32 + 10*32 = **56,128** parameters, the
tweet's first two rows exactly.

**Plain MLP** (Forward-Forward, and the backprop control). Dense layers with
biases, ReLU, no skips. Forward-Forward has no output layer: two hidden layers
of 66 units is 784*66 + 66 + 66*66 + 66 = **56,232** parameters, the tweet's
third row exactly. The backprop control on the same stack adds a 10-unit output
layer (56,902) and trains on the same squared-error loss.

## The three learning rules

**Backprop** — `jax.grad` of the loss. One global backward pass.

**Predictive coding (PC)** — the forward pass is written as a constraint,
`z_i = block_i(z_{i-1})`, and relaxed into a free energy

    F = loss + (1/2) sum_i ||r_i||^2,      r_i = z_i - block_i(z_{i-1}).

Inference runs T gradient steps on F with respect to the *activations*; each
step needs only the two neighbouring layers. Then the weights take one step on
the same F with the activations frozen. T = 2L, the paper's budget rule.

**PC-ALM** — the same, on the augmented Lagrangian

    L = loss + sum_i lambda_i^T r_i + (rho/2) sum_i ||r_i||^2,

with a dual ascent step `lambda_i <- lambda_i + alpha r_i` after each primal
step. alpha = 1, rho = 1. alpha = 0 recovers PC exactly, which is why PC is the
control that matters. In a deep *linear* network the settled lambda_i equals
that layer's backprop credit signal, which is the paper's central claim.

The activity step size is `eta_h = 1/lambda_max`, where lambda_max is the
largest eigenvalue of the inference energy's Hessian. We measure it by power
iteration at initialisation rather than reading the authors' lookup table; at
the headline cell the two agree to within 3%.

**Forward-Forward (FF)** — no backward pass and no global loss. Each layer
maximises its own *goodness* (mean square of its activations) on data carrying
the correct label and minimises it on data carrying a wrong one:

    loss_i = softplus(-(g(h_i^pos) - theta)) + softplus(g(h_i^neg) - theta)

with theta = 2. Each layer's input is detached, so training is layer-local, and
normalised to unit mean square, so a layer cannot inherit its predecessor's
goodness instead of computing its own. Classification sweeps all ten labels
through the network and takes the argmax of goodness summed over layers 2..n,
which costs ten forward passes. A linear softmax probe on detached activations
is trained alongside and reported as a secondary readout.

## Metrics

- **test_acc** — top-1 accuracy on the 10,000 test images. For FF this is the
  goodness sweep; `test_acc_probe` is the linear readout.
- **train_s** — wall-clock of the compiled training epochs, blocked on the
  device. Excludes XLA compilation, reported separately as `compile_s`.
- **infer_s** — wall-clock of one pass over all 10,000 test images, fastest of
  three, after compilation.
- **grad_cos_to_bp** — cosine between a local rule's weight update and the
  backprop gradient at the same weights, on one batch. 1.0 means the rule has
  reproduced backprop; the PC-ALM paper reports 0.909 for PC-ALM and 0.604 for
  PC at its Fashion-MNIST headline cell.

## Findings

1. **The table replicates.** Five seeds each: backprop 0.8704 ± 0.0100 (posted
   0.8739), PC-ALM 0.8533 ± 0.0085 (0.8628), Forward-Forward 0.8869 ± 0.0085
   (0.8883). PC-ALM/backprop training-time ratio 40.8x against the posted 41.6x.
2. **The parameter counts identify the architectures, and they differ.** 56,128
   is the PC-ALM paper's residual MLP at width 32, depth 32; 56,232 is a plain
   784-66-66 stack. Matching parameters across the rows holds the wrong thing
   fixed.
3. **Forward-Forward's advantage is architectural.** Backprop on the 784-66-66
   stack reaches 0.9116 ± 0.0058; Forward-Forward on a 31x32 stack falls to
   0.5751 ± 0.0744.
4. **Forward-Forward's deep collapse is a readout failure.** A linear probe on
   the 31-layer network's own activations reaches 0.8919 where its goodness
   sweep gets 0.5751. Not a learning-rate artifact (best of four rates: 0.5808).
5. **One shared learning rate drives the ranking.** Best rates differ 30x across
   arms; 1e-3 is best for Forward-Forward alone. Tuned per arm, the order of the
   three posted rows reverses: backprop 0.9374, PC-ALM 0.9277, FF 0.8869.
6. **The source paper's claim holds.** PC (alpha = 0) collapses from 0.9055 at
   two layers to 0.6137 at 32 while PC-ALM holds 0.8554; gradient cosine to
   backprop 0.81 against 0.89 at depth 32, 0.999 for both at depth 2.
7. **Forward-Forward degrades with training.** 0.901 at two epochs, 0.860 at
   ten, and its linear probe falls further, 0.9102 to 0.7956 — the features
   themselves get less separable.
8. **The reimplementation checks out against the authors' own run.** On their
   Fashion-MNIST cell, backprop 0.7770 vs 0.7866 and PC-ALM 0.7687 vs 0.7775,
   PC-ALM cosine 0.901 vs 0.909. PC is 4.7 points low and unexplained; their
   tabulated activity step does not account for it.

Report: <https://media.tanh.xyz/seewhy/26-09-15/local-learning_report_replication.html>
Paper: <https://media.tanh.xyz/seewhy/paper/local-learning_paper.html>
