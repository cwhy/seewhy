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

## The four learning rules

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

**EGGROLL** — evolution strategies with low-rank perturbations
([arXiv:2511.16652](https://arxiv.org/abs/2511.16652)), ported from this repo's
`es` project. No derivative anywhere: the update is an average of P antithetic
perturbation pairs,

    eps_i = sigma * A_i B_i^T      (A_i: m x r, B_i: n x r, standard normal)
    f_i   = L(theta + eps_i) - L(theta - eps_i)
    g-hat = 1/(P sigma^2) * sum_i f_i eps_i

fed to Adam as if it were a gradient. Vectors get a dense Gaussian perturbation.
Rank r = 1 throughout. The perturbed loss is whichever loss the arm under test
uses, so the estimated quantity is exactly backprop's gradient. Defaults after
tuning: P = 512, sigma = 3e-3 (residual net) or 3e-4 (plain net), outer rate
3e-2 / 1e-2. Cost is 2P forward passes per weight update and no backward pass.

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
8. **EGGROLL trains well with an almost-orthogonal gradient estimate.** At
   P = 512 the cosine between its update and the true gradient is 0.094, and it
   reaches 0.9025 ± 0.0026 in one epoch — above backprop at the shared default
   rate. The measured cosine tracks sqrt(P/d) to within 5% across P = 32..2048.
9. **EGGROLL is the only arm indifferent to depth**: 0.8990 at 2 layers,
   0.9017 at 64, flat within seed spread, where backprop declines (0.9054 →
   0.8518 under the paper's depth-dependent rate rule) and PC collapses. It is
   also the only arm that prefers the deep-narrow net to the short-wide one
   (0.9025 vs 0.8272), tuned separately on each.
10. **Contention calibration.** EGGROLL's cells were measured while another job
   shared the GPU; re-measuring three arms with known idle timings gives a
   factor of 2.01, 2.17, 2.01 with accuracies unchanged, so EGGROLL's times
   halve for comparison.
11. **The reimplementation checks out against the authors' own run.** On their
   Fashion-MNIST cell, backprop 0.7770 vs 0.7866 and PC-ALM 0.7687 vs 0.7775,
   PC-ALM cosine 0.901 vs 0.909. PC is 4.7 points low and unexplained; their
   tabulated activity step does not account for it.

12. **Five epochs does not change the ordering, only the spread.** Re-tuning
   the rate for the longer budget first (six of eight arms want a smaller one),
   at five epochs: backprop 0.9620, PC-ALM 0.9405, backprop 2x66 0.9395, EGGROLL
   0.9316, EGGROLL 2x66 0.9101, Forward-Forward 0.9045, PC 0.7311, FF 31x32
   0.5596. The backprop-to-FF gap goes 1.3 points the wrong way (posted table),
   5.1 the right way (1 epoch, tuned), 5.8 (5 epochs, tuned).
13. **Forward-Forward's decay was a learning-rate effect.** At 1e-3 it peaks at
   two epochs and falls; at its five-epoch rate of 3e-4 it climbs and plateaus
   (0.597, 0.806, 0.902, 0.905, 0.901). Finding 7 above is corrected by this.
   The deep FF stack still degrades with training (0.6106 -> 0.5596).

14. **The Forward-Forward arm was audited, and two findings were withdrawn.**
   Mechanics verified (locality exactly zero across layers, chance accuracy when
   image and label are shuffled, loss and readout as specified). But uniform
   negative labels are too easy: with Hinton's hard negatives the arm gains
   1.7 points on 784-66-66 (0.9039 ± 0.0030) and 28.2 on the 31x32 stack
   (0.8570 ± 0.0183). So backprop's margin on FF's own architecture is 0.8
   points, not 2.5 — inside the spread — and finding 4 above (the deep collapse
   as a readout failure) is withdrawn: it was a training-signal failure of ours.
15. **Feedback alignment works at shallow depth and dies at depth.** Verified by
   B = W reproducing backprop exactly, and by matching backprop on the networks
   Lillicrap et al. use (1x800: 0.9685 vs 0.9732; 2x800 tanh: 0.9685 vs 0.9766).
   On the residual net: 0.8925 at 2 layers, 0.8791 at 4, 0.5343 at 8, 0.1879 at
   16, 0.0873 at 32, diverged at 64 — with the B-to-W alignment falling 0.303 to
   0.000 in step. With ReLU it fails by killing units (21% dead at 3e-4, 76% at
   3e-3); with tanh it does not.

Report: <https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_replication.html>
Forward-Forward audit: <https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_ff_audit.html>
Five-epoch report: <https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_five_epochs.html>
Paper: <https://media.tanh.xyz/seewhy/paper/local-learning_paper.html>
