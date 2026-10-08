#import "/template.typ": *

= Method

The method is a GAN in which the "real data" of each episode is that
episode's context. The discriminator sees the context only through one KDA
state. It must accept the context images and reject the generator's images.
The generator tries to make images the discriminator accepts.
@fig:method shows one episode.

#fig(include "/diagrams/game.typ", caption: [
  The method. (a) One training episode. The 16 context images are written into
  the discriminator's KDA state $S$; queries are read against $S$. Real queries
  are copies of context images (target 1), fake queries are generated (target
  0). The generator receives noise $z$ and, in the context-aware version, the
  same context through its own KDA state. (b) The two evaluation probes of
  @sec:probes; neither involves training.
]) <fig:method>

== Notation

#notation(
  ($c_1, dots, c_M$, [the context: $M = 16$ images, each a vector in $[0,1]^784$]),
  ($x$, [a query image]),
  ($z$, [generator noise, $z tilde cal(N)(0, I_64)$]),
  ($e(dot)$, [an image encoder, $RR^784 -> RR^256$]),
  ($S$, [a KDA state: 4 heads, each $64 times 64$]),
  ($D(x; c)$, [discriminator logit for query $x$ given context $c$; $sigma(D)$ = P(in context)]),
  ($G(z; c)$, [generated image; the blind generator ignores $c$]),
  ($sigma$, [the logistic function $1 / (1 + e^(-u))$]),
)

== Discriminator

*Encoder.* $e(x) = W_2 thin "GELU"(W_1 x + b_1) + b_2$, with widths
$784 -> 512 -> 256$.

*Writing the context.* Each context image $c_t$, $t = 1 dots M$, in order,
gives per head ($h = 1 dots 4$, each a 64-wide slice of the 256 channels):

$ k_t = "l2norm"(W_k e(c_t)), quad v_t = W_v e(c_t), $
$ alpha_t = sigma(W_alpha e(c_t) + b_alpha), quad beta_t = sigma(W_beta e(c_t) + b_beta), $
$ S <- S dot "Diag"(alpha_t), quad S <- S + beta_t (v_t - S k_t) k_t^top , $

starting from $S = 0$. $b_alpha$ is initialised so that $alpha^(8 M) = 1 slash
e$, a memory horizon much longer than the context.

*Reading a query.* With $q = "l2norm"(W_q e(x))$ per head,

$ o = S q slash sqrt(64), quad D(x; c) = w^top "GELU"(W_h [o ; e(x)] + b_h) + b . $

The final state $S$ is the only path from the context to the output.

== Generator

*Blind.* $G(z) = sigma(W_5 thin "GELU"(W_4 thin "GELU"(W_3 z + b_3) + b_4) + b_5)$,
widths $64 -> 512 -> 512 -> 784$.

*Context-aware.* The generator has its own encoder and its own KDA state $S_G$,
written from the same context with the same rule. It reads $S_G$ with a query
made from the noise, $q = W_z z$, and feeds the result $o_G$ to the same MLP
alongside the noise: $G(z; c) = "MLP"([z ; o_G])$. Nothing in it copies a
context image; whether and how it uses the context is learned.

== Loss

For one episode with real queries $x_1 dots x_4$ (copies of context images)
and fakes $tilde(x)_j = G(z_j; c)$, $j = 1 dots 4$:

$ cal(L)_D = 1/8 [ sum_(i=1)^4 -log sigma(D(x_i; c)) + sum_(j=1)^4 -log(1 - sigma(D(tilde(x)_j; c))) ]
  + gamma / 2 dot 1/4 sum_(i=1)^4 norm(nabla_x D(x_i; c))^2 , $

$ cal(L)_G = 1/4 sum_(j=1)^4 -log sigma(D(G(z_j; c); c)) . $

Both are averaged over the 64 episodes of a step. Each step makes one update
of $D$, then one update of $G$ against the updated $D$. The gradient penalty
$gamma = 10$ applies to the real queries only.

== Training setup

#figure(
  table(
    columns: 2, align: (left, left), stroke: none,
    table.hline(),
    [*setting*], [*value*],
    table.hline(stroke: 0.5pt),
    [context size $M$], [16],
    [queries per episode], [4 real + 4 fake],
    [episodes per step], [64],
    [training steps], [20 000],
    [optimiser], [Adam @kingma2015adam, both networks],
    [learning rate], [$2 times 10^(-4)$, both networks, constant],
    [Adam $beta_1, beta_2$], [$0, 0.99$ (as in StyleGAN2 @karras2020stylegan2)],
    [gradient penalty $gamma$], [10, on real queries],
    [noise dimension], [64],
    [KDA heads × key size], [4 × 64],
    [parameters], [928 261 (discriminator); 1 576 724 (context-aware generator)],
    [seeds], [12 per generator (50–61)],
    [hardware, time], [one RTX 4090 per run, about 3 minutes per run],
    table.hline(),
  ),
  caption: [Every setting of the method. All numbers in this paper are
  medians or ranges over the 12 runs unless a single run is named.],
) <tab:hparams>

The Adam setting matters for reliability; @sec:stability compares it with
the more common $beta_1 = 0.5, beta_2 = 0.999$.

== Evaluation probes <sec:probes>

All probes use the test split and do not change any weights.

*Swap test.* Take an evaluation batch of episodes. Score its real queries
(context images) and the generator's images twice: once with each episode's
own context, once with the context of another episode in the batch. Report
the AUROC of real against generated under each. If the two are equal, the
discriminator ignores its state. If the AUROC falls with the wrong context,
the discriminator relies on its state.

*Digit classifier.* Build ten contexts, one per digit, each of 16 random test
images of that digit. Score every other test image against all ten, and assign
it the digit whose context gives the highest score. Report 10-way accuracy and,
per digit, the AUROC of "this digit vs the rest" under that digit's context.
Five random draws of the contexts are averaged. The discriminator never saw a
label, nor a single-digit context, in training.

*Density over real images.* Take 20 random contexts of 16 test images. Score
every other test image, about 10 000 per context. Compare the scores of context
images, of unseen images whose digit appears in the context, and of unseen
images whose digit does not.

*Generator context use.* For the same noise, compare the generator's output
under its own context and under another episode's, and divide by the change
caused by redrawing the noise under the same context. 0 means the context
has no effect.

*Collapse.* The pixel standard deviation across a batch of generated images,
averaged over pixels. Below 0.06 the generator draws essentially one image.
A run *fails* if, at the end, its digit-classifier accuracy is below 0.5 or
its generator has collapsed.

*Baselines.* The digit classifier is repeated with two scores that use no
learning: the negative pixel distance to the nearest of the 16 context images,
and a pixel-space RBF kernel density, $sum_i exp(-norm(x - c_i)^2 slash 2
s^2)$. The bandwidth $s$ is set by the median heuristic (the median pairwise
distance between test images), and also chosen from a grid on the evaluation
data itself, which gives that baseline its best possible score.
