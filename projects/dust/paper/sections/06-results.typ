#import "/template.typ": *

// OBLIGATIONS
//  - Every metric carries its chance level or baseline IN THE SAME ROW.
//  - Variance across seeds, or an explicit statement that a number is a single run.
//  - One figure per claim. Report what happened; interpretation is section 7.

= Results <sec-results>

Every loss is the test loss of the checkpoint with the lowest validation loss, in nats per token. Two reference
points go with every ladder: backprop, measured here, and the unigram floor of 6.994, the loss of a model that
has learned token frequencies and no context (@sec-task). Unless a cell says otherwise it is a mean over seeds
42, 43 and 44, with the sample standard deviation beside it. Wall-clock times are seconds per update on one RTX 4090,
after compilation.

== The port reproduces the published baselines <res-port>

Before comparing methods, the network, the data and the optimiser have to give the paper's numbers.

#table(
  columns: (1.6fr, auto, auto, auto),
  align: (left, right, right, right),
  [*Run*], [*Ours*], [*Published*], [*Ours − published*],
  [backprop, 1M tokens], [5.937 ± 0.027], [5.959], [−0.022],
  [backprop, 10M tokens], [4.984 ± 0.008], [4.989], [−0.005],
  [DUST, 64 draws, 1M], [6.315 ± 0.043], [6.290], [+0.025],
  [DUST, 256 draws, 1M], [6.074 ± 0.020], [6.067], [+0.007],
  [DUST, 1,024 draws, 1M], [5.939 ± 0.010], [5.934], [+0.005],
)

Each difference is smaller than the spread across seeds, which is the most a three-seed mean can resolve.
Run on identical randomised weights, the authors' PyTorch DUST and ours produce gradient estimates with
the same cosine to the autograd gradient in every layer type to within 0.03 for the five large ones,
and the same total norm (3.328 against 3.323 times backprop's); the two frameworks' autograd gradients
agree to cosine 1.00.

#fig(include "/figures/parity_cosines.typ",
  caption: [Cosine between DUST's gradient estimate and the autograd gradient at 256 draws, for the authors'
    PyTorch code and for this port, on identical weights and one batch. Q, K, V, the value gate and the value
    embedding are within 0.1 of zero in both.])

== The ladder at 1M tokens <res-ladder>

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (right, right, right, right, right, right),
  [*Population*], [*DUST, ours*], [*DUST, paper*], [*EGGROLL, ours*], [*EGGROLL, paper*], [*Gap, ours (paper)*],
  [64], [6.315 ± 0.043], [6.290], [7.764 ± 0.011], [7.428], [1.45 (1.14)],
  [256], [6.074 ± 0.020], [6.067], [7.409 ± 0.012], [7.200], [1.33 (1.13)],
  [1,024], [5.939 ± 0.010], [5.934], [7.133 ± 0.004], [7.046], [1.19 (1.11)],
  [4,096], [5.890#super[†]], [5.911], [6.970#super[†]], [6.892], [1.08 (0.98)],
)

#super[†] One seed. The gap is EGGROLL minus DUST. Backprop scores 5.937 ± 0.027 (published 5.959), and the unigram floor is 6.994. Population is draws for DUST and
forward passes of the batch for EGGROLL, two per antithetic pair. DUST at 64 draws is below the unigram floor; EGGROLL is above
it at 1,024 passes and just below it, 6.970, at 4,096. DUST at 1,024 draws is level with our backprop, +0.002 with a backprop spread of 0.027; the
paper has it slightly below. At 4,096 draws DUST is 0.047 below our backprop on one seed (the paper: 0.048 below).

#fig(include "/figures/ladder_1M.typ",
  caption: [Test loss at 1M tokens against population. Dark: this work, three seeds (one at 4,096). Light: the paper's Table 1.
    Dashed: the unigram floor, and backprop (ours and the paper's).])

*A control on the rates.* The released DUST optimiser gives the value embeddings and the residual scalars rates of their own (0.3 and 0.03),
where the released backprop baseline uses its base rate. Backprop with DUST's rates for those two groups scores 5.938 ± 0.035 against 5.937 ± 0.027
(exp18, three seeds), so the groups are not what brings DUST level with backprop.

DUST at 64 draws, 6.315, beats EGGROLL at 1,024 passes, 7.133, by 0.82 nats, and EGGROLL at 4,096 passes, 6.970, by 0.65. Validation loss over training tells the
same story from the start: DUST at 256 and 1,024 draws follows backprop's curve closely, while every EGGROLL curve
falls behind early.

#fig(include "/figures/curves_1M.typ",
  caption: [Validation loss over the 61 updates at 1M tokens, mean over seeds, for backprop, DUST and EGGROLL at 256
    and 1,024.])

*Cost.* A DUST draw and an EGGROLL forward pass cost about the same here. At equal population EGGROLL takes between
0.95 and 1.11 times as long per update.

#table(
  columns: (auto, auto, auto, auto),
  align: (right, right, right, right),
  [*Population*], [*DUST s/update*], [*EGGROLL s/update*], [*EGGROLL ÷ DUST*],
  [64], [2.2], [2.1], [0.95],
  [256], [7.6], [8.2], [1.08],
  [1,024], [29.5], [32.4], [1.10],
  [4,096], [116.5], [129.5], [1.11],
)

Backprop takes 0.13 s per update. So the comparison at equal population is also at equal compute, and DUST at 64 draws
(2.2 s) beats EGGROLL at 1,024 passes (32.4 s) at about fifteen times lower cost, and EGGROLL at 4,096 passes (129.5 s)
at about sixty times lower.

#fig(include "/figures/cost_1M.typ",
  caption: [Test loss at 1M tokens against measured seconds per update, one point per population.])

== Our EGGROLL against the paper's <res-baseline>

Our EGGROLL is weaker than the paper's: by 0.34, 0.21, 0.09 and 0.08 nats at 64, 256, 1,024 and 4,096 passes, the gap
closing with population. We tuned it on seed 42 over the Adam rate, the noise scale $sigma$, the paper's Adam constants
($beta_1 = 0.5$, $beta_2 = 0.95$) and, at 64 passes, rank-shaped fitness and float32 compute. The rate is the
constant that matters: at 64 passes and $sigma = 0.003$ the loss runs 8.244, 8.064, 7.815 and 8.319 for rates 0.0003,
0.001, 0.003 and 0.01. A decade of $sigma$ moves it by 0.014. The paper's Adam constants gain 0.04 to 0.06; rank shaping
gains 0.02 and float32 0.002, both inside a seed's spread. None of it closes the gap.

#fig(include "/figures/eggroll_tuning.typ",
  caption: [EGGROLL tuning at 1M tokens, one seed: test loss against Adam rate for each setting tried, with the paper's
    EGGROLL losses at 64 and 256 passes as dashed lines.])

== Gradient alignment <res-align>

On a 100M-token backprop checkpoint, one batch of eight held-out sequences, at 16,384 draws or passes:

#table(
  columns: (1.4fr, auto, auto, auto),
  align: (left, right, right, right),
  [*Layer type*], [*DUST*], [*EGGROLL*], [*Paper's DUST law*],
  [LM head], [0.989], [0.024], [0.978],
  [attention output], [0.728], [0.010], [0.573],
  [MLP (input, output mean)], [0.625], [0.011], [0.472],
  [token embedding], [0.552], [0.000], [0.494],
  [values (V)], [0.412], [0.014], [0.258],
  [queries (Q)], [0.574], [0.005], [0.419],
  [keys (K)], [0.212], [0.005], [0.161],
)

Cosine to the backprop gradient. DUST rises with population in every layer type and reaches 0.99 on the head by
16,384 draws; at 64 draws the head is 0.385 and the MLPs are at 0.06. In the layer types listed, EGGROLL stays below 0.025. Its
measured cosines match the prediction for random-direction estimation, $s_l slash sqrt(s_l^2 + d_l slash n)$ (@sec-analysis):
for the head at 16,384 passes, 0.024 measured and 0.023 predicted. DUST's cosines at 16,384 draws are above
the paper's fitted law for its checkpoint by 0.01 to 0.16.

#fig(include "/figures/cosine_vs_eggroll_bp100M_step6100.typ",
  caption: [Cosine to the backprop gradient against population on the 100M-token checkpoint, for DUST and EGGROLL, in
    three layer types.])

At the population the paper quotes, 131,072 passes, EGGROLL's cosines on this checkpoint are 0.064 on the head and at most 0.036
on every other type: under 0.05 except the head, as the paper reports. From 65,536 to 131,072 passes they grow by 1.36 to 1.52,
where a square-root law says 1.41. One measurement each, 35 and 70 minutes.

== How much population would EGGROLL need? <res-extrap>

The paper's Table 2 extrapolates EGGROLL's 1M-token ladder to the population at which it would match DUST's loss at 64 draws
(6.315 here) and reports it as a multiple of 64. The same fits on our ladder of 64, 256, 1,024 and 4,096 passes:

#table(
  columns: (1.8fr, auto, auto),
  align: (left, right, right),
  [*Fit*], [*Ours*], [*Paper*],
  [straight line in log population, top three], [3,554×], [8,200×],
  [straight line, top two], [16,810×], [6,000×],
  [power law $A P^(-alpha) + E$, all points], [never (floor 6.55)], [23,000×],
)

The fits spread, as they do in the paper. Before the 4,096-pass rung we had three points and the line through them gave 547×:
the ladder bends. Our EGGROLL gained 0.158 nats per doubling of population from 64 to 1,024 passes against the paper's 0.095,
and 0.081 from 1,024 to 4,096 against 0.077. The 4,096 cells are single seeds.

== The ladder at 10M tokens <res-10m>

One seed for DUST and EGGROLL at 256 draws or passes, three for backprop.

#table(
  columns: (1.6fr, auto, auto, auto),
  align: (left, right, right, right),
  [*Run*], [*Ours*], [*Published*], [*Ours − published*],
  [backprop], [4.984 ± 0.008], [4.989], [−0.005],
  [DUST, 256 draws], [5.232], [5.231], [+0.001],
  [EGGROLL, 256 passes], [7.128], [6.906], [+0.222],
)

The unigram floor at this budget is 6.978. DUST is 0.248 behind backprop; the paper's DUST is 0.242 behind. EGGROLL's settings are its
1M-token winner, not tuned here. The EGGROLL–DUST gap is 1.90 nats against the paper's 1.67.

#fig(include "/figures/ladder_10M.typ",
  caption: [Test loss at 10M tokens against population. Dark: this work, one seed at 256. Light: the paper. Dashed: the unigram
    floor and backprop.])

== The ladder at 100K tokens <res-100k>

At 100K tokens a run is six updates. Every method is tuned here, DUST and EGGROLL on the paper's grids with seed 42
and the winner repeated, backprop likewise, with the grid extended one step wherever a winner sat on an edge.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (right, right, right, right, right, right),
  [*Population*], [*DUST, ours*], [*DUST, paper*], [*EGGROLL, ours*], [*EGGROLL, paper*], [*Gap, ours (paper)*],
  [64], [7.264 ± 0.006], [7.240], [8.276 ± 0.006], [7.961], [1.01 (0.72)],
  [256], [7.235 ± 0.114], [7.165], [8.135 ± 0.028], [7.605], [0.90 (0.44)],
  [1,024], [7.186 ± 0.032], [7.151], [7.804 ± 0.009], [7.386], [0.62 (0.23)],
)

The gap is EGGROLL minus DUST. Backprop scores 7.371 ± 0.184 here (published 7.216), and the unigram floor from the 100K prefix is 7.027. *Nothing
here is below that floor*, ours or the paper's: six updates are not enough for any method to use context, and the
ranking is among methods still learning how often each token occurs. DUST is ahead of EGGROLL at every population, by
0.62 to 1.01 nats against the paper's 0.23 to 0.72, so the same gap is wider here. Our backprop is noisy at this budget, with a
spread of 0.184 over three seeds against 0.027 at 1M tokens, so we draw no conclusion about DUST against backprop at 100K.

#fig(include "/figures/ladder_100K.typ",
  caption: [Test loss at 100K tokens (six updates) against population. Dark: this work. Light: the paper. Dashed: the
    unigram floor from the 100K prefix, and backprop. Nothing reaches the floor.])

== Adam and Muon at 1M tokens <res-opt>

Everything above uses the released optimisers: SGD with momentum for backprop and DUST, Adam for EGGROLL. This section repeats the 1M-token
ladder under Adam and under Muon for all three methods, with the settings of @method-opt. Each (method, optimiser, population) searched the matrix
learning rate over three values on seed 42 (exp19 to exp23), judged on validation loss. The winner was then repeated on seeds 43 and 44.
The 1,024-draw, 4,096-draw and 1,024-pass cells take the 256-population winner's rate and run on seed 42 alone. EGGROLL's Adam column is the ladder above.

#table(
  columns: (1.6fr, auto, auto, auto),
  align: (left, right, right, right),
  [*Run*], [*SGD*], [*Adam*], [*Muon*],
  [backprop], [5.937 ± 0.027], [5.358 ± 0.014], [5.372 ± 0.009],
  [DUST, 64 draws], [6.315 ± 0.043], [6.036 ± 0.006], [5.864 ± 0.020],
  [DUST, 256 draws], [6.074 ± 0.020], [5.860 ± 0.103], [5.617 ± 0.011],
  [DUST, 1,024 draws], [5.939 ± 0.010], [5.836#super[†]], [5.466#super[†]],
  [DUST, 4,096 draws], [5.890#super[†]], [–], [5.408#super[†]],
  [EGGROLL, 64 passes], [–], [7.764 ± 0.011], [7.816 ± 0.015],
  [EGGROLL, 256 passes], [–], [7.409 ± 0.012], [7.438 ± 0.011],
  [EGGROLL, 1,024 passes], [–], [7.133 ± 0.004], [7.148#super[†]],
)

#super[†] One seed. The SGD column is the ladder above, where EGGROLL was not run with SGD. The unigram floor is 6.994.

#fig(include "/figures/optimiser_zoom.typ",
  caption: [Test loss at 1M tokens against DUST's population under SGD, Adam and Muon (violet, light to dark), with backprop under the same
    three optimisers as flat lines (red, light to dark; Adam and Muon nearly coincide). Mean over three seeds, except at 4,096 draws and
    under Adam and Muon at 1,024 draws, which are one seed.])

Backprop gains 0.58 nats from Adam and 0.57 from Muon, and the two are level. DUST gains from both, and more from Muon: at 64, 256 and 1,024 draws Muon
lowers it by 0.45, 0.46 and 0.47 nats, and Adam by 0.28, 0.21 and 0.10. EGGROLL is not helped by Muon. It sits above its Adam arm by 0.052, 0.029
and 0.015 nats at 64, 256 and 1,024 passes, against a seed spread of 0.011 to 0.015.

#fig(include "/figures/optimiser_ladder.typ",
  caption: [Test loss at 1M tokens against population for DUST under SGD, Adam and Muon and EGGROLL under Adam and Muon, with backprop
    under each optimiser as a flat line (red, light to dark; Adam and Muon nearly coincide).])


Backprop gains more than DUST does, so DUST's distance to backprop within each optimiser changes:

#table(
  columns: (auto, auto, auto, auto),
  align: (right, right, right, right),
  [*DUST draws*], [*SGD*], [*Adam*], [*Muon*],
  [64], [+0.378], [+0.678], [+0.493],
  [256], [+0.137], [+0.502], [+0.245],
  [1,024], [+0.002], [+0.478#super[†]], [+0.095#super[†]],
  [4,096], [−0.047#super[†]], [–], [+0.036#super[†]],
)

DUST minus backprop under the same optimiser; positive means DUST is behind. At 1,024 draws DUST is level with backprop under SGD, 0.095 behind it under Muon on one seed, and 0.478
behind it under Adam on one seed. At 4,096 draws, one seed, it is 0.047 ahead of backprop under SGD and 0.036 behind it under Muon.

#fig(include "/figures/optimiser_gap.typ",
  caption: [DUST's test loss minus backprop's under the same optimiser, against DUST's population, at 1M tokens. Zero, dashed, is level with backprop;
    positive means DUST is behind. The table above gives the numbers.])

*Rate check.* The 1,024-draw Muon cell used 0.003, the 256-draw winner. At 0.002, 0.003 and 0.005 it scores 5.488, 5.466 and 5.547 on seed 42 (exp25, exp28), and
0.003 also has the lowest validation loss, so the carried-over rate was the best of the three. The 4,096-draw cell's rate is unchecked.

#fig(include "/figures/optimiser_curves.typ",
  caption: [Validation loss over the 61 updates at 1M tokens, mean over seeds, for backprop (red) and DUST at 256 draws (violet) under SGD, Adam and Muon (light to dark).])

DUST at 256 draws under Muon pulls ahead of DUST under SGD and under Adam early in training and stays ahead. Backprop under Adam and under Muon track each
other closely to the end, and both stay well below backprop under SGD.

*The grids.* Test loss at each matrix learning rate searched, seed 42. The top Adam rate does badly in all three grids, 6.703 to 7.102, so Adam's
winners sit in the middle of the grid, as Muon's do.

#table(
  columns: (1.4fr, auto, auto, auto),
  align: (left, right, right, right),
  [*Run*], [*Rate 1*], [*Rate 2*], [*Rate 3*],
  [backprop, Adam (0.001, 0.002, 0.004)], [5.513], [5.342], [6.703],
  [backprop, Muon (0.001, 0.003, 0.01)], [5.513], [5.381], [5.576],
  [DUST 64, Adam (0.001, 0.00175, 0.003)], [6.329], [6.035], [7.021],
  [DUST 64, Muon (0.001, 0.003, 0.01)], [6.138], [5.850], [6.269],
  [DUST 256, Adam (0.001, 0.0015, 0.003)], [5.981], [5.895], [7.102],
  [DUST 256, Muon (0.001, 0.003, 0.01)], [5.883], [5.613], [6.018],
  [EGGROLL 64, Muon (0.001, 0.003, 0.01)], [7.865], [7.827], [7.870],
  [EGGROLL 256, Muon (0.001, 0.003, 0.01)], [7.484], [7.427], [7.460],
)

*Run-to-run noise.* The same DUST cell run twice (seed 42, 256 draws, Adam at 0.0015) scored 5.895 and 5.978. The two runs are identical for the first
four updates and then part, which is GPU floating-point nondeterminism amplified by training. A one-seed grid under Adam therefore carries about
0.08 nats of noise before seeds differ at all, and DUST's three-seed spread at 256 draws under Adam, 0.103, shows it. Under Muon the spreads are 0.009 to 0.020.

*Cost.* Under Adam and Muon a DUST update draws 1.375 times the nominal population and takes 2.8 s and 2.9 s at 64 draws, and 10.4 s and 10.5 s at 256,
against 2.2 s and 7.6 s under SGD. Backprop takes 0.13 s under Adam and 0.14 s under Muon. These runs shared the machine with another user's jobs
for part of their time (@sec-limits), so seconds per update at 1,024 draws and above are not quoted.
