#import "/template.typ": *

// OBLIGATIONS
//  - Interpretation lives here, not in section 6. Say what each result means,
//    what else could produce it, and what would distinguish the readings.
//  - Never claim more than the evidence in section 6 supports.

= Analysis <sec-analysis>

== Why DUST is ahead

The ladder's gap has a cause that was measured directly rather than inferred. At 16,384 draws or passes, the
gradient estimate DUST hands its optimiser has a cosine of 0.99 with backprop's on the head, 0.73 on the attention
outputs and 0.63 on the MLPs. EGGROLL's is 0.024 on the head and about 0.01 elsewhere. The two differ by one to two
orders of magnitude in every layer type where both are measured.

EGGROLL's number needs no explanation beyond arithmetic. A population member returns one scalar, and every layer's
share of it is signal while the other layers' shares are noise. For a layer with a fraction $s$ of the gradient's norm
and $d_l$ parameters, averaging $n$ antithetic pairs along random directions leaves a cosine of
$s slash sqrt(s^2 + d_l slash n)$, which is about $s sqrt(n slash d_l)$ while $n$ is small. We measured $s$ and $d_l$
from the true gradient and set the prediction beside the measurement: for the head at 16,384 passes, 0.023 predicted
and 0.024 measured; for the attention output, 0.010 and 0.010; for the MLPs, 0.012 and 0.011. The prediction is within 0.002 of the measurement at
every population we ran. EGGROLL's low rank changes how cheaply a member is stored, and the
prediction treats it as irrelevant to how much a member tells you; the measurement agrees.

DUST escapes that arithmetic by changing what a member reports. A draw returns a loss per token, and a layer's estimated
output error at each token is built from that token's own reward. One forward pass therefore feeds about sixteen
thousand separate estimates, each of dimension $d_"out"$ rather than $d_"out" times d_"in"$. The measured cosines
climb with population in every layer type, along the shape the paper's fitted law predicts, and sit above that law by
0.01 to 0.16 on our 100M-token checkpoint. The checkpoint is ours and the law is fitted to theirs, so the offset is not a
discrepancy; the shape is the claim.

Two layer types lag. Keys and value embeddings reach only 0.21 and 0.19 at the largest population. They are the layers
DUST credits through its *estimate* of the attention-output error rather than through the loss. Their estimate inherits
the attention output's own error, which is itself still climbing at 16,384 draws. This predicts that the ladder will flatten
before it reaches backprop's loss if those layers hold up the optimiser. We have no ladder at populations high enough
to test that.

== How large is the advantage

*The ordering reproduces, and by a margin the baseline's weakness cannot account for.* DUST at 64 draws beats EGGROLL
at 1,024 passes by 0.82 nats at about a fifteenth of the cost, and at 4,096 passes by 0.65 nats at about a sixtieth. The gap at 256
is 1.33 nats here against 1.13 in the paper. Our EGGROLL is weaker than the paper's, by 0.34, 0.21, 0.09 and 0.08 nats at 64, 256,
1,024 and 4,096 passes, but the ladder's gap is six times the baseline's at 256, so the conclusion survives any plausible repair of it.

*The order of magnitude of the efficiency advantage reproduces too, with a wide interval.* The paper's headline is a factor of
$10^3$ to $10^4$, obtained by extrapolating EGGROLL's ladder to the population at which it would reach DUST's loss at 64
draws. Their fits give 6,000 to 8,200 times 64 at 1M tokens. On our four-point ladder the line through the top three gives 3,554,
the line through the top two 16,810, and the power law has a floor of 6.55, above DUST's 6.315, so it says EGGROLL never gets
there. The straight lines bracket the paper's numbers; the power law is the paper's own caveat, a ladder that flattens.

We nearly reported something different. With three points, before the 4,096-pass rung, the same line gave 547, a tenth of the
paper's, and the story was that our EGGROLL, weaker at small populations and steeper afterwards, had a much cheaper extrapolated
cost. The fourth point removed it: EGGROLL's gain per doubling fell from 0.158 over 64 to 1,024 passes to 0.081 over 1,024 to
4,096, against the paper's 0.095 and 0.077. The lesson is about the method, not the number. A straight line in log population through
the early part of a bending ladder is not an extrapolation anyone should trust, and ours, like the paper's, is drawn through few
points. What the data support is that matching DUST's 64 draws takes EGGROLL thousands of times the population if it can be matched
at all.

== DUST and backprop

Under SGD, DUST closes on backprop as population grows: +0.378, +0.137 and +0.002 nats above it at 64, 256 and 1,024 draws, against
a backprop spread of 0.027. At 1,024 draws they are level. The paper has DUST below backprop there, and at 1M tokens our
backprop (5.937) is a little stronger than theirs (5.959), which accounts for the difference. At 4,096 draws DUST is 0.047 below
our backprop on one seed, the paper's 0.048, so the sign of the paper's claim shows up. One seed against a spread of 0.027 is
about 1.7 standard deviations, enough to call it suggestive and not enough to call it established. The paper's further suggestion,
that activation-space search finds better directions than the gradient, is untested here. One confound is closed: backprop with DUST's own rates
for the value embeddings and the residual scalars scores 5.938 ± 0.035 (exp18), the same as with its base rate, 5.937 ± 0.027. At 10M tokens the picture is the
other way round: DUST at 256 draws is 0.248 behind backprop, the paper's gap, and the paper's DUST stays behind at every
population it ran at that budget.

== What the optimiser changes <ana-opt>

*The SGD headline does not survive a better optimiser.* Under SGD DUST at 1,024 draws is level with backprop. Under Muon backprop gains 0.57 nats and DUST
0.47, so DUST is 0.095 behind at the same population, and under Adam 0.478 behind (one seed each). What survives is the ordering of DUST and EGGROLL:
DUST is ahead of EGGROLL by more than a nat at every population under every optimiser, by 1.30 to 1.95 nats within the same optimiser. "DUST reaches the loss
of backprop at 1,024 draws" is a statement about SGD, the optimiser the released recipes were tuned for.

The DUST paper's own Adam ladder tells the same story with less to see. It fits a power law to DUST's Adam ladder and reports a limit of 5.248 (bootstrap interval
5.17 to 5.30), below its backprop's 5.361, and reads that as DUST working with optimisers made for backprop. Our backprop under Adam, 5.358 ± 0.014, agrees with theirs.
Our DUST cells have the paper's shape in one respect, Adam lowers DUST at every population, and not in the other: DUST's distance to backprop under Adam is +0.678, +0.502 and
+0.478 nats at 64, 256 and 1,024 draws, shrinking and then flattening, and the last two cells differ by less than the 0.08 nats of run-to-run noise measured in @res-opt. Three populations do not support a fitted
limit, and the 1,024-draw cell is far from backprop. Under Muon the distance shrinks faster: +0.493, +0.245, +0.095 and, on one seed, +0.036 at 4,096 draws. It more than halves with each quadrupling of the
population and has not reached zero. Four points do not say whether it will, and the extrapolation of EGGROLL's ladder (@res-extrap) is a reason not to guess.

*Why Muon helps DUST and not EGGROLL* we did not isolate. The DUST paper finds the same of Adam for EGGROLL: its tuned Adam ladder lands within 0.01 of its SGD ladder. Two facts bound the explanation. Muon is no better than Adam for backprop (5.372 against 5.358), so it is not
simply the better optimiser at this scale. And the two estimates differ in what they get right: DUST's has a cosine to the gradient of 0.06 to 0.99
depending on population and layer, EGGROLL's about 0.01 (@sec-results). One reading is that Muon's equalised singular values suit an estimate whose directions are partly
right and whose size along each is not to be trusted, and are of no use when the directions are noise. It is a hypothesis, and no run here tests it. A test
would be the cosine between the orthogonalised estimate and the orthogonalised true gradient, layer by layer, which we did not measure.

*What the Adam numbers can and cannot say.* DUST under Adam is 0.17 to 0.37 nats behind DUST under Muon. Part of that may be our reconstruction of the Adam
recipe, since the paper's split of draws across blocks is not given, and part may be the learning-rate search: Adam's single-seed cells carry about 0.08 nats of
nondeterministic noise, and the 1,024-draw cells take a rate from a 256-draw grid that carried it. The Adam result that stands is the one that is larger than the noise:
Adam lowers DUST at every population, by 0.28, 0.21 and 0.10 nats at 64, 256 and 1,024 draws, but by less than Muon.

== What stays unexplained

The remaining distance between our EGGROLL and theirs. We exhausted the knobs the paper names that were within reach
(rate, noise scale, Adam constants, fitness shaping, precision) and none moved the loss by more than 0.06. Their EGGROLL-Transformer
is not released, so the difference may be in how perturbations are applied to the embeddings, the head or the scalars,
in their loss normalisation, or in something else. The distance shrinks from 0.34 nats at 64 passes to 0.08 at 4,096, so whatever it
is matters most at small populations. It matters for one thing in particular: the extrapolated efficiency factor, which depends
on the slope of EGGROLL's ladder.
