# dust-history — Proposal

## The idea

DUST gets its population from many nudges at once: K forward passes per update, each adding a small random nudge to one place
in the network, at every word of the batch. That costs K+1 passes per update, and K runs from 64 to 16,384.

Here the population is **the history of past updates**. Each update spends one noisy pass (plus the clean pass DUST always
runs), with an independent nudge at every site at once. It is scored against the clean loss, as DUST scores a single draw. The
estimate goes into a window of the last N estimates, and the weights move along the window's *consensus*. A brain cannot run
many copies of itself in parallel, and N updates in a row cost N passes however they are used.

## What would make it work, and what would stop it

- **Averaging cancels noise.** N one-nudge estimates averaged have about the quality of a population of N, if the true gradient
  stays put while the window slides. `scripts/check_estimator.py` measures this half alone, at fixed weights.
- **The weights move.** An estimate from j updates ago was scored on an older network and different words. Its error grows
  with how far the weights have travelled since. So the window N has a sweet spot, and fewer, larger-step runs (61 updates)
  are the hard case. More, smaller updates are the easy one.
- **Cross-talk.** Nudging every site at once means each site's estimate also carries the other sites' nudges as noise. DUST
  avoids that by nudging one site per draw, at the price of one pass per draw.

## Comparisons

| | passes per update | updates | what it tells us |
|---|---|---|---|
| plain DUST, K = 64 (project `dust`) | about 65 | 61 | the bar at 1M tokens: 6.315 |
| history, N = 1 | about 3 | 61 | one-nudge SGD, no history |
| history, N = 4, 16 | about 3 | 61 | what the window buys at equal text |
| history, N large | about 3 | hundreds | the same compute as DUST gets many more updates |

Equal text is the 1M-token protocol of `projects/dust`. Equal compute gives the cheap version about 20 times more updates, so
it sees about 20 times more text: it reads the 10M-token file for up to 610 updates.

## Not in the first version

- The 2L residual-mixing scalars (DUST trains them by weight-space ES, two full passes per draw). They stay at their
  initial values here, and any parallel-DUST control zeroes their gradient to match.
- Aiming the nudges with the consensus ("guided" nudges). First, whether history alone pays.
- A learned reference loss in place of the clean pass: the clean pass is DUST's own, so it stays.

## Consensus rules

`mean` averages the window. `agree` averages it and keeps only the weights on which at least a fraction `tau` of the window
shares the sign of the mean. Momentum cannot do the second, because it blends estimates it cannot take apart again.

## Findings so far (2026-10-09), stopped by the user

All of this is the all-sites-at-once, one-nudge-per-update version; the plain `mean` window unless stated.

- **Estimator check** (`scripts/check_estimator.py`, fixed weights, tiny model): the cosine of the N-average to the backprop
  gradient rises with N in every layer type (head 0.26, 0.50, 0.75, 0.85 at N = 1, 4, 16, 32) and the size along the true
  gradient is near 1 for head, proj, fc. q, k, v stay near 0, as in the authors' own DUST.
- **exp1, equal text** (61 updates, 1M tokens): N = 1, 4, 16 give 7.231, 7.233, 7.262. Plain DUST at 64 draws: 6.315.
- **exp2, equal compute with DUST at 64 draws** (500 updates, 10M file): N = 1, 4, 16 give 7.050, 7.050, 7.054; N = 64 7.134;
  the `agree` rule 7.053 at N = 16 (tau 0.6), worse at tau 0.75. All plateau within 200 updates at about 7.06, the
  token-frequency floor (6.994): the head learns frequencies and the interior layers never start.
- **exp3 and exp4, equal compute with DUST at 1,024 draws** (6,100 updates, 100M file): the first two runs finished, N = 1 and
  N = 16 at lr 1/64, both 7.002 (the same curve, flat at 7.01 from update 1,600 on). Cancelled before the rest ran (N = 64,
  other rates).

Reading: with a nudge at every site in one pass, each site's estimate carries the others' nudges as noise, and the interior
layers get too little signal. A window averages noise that is independent over time; it cannot add signal. Not tested:
one site per update (no cross-talk), aimed nudges, a longer window at a smaller learning rate.
