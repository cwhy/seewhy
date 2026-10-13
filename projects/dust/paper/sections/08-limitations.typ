#import "/template.typ": *

// OBLIGATIONS
//  - What would make a skeptical reader distrust each headline claim, and
//    what was or was not done about it.

= Limitations <sec-limits>

*The EGGROLL arm is ours, not the paper's.* The authors did not release their EGGROLL-Transformer, so
the baseline here is this repository's EGGROLL, built from the published description: rank-one
perturbations, antithetic pairs, raw loss differences, Adam. Their version is tuned over step size,
momentum, noise scale and fitness shaping, and reaches somewhat lower losses than ours at the same
population (@sec-results). A stronger baseline would narrow every gap reported here. We did what the paper's
protocol asks for within reach of a single GPU, and extended grids that had their best point on the
edge. We could not close the remaining difference to the published EGGROLL numbers, and do not know
which untried detail accounts for it.

*Populations stop at 4,096.* The paper's main ladder runs to 16,384 draws, on eight GPUs, and to 20M tokens.
A DUST update at 4,096 draws takes about two minutes on one RTX 4090; 16,384 would take about eight, and a 10M-token run 610 of
them. The claims that DUST "exceeds backprop" at large populations and that EGGROLL needs $10^3$ to $10^4$ times the population
are therefore tested here by extrapolation from four points, one more than the three we first had and one fewer than the paper's
five. The first three gave an answer a factor of six to seventeen away from the four-point answer, which is the size of
uncertainty to attach to either.

*Few seeds, one at 4,096 and at 10M.* Seeds 42, 43 and 44 for the 1M-token ladder up to 1,024 and the 100K ladder; one seed at 4,096
draws or passes and at 10M tokens. A difference smaller than the seed-to-seed spread, reported beside each number, should not be
read as a ranking. DUST's 0.047 lead over backprop at 4,096 draws is one seed against a backprop spread of 0.027.

*The Adam and Muon cells are thinner than the SGD ladder.* Each (method, optimiser, population) searched the matrix learning rate over three values on one
seed, and Adam's single-seed cells carry about 0.08 nats of nondeterministic noise (@res-opt). DUST at 1,024 draws and above, and EGGROLL at 1,024 passes, take the
256-population rate on one seed. A check at 0.002 and 0.005 for 1,024 draws under Muon did not beat the carried-over 0.003 (exp28); 4,096 draws is unchecked. The DUST paper re-tunes every hyperparameter of DUST at every population, and EGGROLL's step size, momentum and fitness shaping.
Ours searched the matrix rate only, kept EGGROLL's other constants from its Adam tuning, and reconstructed DUST's Adam allocation of draws without the paper's split
across blocks. Any of these could move DUST's Adam cells, which sit 0.17 to 0.37 nats behind its Muon cells.

*The machine was shared for the Adam and Muon runs.* Every run was started on an idle GPU, but from exp19 onward another user's jobs sometimes joined it later. That
slows an update and does not change what it computes, so the losses stand and the seconds per update at 1,024 draws and above are not quoted.

*Tuning is partial.* DUST at 256 draws and above, and backprop at 1M and 10M, use the authors' released
settings rather than a search of ours. DUST at 64 draws and at 100K tokens, and every EGGROLL setting,
were tuned by us on one seed with the paper's grids, judged on validation loss. EGGROLL's 1,024-pass
and 4,096-pass cells reuse the 256-pass winner, and its 10M-token cell reuses its 1M-token winner. The paper tunes every cell.

*Numerics differ from the release.* JAX, not PyTorch; weights are kept in float32 and cast to bfloat16 per
step, where the release stores the two embedding tables in bfloat16; the GPU's matmuls run in TF32 where the
CPU reference ran float32; batches are shuffled by numpy rather than torch. On identical weights the two
estimators agree (@sec-results), and our backprop and DUST land on the published losses, but none of these differences
was isolated one at a time.

*Alignment is measured on one batch of eight sequences.* The cosines are single-batch values with a few noise
draws each. Our 100M-token checkpoint was trained with the paper's 10M-token optimiser settings for 6,100
updates; the paper does not say how it trained its own, so the two checkpoints may sit at different points in
training even though both are labelled "100M tokens". We did not reach their 1B-token checkpoint.

*Cost is measured on one implementation.* Seconds per update are wall-clock on one GPU, for code written for
clarity rather than speed: our DUST reruns attention with a blocked XLA kernel (cuDNN attention would not build
on this machine), and the EGGROLL forward carries its perturbation as an explicit side branch. They indicate
relative cost, and the paper itself says it does not try to make DUST compute-efficient.
