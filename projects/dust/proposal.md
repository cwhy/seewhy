# DUST vs EGGROLL — Proposal

## Premise

[DUST](https://qlabs.sh/research/dust) (Dahal, Mandal, Gülbahar, Vegesna; Q Labs, Oct 2026)
trains a transformer language model **without a backward pass**. It perturbs *activations*,
independently at every token, so that one forward pass evaluates one population member per
token. The paper's claim: from 1M training tokens up it is "on the order of 10³ to 10⁴ times
more efficient" than weight-space ES (EGGROLL) and approaches — at 1M tokens, slightly beats —
backprop.

This repo already has an EGGROLL arm ([local-learning](../local-learning/concepts.md)),
audited and found sound. The question here: **does DUST's advantage over that EGGROLL
reproduce on the paper's own task, at the scale two RTX 4090s allow?**

The authors released a minimal PyTorch DUST (`github.com/qlabs-eng/dust`) and a backprop
baseline. They did **not** release their EGGROLL-Transformer, so the EGGROLL arm is ours, in
the setting of `local-learning/lib/es.py`: rank-1 antithetic perturbations, raw loss
differences, Adam.

## Claims under test

Published numbers are test loss at the validation-selected checkpoint, three-seed means
(paper Table 1). Our unigram floor — the loss of a model that knows only token frequencies —
is 6.994 on this test set.

| # | Claim (paper) | Published | What would confirm it here |
|---|---|---|---|
| C1 | The network, data and SGD protocol are right | backprop 5.959 (1M), 4.989 (10M) | our backprop within ~0.05 |
| C2 | DUST beats weight-space ES at equal population | DUST 64 draws 6.290 vs EGGROLL 64 passes 7.428 | DUST < EGGROLL at every matched population |
| C3 | 64 DUST draws beat 16,384 EGGROLL passes | 6.290 vs 6.708 (1M) | the same ordering at our largest EGGROLL population |
| C4 | DUST closes on backprop with population and passes it at 1M | 6.067 → 5.934 → 5.911 vs 5.959 | DUST ≤ backprop by ~1k draws |
| C5 | DUST's estimate aligns with backprop's gradient along `c_max/√(1+c/K)`; EGGROLL's stays < 0.05 at 128k passes | Table 8 fits at 100M tokens | our cosines near the fitted law; EGGROLL ≈ `√(P/d)` |
| C6 | EGGROLL's population cost to match DUST's smallest is ~10³–10⁴× | Appendix D, Table 2 | extrapolation from our EGGROLL ladder |
| C7 | *(stretch)* larger models are more population-efficient | Fig. 4, 10M tokens | the loss-vs-size curve at 64 draws is flat or falling |

## Stages

1. **Port and validate.** JAX port of model + DUST; EGGROLL on the same model. Parity against
   the authors' PyTorch on identical weights; backprop reproduces C1; alignment (C5) as the
   estimator test that needs no training.
2. **The 1M-token ladder** (C2–C4, C6): DUST and EGGROLL at 64 / 256 / 1,024 (and 4,096 if
   time allows), three seeds, plus backprop. EGGROLL tuned per population on seed 42.
3. **Other budgets.** 100K tokens (cheap, 6 updates) and 10M tokens (610 updates) at the
   smaller populations.
4. **Model size** (C7), if the budget remains.

## Deviations from the source

- JAX, not PyTorch. Weights are kept in f32 and cast to bf16 per step; the reference stores
  the embedding tables in bf16.
- The authors' DUST optimiser settings are used as released, not re-tuned, at 256 draws and up;
  64 draws and 100K tokens are tuned on the paper's own 12-point grid, seed 42 then three seeds.
- **EGGROLL is the `local-learning` setting** (Adam, raw differences, rank 1), not the paper's
  EGGROLL-Transformer, which is tuned over step size, momentum, noise scale and fitness shaping
  and is not released. The paper reports EGGROLL gains < 0.01 from SGD→Adam, so the choice of
  optimiser should not move the ladder; the fitness-shaping and Adam-β ablations check this.
- Populations stop at 1k–4k draws for training; 16k and 20M-token cells cost days on one GPU
  (the authors used 8 GPUs). Alignment is measured up to 16k.
- Seeds: 42, 43, 44, as in the release; `seed` here seeds both initialisation and data order.
