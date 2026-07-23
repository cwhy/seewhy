# DiVeQ-PQ on MNIST — Experiment Report

**Date**: 2026-04-28  
**Experiments completed**: exp1–exp51, exp52 running

---

## Architecture

All experiments use the same MLP autoencoder:

```
Encoder: 784 → 512 → 512 → LATENT_DIM  (GELU activations)
Optional: running-stats BN at bottleneck
DiVeQ-PQ: G groups × K codewords each, D_g = LATENT_DIM / G
Decoder: LATENT_DIM → 512 → 512 → 784  (sigmoid output)
```

**DiVeQ gradient path**: `zq = z + ‖c*−z‖₂ · sg[normalize(v + d̃)]`  
Gradient flows only through the scalar magnitude — no STE, no commitment loss.

---

## Architecture Ceilings (no quantization)

| exp | LATENT | MSE       |
|-----|--------|-----------|
| 20  | 16     | 0.006779  |
| 26  | 32     | 0.003618  |
| 43  | 64     | 0.001810  |

---

## LATENT=16 Results (no BN)

| exp | Config                      | bits | MSE      | % above ceiling | Epochs |
|-----|-----------------------------|------|----------|-----------------|--------|
| 21  | G=8 K=512 D_g=2 DiVeQ       | 72   | 0.006913 | +2.0%           | 500    |
| 17  | G=8 K=512 D_g=2 DiVeQ       | 72   | 0.007041 | +3.9%           | 200    |
| 14  | G=8 K=256 D_g=2 DiVeQ       | 64   | 0.007343 | +8.3%           | 200    |
| 13  | G=4 K=512 D_g=4 DiVeQ       | 36   | 0.013170 | +94%            | 200    |
| 25  | G=8 K=512 D_g=2 STE β=0.25  | 72   | 0.007577 | +11.8%          | 500    |
| 24  | G=8 K=512 D_g=2 STE β=0.25  | 72   | 0.007000 | +3.3%           | 200    |
| 41  | G=8 K=512 D_g=2 DiVeQ+BN    | 72   | 0.007207 | +6.3%           | 200    |

**Best**: exp21 — 2% above ceiling at 72 bits.  
BN hurts LATENT=16 (no lazy groups; BN destroys the natural scale structure).

---

## LATENT=32 Results — Running-stats BN, D_g=2 Pareto

| exp | K    | bits | MSE      | % above ceiling | Epochs | codebook utilization |
|-----|------|------|----------|-----------------|--------|---------------------|
| 35  | 512  | 144  | 0.003716 | +2.8%           | 500    | 100% (all 16 groups) |
| 44  | 512  | 144  | 0.003700 | +2.3%           | 500    | 100% (SIGMA=0.001)   |
| 34  | 512  | 144  | 0.003878 | +7.2%           | 200    | 100%                 |
| 40  | 256  | 128  | 0.003887 | +7.5%           | 500    | 100%                 |
| 38  | 256  | 128  | 0.003993 | +10.4%          | 200    | 100%                 |
| 49  | 128  | 112  | 0.004230 | +16.9%          | 500    | 100%                 |
| 39  | 128  | 112  | 0.004335 | +19.8%          | 200    | 100%                 |
| 50  | 64   | 96   | 0.004790 | +32.4%          | 500    | 100%                 |
| 51  | 32   | 80   | 0.005790 | +60.0%          | 500    | ~99% (30-32/32)      |
| 52  | 16   | 64   | pending  | —               | 500    | —                    |

**Pattern**: Each halving of K adds ~10-30% extra deviation from ceiling. Perfect codebook utilization at all K levels thanks to running-stats BN.

**Key finding**: LATENT=32+BN beats LATENT=16 in absolute MSE at all tested bit levels ≥80 bits:
- 80 bits: L=32+BN 0.00579 vs L=16 at 72 bits 0.006913 (L=32 wins with 8 extra bits)

---

## LATENT=32 Results — Lazy Groups Without BN

| exp | Config                           | bits | MSE      | lazy groups |
|-----|----------------------------------|------|----------|-------------|
| 28  | contiguous G=16 D_g=2, 500ep    | 144  | 0.004223 | 6/16        |
| 30  | interleaved G=16 D_g=2, 200ep   | 144  | 0.004100 | 3/16        |
| 32  | learned proj G=16 D_g=2, 200ep  | 144  | 0.004440 | 6/16        |
| 33  | batch BN (current stats), 200ep  | 144  | 0.007206 | 0/16 (but train-eval mismatch!) |
| 34  | running-stats BN, 200ep          | 144  | 0.003878 | 0/16 ✓     |

---

## LATENT=64 Results (BN required)

AE ceiling: **0.001810** (exp43)

| exp | Config                  | bits | MSE      | % above ceiling | lazy groups |
|-----|-------------------------|------|----------|-----------------|-------------|
| 45  | G=32 D_g=2 K=512, 200ep | 288  | 0.002610 | +44%            | 6/32        |
| 46  | G=32 D_g=2 K=512, 500ep | 288  | 0.002390 | +32%            | 10/32 (worse!) |
| 47  | G=16 D_g=4 K=512, 200ep | 144  | 0.004040 | +123%           | 0 but poor |

**Conclusion**: LATENT=64 is a dead end for MNIST with MLP architecture. More training makes lazy groups worse (10/32 at 500ep vs 6/32 at 200ep). MNIST's intrinsic dimensionality (~10-16 dims) cannot fill 32 groups. Wider groups (D_g=4) avoid laziness but have poor absolute quality.

---

## STE vs DiVeQ

### LATENT=16 (no BN)
| | 200 ep | 500 ep |
|--|--------|--------|
| DiVeQ | 0.007041 | **0.006913** |
| STE β=0.25 | 0.007000 | 0.007577 |

DiVeQ ≈ STE at 200ep. DiVeQ improves at 500ep; STE **degrades** — commitment loss fights reconstruction loss in late training.

### LATENT=32 (running-stats BN)
| | 200 ep | 500 ep |
|--|--------|--------|
| DiVeQ+BN | 0.003878 (exp34) | **0.003716** (exp35) |
| STE+BN β=0.25 | — | 0.003901 (exp42) |

DiVeQ beats STE with BN too. BN ensures unit variance; STE commitment loss pulls z toward c*, fighting BN normalization.

---

## Failure Modes

| Mode | Evidence |
|------|----------|
| D_g=1 (scalar VQ) | NaN or collapse (exp19/22/23) — gradient vanishes |
| HIDDEN=1024 | Codebook collapse to ~130/512 (exp8/10) — tight clusters |
| High LR (1e-3) | Unstable quantization (exp12) |
| Batch BN (current-batch stats) | Train-eval mismatch: 0.003920 train vs 0.007206 eval (exp33) |
| Contiguous groups, LATENT=32 | 6/16 lazy groups (exp28) |
| Interleaved groups, LATENT=32 | 3/16 lazy groups (exp30) — better but not fixed |
| Learned projection, LATENT=32 | 6/16 lazy groups persist (exp32) — gets stuck |
| LATENT=64 any config | Lazy groups persist or worse with more training |

---

## Key Findings Summary

1. **Product quantization with D_g=2 is essential** — single-VQ best ≈0.031; D_g=2 PQ → 0.006913 (4.5×)
2. **D_g=2 always optimal** — larger group dims hurt quantization efficiency
3. **DiVeQ beats STE at long training** — no commitment loss = no conflict with reconstruction
4. **Running-stats BN fixes lazy groups at LATENT≥32** — batch BN causes train-eval mismatch; population BN doesn't
5. **BN hurts LATENT=16** — natural scale structure is informative; BN destroys it
6. **SIGMA is not critical** — 0.001 vs 0.01 gives marginal 0.5% improvement at best
7. **LATENT=64 has a structural ceiling** — MNIST's intrinsic dim (~10-16) can't fill 32 groups; more training makes lazy groups worse
8. **LATENT=32+BN dominates at ≥80 bits** — better absolute MSE than LATENT=16 despite higher % above ceiling

---

## Pareto Front — Absolute MSE

| bits | L=16 (DiVeQ) | L=32+BN (DiVeQ) |
|------|-------------|-----------------|
| 64   | 0.007343 (200ep) | exp52 (pending) |
| 72   | **0.006913** | 0.006650 (exp36, D_g=4) |
| 80   | —            | 0.005790 |
| 96   | —            | 0.004790 |
| 112  | —            | 0.004230 |
| 128  | —            | 0.003887 |
| 144  | —            | **0.003700** |

Note: L=32 at 72 bits uses D_g=4 (G=8), which is suboptimal vs D_g=2.
L=32 at 64 bits uses D_g=2 (G=16, K=16, 4 bits/group).
