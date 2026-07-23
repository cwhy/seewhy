# Same-Architecture Chain Distillation (10 Hops)

**Setup**: For each architecture, train 10 successive distillation hops where each
student learns from the previous hop's params as a frozen teacher (same arch).

- **Hop 0**: Random init — evaluated directly, no training
- **Hops 1–10**: Each distills from the previous hop's frozen params

Architectures: MLP-2, ViT-patch, Gram-dual, Gram-single, Gram-GLU
(MLP, ViT, Conv, Conv-2 excluded — MLP/ViT weaker than counterparts;
Conv/Conv-2 stuck at ~21% because random conv features are near-chance,
making the chain unable to bootstrap from noise)

---

## Results

![Line plot](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_chain_same_line.png)

![Bar chart](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_chain_same_convergence.png)

![Delta chart](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_chain_same_delta.png)

---

## Full Results Table (Probe Accuracy)

| Architecture | H0 (rand) | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 | Δ H1→H10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP-2 | 91.6% | 97.7% | 98.0% | **98.1%** | 97.8% | 98.0% | 97.8% | 97.8% | 97.7% | 97.5% | 97.5% | -0.2 pp |
| ViT-patch | 77.5% | 95.1% | 97.0% | 97.3% | 97.5% | 97.8% | 97.8% | 97.8% | 97.9% | **97.9%** | 97.8% | +2.7 pp |
| Gram-dual | 89.6% | 97.2% | **97.5%** | 97.3% | 97.3% | 97.0% | 97.2% | 96.9% | 96.9% | 97.2% | 97.2% | -0.0 pp |
| Gram-single | 89.7% | 96.5% | **96.8%** | 96.8% | 96.5% | 96.8% | 96.7% | 96.6% | 96.5% | 96.7% | 96.5% | -0.0 pp |
| Gram-GLU | 86.5% | 97.0% | 97.5% | 97.6% | 97.6% | **97.7%** | 97.7% | 97.7% | 97.5% | 97.5% | 97.6% | +0.6 pp |

*(bold = peak value; Δ H1→H10 = whether chaining beyond hop 1 helps)*

## KNN Results

| Architecture | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP-2 | 96.5% | 97.0% | 96.6% | 96.5% | 96.2% | 96.2% | 95.8% | 95.7% | 95.4% | 95.2% | 95.2% |
| ViT-patch | 75.7% | 92.0% | 94.7% | 95.5% | 96.0% | 96.4% | 96.6% | 96.6% | 96.4% | 96.6% | 96.7% |
| Gram-dual | 84.0% | 95.0% | 96.1% | 96.2% | 96.2% | 96.2% | 96.2% | 96.2% | 96.1% | 96.2% | 96.0% |
| Gram-single | 83.4% | 94.8% | 96.2% | 96.2% | 96.2% | 96.2% | 96.0% | 95.9% | 96.0% | 95.8% | 96.0% |
| Gram-GLU | 83.7% | 94.8% | 96.0% | 96.3% | 96.4% | 96.5% | 96.4% | 96.4% | 96.2% | 96.2% | 96.3% |

## Key Findings

**Does extended chaining (hop 2–10) improve over hop 1?** 2/5 architectures gain at hop 10:

- ViT-patch: +2.7 pp
- Gram-GLU: +0.6 pp

Hurt or flat by hop 10:
- MLP-2: -0.2 pp
- Gram-single: -0.0 pp
- Gram-dual: -0.0 pp

**Peak hop per architecture**:
- MLP-2: hop 3 (98.1%)
- ViT-patch: hop 9 (97.9%)
- Gram-dual: hop 2 (97.5%)
- Gram-single: hop 2 (96.8%)
- Gram-GLU: hop 5 (97.7%)

**Biggest jump from random → hop 1**:
- ViT-patch: +17.6 pp  (random=77.5% → hop1=95.1%)
- Gram-GLU: +10.5 pp  (random=86.5% → hop1=97.0%)
- Gram-dual: +7.6 pp  (random=89.6% → hop1=97.2%)
