# DAE-Seeded Same-Architecture Chain Distillation (10 Hops)

**Setup**: Same as random-seeded chain, but hop 0 starts from a DAE-trained teacher
instead of random init. Each hop's student distills from the previous hop's frozen params.

- **Hop 0**: DAE-trained teacher — evaluated directly (no training in this script)
- **Hops 1–10**: Each distills from the previous hop's frozen params (same arch)

Architectures: MLP-2, ViT-patch, Gram-dual, Gram-single, Gram-GLU

---

## Results

![Line plot](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_chain_dae_line.png)
*(solid lines = DAE-seeded, dashed lines = random-seeded for comparison)*

![Delta vs DAE teacher](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_chain_dae_delta.png)

![DAE vs Random comparison](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_chain_dae_vs_rand.png)

---

## Full Results Table (Probe Accuracy %)

| Architecture | H0 (DAE) | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 | Δ H0→H10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP-2 | 97.4% | 97.9% | **98.0%** | 98.0% | 97.9% | 97.9% | 97.7% | 97.7% | 97.8% | 97.6% | 97.5% | +0.2 pp |
| ViT-patch | 96.7% | 97.5% | 97.9% | 97.9% | 97.9% | 97.9% | 97.9% | 97.9% | **98.1%** | 97.9% | 98.0% | +1.2 pp |
| Gram-dual | 97.1% | **97.6%** | 97.3% | 97.2% | 97.2% | 97.0% | 97.2% | 97.2% | 97.0% | 97.2% | 97.5% | +0.4 pp |
| Gram-single | 96.3% | **97.0%** | 96.9% | 96.8% | 96.9% | 97.0% | 96.7% | 96.7% | 96.2% | 96.9% | 96.6% | +0.3 pp |
| Gram-GLU | 97.3% | **97.3%** | 97.0% | 96.8% | 96.9% | 96.6% | 96.4% | 96.4% | 96.2% | 96.3% | 96.1% | -1.2 pp |

*(bold = peak value; Δ H0→H10 = net change from DAE teacher to final student)*

## KNN Results

| Architecture | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP-2 | 96.6% | 96.5% | 96.2% | 95.8% | 95.5% | 95.4% | 95.3% | 95.2% | 95.4% | 95.1% | 95.2% |
| ViT-patch | 97.4% | 97.5% | 97.6% | 97.6% | 97.7% | 97.7% | 97.6% | 97.6% | 97.5% | 97.4% | 97.5% |
| Gram-dual | 95.7% | 96.0% | 96.1% | 96.1% | 96.1% | 96.1% | 96.1% | 96.2% | 96.1% | 96.0% | 96.2% |
| Gram-single | 95.8% | 96.2% | 96.0% | 96.2% | 96.2% | 96.1% | 96.0% | 96.0% | 95.9% | 95.8% | 96.0% |
| Gram-GLU | 96.8% | 96.8% | 96.6% | 96.5% | 96.2% | 96.0% | 95.6% | 95.5% | 95.1% | 95.2% | 95.0% |

## DAE-seeded vs Random-seeded Comparison (Probe %)

| Architecture | DAE H0 | Rand H0 | DAE H1 | Rand H1 | DAE H10 | Rand H10 |
|---|---|---|---|---|---|---|
| MLP-2 | 97.4% | 91.6% | 97.9% | 97.7% | 97.5% | 97.5% |
| ViT-patch | 96.7% | 77.5% | 97.5% | 95.1% | 98.0% | 97.8% |
| Gram-dual | 97.1% | 89.6% | 97.6% | 97.2% | 97.5% | 97.2% |
| Gram-single | 96.3% | 89.7% | 97.0% | 96.5% | 96.6% | 96.5% |
| Gram-GLU | 97.3% | 86.5% | 97.3% | 97.0% | 96.1% | 97.6% |

## Key Findings

**Does the chain improve on the DAE teacher (hop 0)?** 4/5 architectures gain at hop 10:

- ViT-patch: +1.2 pp  (DAE=96.7% → hop10=98.0%)
- Gram-dual: +0.4 pp  (DAE=97.1% → hop10=97.5%)
- Gram-single: +0.3 pp  (DAE=96.3% → hop10=96.6%)
- MLP-2: +0.2 pp  (DAE=97.4% → hop10=97.5%)

Worse than DAE teacher at hop 10:
- Gram-GLU: -1.2 pp  (DAE=97.3% → hop10=96.1%)

**DAE-seeded vs Random-seeded at hop 1** (how much does a better start help?):
- MLP-2: DAE 97.9% vs Random 97.7% (+0.2 pp)
- ViT-patch: DAE 97.5% vs Random 95.1% (+2.4 pp)
- Gram-dual: DAE 97.6% vs Random 97.2% (+0.4 pp)
- Gram-single: DAE 97.0% vs Random 96.5% (+0.5 pp)
- Gram-GLU: DAE 97.3% vs Random 97.0% (+0.3 pp)

**Peak hop per architecture (DAE-seeded)**:
- MLP-2: hop 2 (98.0%)
- ViT-patch: hop 8 (98.1%)
- Gram-dual: hop 1 (97.6%)
- Gram-single: hop 1 (97.0%)
- Gram-GLU: hop 1 (97.3%)
