# Chain Distillation: DAE Conv-2 → MLP-2 → Sub-student

**Hypothesis**: Can a sub-student learn better from an already-distilled MLP-2
than directly from the original DAE Conv-2 teacher?

**Chain**:
- L1 teacher: DAE Conv-2 (frozen, 98.0% probe baseline)
- L2 student: MLP-2 trained on L1 targets (best pair from heatmap)
- L2 result: probe=98.6%  knn=97.3%
- L3 sub-students: all 9 archs trained on L2 (MLP-2) targets

**Baseline**: Direct DAE Conv-2 → sub-student (from `results_distill_heatmap_dae.jsonl`).

---

## Results

![Bar chart](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_chain_bar.png)

![Delta chart](https://media.tanh.xyz/seewhy/26-04-07/ssl_distill_chain_delta.png)

---

## Comparison Table

| Sub-student | Direct probe | Chain probe | Δ probe | Direct KNN | Chain KNN | Δ KNN |
|-------------|-------------|-------------|---------|------------|-----------|-------|
| MLP | 98.2% | 97.9% | -0.3 pp | 97.5% | 97.4% | -0.1 pp |
| MLP-2 | 98.6% | 98.3% | -0.3 pp | 97.3% | 96.9% | -0.4 pp |
| Conv | 97.5% | 97.1% | -0.4 pp | 97.4% | 96.4% | -0.9 pp |
| Conv-2 | 97.9% | 97.8% | -0.1 pp | 97.7% | 97.3% | -0.5 pp |
| ViT | 97.9% | 97.7% | -0.1 pp | 97.5% | 97.2% | -0.2 pp |
| ViT-patch | 98.5% | 98.1% | -0.3 pp | 97.9% | 97.8% | -0.1 pp |
| Gram-dual | 97.6% | 97.8% | +0.2 pp | 95.5% | 96.3% | +0.8 pp |
| Gram-single | 97.2% | 97.5% | +0.3 pp | 96.1% | 96.3% | +0.2 pp |
| Gram-GLU | 98.2% | 98.2% | +0.1 pp | 97.2% | 97.3% | +0.0 pp |

## Key Findings

**Chain wins on 3/9 sub-students** (probe accuracy):

- Gram-single: +0.3 pp
- Gram-dual: +0.2 pp
- Gram-GLU: +0.1 pp

**Direct wins on**:
- Conv: -0.4 pp
- ViT-patch: -0.3 pp
- MLP-2: -0.3 pp
- MLP: -0.3 pp
- ViT: -0.1 pp
- Conv-2: -0.1 pp

**Average Δ probe (chain − direct)**: -0.12 pp

### Interpretation

The chain path is roughly **neutral** — the MLP-2 intermediate neither helps nor hurts significantly. Information passes through the chain without major loss or gain.
