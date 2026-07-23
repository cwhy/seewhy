# Experiment Workflow — UTM

Replication of "Universal Transformers Need Memory" (arxiv 2604.21999): a single
recurrent transformer block with ACT and learnable memory tokens solving Sudoku-Extreme.

---

## Directory Layout

```
projects/utm/
├── workflow.md             # This file
├── experiments1.py         # Baseline: D=512, K=18, T=8 mem, no ponder penalty
├── results.jsonl           # Append-only results log
├── params_expN.pkl         # EMA model params after each run (gitignored)
├── history_expN.pkl        # Per-epoch training history (gitignored)
├── logs/                   # Experiment logs (gitignored)
├── lib/
│   └── __init__.py
└── scripts/
    ├── run_experiments.py
    ├── poll_result.py
    └── tmp/
```

---

## Paper Architecture (arxiv 2604.21999)

| Component | Value |
|-----------|-------|
| Hidden dim D | 512 |
| Heads H | 8, head_dim=64 |
| FFN width | 8/3 × D (SwiGLU) |
| Max ponder steps K | 18 |
| Memory tokens T | 8 (optimal; T=0 always fails) |
| Total params | ~3.2M |
| Norm | DerfNorm (erf-based; we use RMSNorm) |
| Attention | RoPE + QK-norm, full bidirectional |
| ACT halt bias | −3 ("deep-start") |

**Training:** AdamW lr=3e-4, cosine decay, batch=256, 4 epochs, EMA=0.999, wd=0.01.

**Dataset:** `sapientinc/sudoku-extreme` — 3.83M train / 423K test, 81-cell puzzles,
17-24 givens. Encoder-style: predict all 81 cells simultaneously. Metric: exact
accuracy (all 81 cells must be correct).

**Key finding:** Memory tokens are necessary — T=0 never achieves >0% exact accuracy.
Optimal T=8–32. ACT lowers variance vs fixed depth. Lambda warmup (20k steps to
λ=0.001) achieves same accuracy with 34% fewer ponder steps.

---

## Experiment Plan

| Exp | Change | Expected |
|-----|--------|----------|
| exp1 | Baseline T=8, λ=0, n_tr=200K | ~5-20% (subset, proof of concept) |
| exp2 | Full dataset n_tr=3.83M, 4 epochs | ~57% (matching paper) |
| exp3 | T=0 ablation | ~0% (memory necessity) |
| exp4 | λ=0.001 ponder penalty, 20k warmup | ~57% with fewer steps |

---

## Invariant Rules

1. **Always `uv run python`** — never raw `python`.
2. **Always use `run_experiments.py`** to launch experiments.
3. **Throwaway scripts go in `scripts/tmp/`**.
4. **No bash polling loops**.

---

## Running Experiments

```bash
uv run python projects/utm/scripts/run_experiments.py --bg exp1
uv run python projects/utm/scripts/poll_result.py --all
tail -20 projects/utm/logs/exp1.log
```

---

## Dataset

Loaded via `shared_lib.datasets.load_sudoku_extreme(n_tr, n_tst)`.
Returns `SudokuDataset` with `X_train` (int32, 0=empty, 1-9=digit) and
`y_train` (int32, 1-9=solution). Cached to `~/.cache/huggingface/processed_datasets/`.

RoPE positions: grid cells 0–80, memory tokens 81–88 (separate indices per paper).
