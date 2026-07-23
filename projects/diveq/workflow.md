# Experiment Workflow — DiVeQ

VQ-VAE on MNIST where the quantization layer uses DiVeQ (Distributional Vector
Quantization) in place of the standard VQ + straight-through estimator.

---

## Directory Layout

```
projects/diveq/
├── workflow.md             # This file
├── experiments1.py         # One file per experiment; numbered sequentially
├── results.jsonl           # Append-only results log — one JSON object per line
├── params_expN.pkl         # Saved model params after each run (gitignored)
├── history_expN.pkl        # Per-epoch training history (gitignored)
├── logs/                   # Experiment and runner logs (gitignored)
├── lib/
│   └── __init__.py
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs  ← always use this
    ├── poll_result.py      # Wait for / display results
    └── tmp/                # Throwaway / generator scripts (not committed)
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `recon_mse` | Reconstruction MSE in [0,1] pixel space (primary metric) |
| `qe` | Quantization error: MSE(z, c*) — measures codebook alignment |
| `codebook_use` | Number of unique codewords used on the test set (max = K) |

---

## Invariant Rules

1. **Always `uv run python`** — never raw `python` or `python3`.
2. **Always use `run_experiments.py`** to launch experiments.
3. **Throwaway scripts go in `scripts/tmp/`** — never `python -c "..."` inline.
4. **No bash polling loops** — read the log once or write a Python script.

---

## Running Experiments

```bash
# Fire and forget
uv run python projects/diveq/scripts/run_experiments.py --bg exp1

# Sequential blocking
uv run python projects/diveq/scripts/run_experiments.py exp1 exp2

# Check progress
tail -20 projects/diveq/logs/exp1.log

# Poll until result appears
uv run python projects/diveq/scripts/poll_result.py exp1
uv run python projects/diveq/scripts/poll_result.py --all
```

---

## DiVeQ Forward Pass (training)

```
z  = encode(x)                            # (B, D)
i* = argmin_j ‖z − c_j‖₂                # nearest codeword index
c* = codebook[i*]                         # (B, D)
d̃  = sg[c*] − sg[z]                      # direction toward c* (constant)
vd = N(0, σ²I) + d̃                       # biased noise
zq = z + ‖c* − z‖₂ · sg[vd / ‖vd‖₂]    # quantized; grad via magnitude only
x̂  = decode(zq)
loss = MSE(x̂, x)                          # single loss, no commitment term
```

At inference: `zq = c*` (hard quantization, no noise).

---

## Experiment Log

| exp | LATENT | K | sigma | recon_mse | codebook_use | notes |
|-----|--------|---|-------|-----------|-------------|-------|
| exp1 | 16 | 128 | 0.01 | — | — | baseline |
