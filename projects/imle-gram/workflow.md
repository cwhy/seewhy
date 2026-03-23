# Experiment Workflow — imle-gram

This project follows the same conventions as vae-gram. This file documents
imle-gram-specific details. For the full generic workflow (lax.scan, GPU
isolation, JSONL logging, etc.) see [vae-gram/workflow.md](../vae-gram/workflow.md).

For model-specific concepts (IMLE algorithm, decoder architecture, metrics)
see [concepts.md](concepts.md).

---

## Directory Layout

```
projects/imle-gram/
├── workflow.md             # This file
├── concepts.md             # IMLE algorithm, decoder architecture, metrics
├── experiments1.py         # One file per experiment
├── results.jsonl           # Append-only results log
├── params_expN.pkl         # Saved decoder params (gitignored)
├── lib/
│   ├── __init__.py
│   └── viz.py              # Visualization (adapted for IMLE — no encoder)
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs
    └── poll_result.py      # Wait for / display results
```

---

## Running Experiments

```bash
# Single experiment
uv run python projects/imle-gram/experiments1.py

# Sequential (recommended — avoids XLA contention)
uv run python projects/imle-gram/scripts/run_experiments.py exp1 exp2

# Parallel across GPUs
uv run python projects/imle-gram/scripts/run_experiments.py --parallel exp1 exp2
```

---

## Monitoring

```bash
uv run python projects/imle-gram/scripts/poll_result.py exp1
uv run python projects/imle-gram/scripts/poll_result.py --all
```

---

## IMLE-Specific Workflow Notes

### Epoch structure

Each epoch has two phases:
1. **Sampling phase** (no grad): sample N_SAMPLES codes, decode to pixel images.
   This is done once per epoch and cached in numpy.
2. **Training phase** (with grad): for each mini-batch, find nearest neighbor
   in the cached set, re-decode matched codes, compute NLL loss, step.

The sampling phase cost scales with N_SAMPLES. The training phase cost is
the same as the VAE (O(BATCH_SIZE) decodes per step).

### N_SAMPLES sensitivity

N_SAMPLES is the most important IMLE hyperparameter. Too small → poor coverage,
decoder learns a subset of modes. Too large → slow epoch time. Monitor
`match_loss` — if it plateaus high, increase N_SAMPLES.

### No FREE_BITS, no KL warmup

IMLE has no encoder and no KL term. There is no posterior collapse risk.
The `beta` annealing schedule from vae-gram is not needed here.

### Eval metric

`bin_acc` depends on `M_EVAL` (the pool size used for eval matching). Always
report `M_EVAL` alongside `bin_acc` for results to be comparable.

---

## Experiment File Structure

Same structure as vae-gram, minus encoder-related functions:

```
module docstring    — algorithm description, what changed, usage line
hyperparameters     — UPPER_SNAKE_CASE, including N_SAMPLES and M_EVAL
precomputed consts  — ROWS, COLS, BIN_CENTERS
utilities           — quantize, n_params, append_result
init_params()       — decoder params only
decode()            — z → logits (B, 784, N_VAL_BINS)
sample_and_decode() — sample Z, decode to soft-mean numpy (no grad)
find_nearest()      — L2 nearest-neighbor matching (numpy)
imle_loss()         — NLL of real images under matched codes
make_train_step()   — returns JIT-compiled step fn
eval_metrics()      — match_loss + bin_acc on eval set
train()             — epoch loop: sample → match → train step
__main__            — load data → init → train → visualize → append_result
```
