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
├── history_expN.pkl        # Per-epoch training history (gitignored)
├── label_probs_expN.npy    # Per-cluster label distributions (gitignored)
├── lib/
│   ├── __init__.py
│   └── viz.py              # Visualization (adapted for IMLE — no encoder)
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs
    ├── poll_result.py      # Wait for / display results
    ├── gen_viz_clustering.py  # Clustering analysis plots (takes exp_name arg)
    └── gen_viz_expN.py     # Per-experiment post-training visualizations
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

## Visualization

### During training (`lib/viz.py`)

Called directly from `experiments*.py` while the process is running, using
in-memory `params` and `history`. Nothing is read from disk.

| Function | Output | When |
|---|---|---|
| `save_reconstruction_grid` | Real vs nearest decoded sample (Gaussian NN match) | Every `VIZ_EVERY` epochs + final |
| `save_sample_grid` | Grid of `z ~ N(0,I)` samples | End of training |
| `save_learning_curves` | Match loss + bin accuracy over epochs | End of training |

Note: `save_reconstruction_grid` uses a **Gaussian prior** regardless of the
experiment's actual prior. For clustering experiments the prototype
reconstructions (via `decode_prototypes`) are more meaningful.

### Post-training (`scripts/`)

Run manually after training. Reads params and history from disk — no need to
re-run the experiment.

```bash
# Clustering experiments (exp17b, exp17c, ...)
uv run python projects/imle-gram/scripts/gen_viz_clustering.py exp17b

# Per-experiment scripts for earlier exps
uv run python projects/imle-gram/scripts/gen_viz_exp17.py
```

`gen_viz_clustering.py` generates 4 plots when both `history_expN.pkl` and
`params_expN.pkl` are present:
1. Cumulative cluster assignment line plot (coloured by majority label)
2. Label accuracy curves (hard majority-vote vs Bayes)
3. Per-cluster label distribution grid + inverted reconstruction overlay
4. Prototype reconstruction grid

If `params_expN.pkl` is missing, plots 3 and 4 are skipped (no re-run needed
for the curves).

### What needs to be saved for post-training viz

Experiments that want full post-training visualization must save at the end
of `__main__`:

```python
# params (for reconstructions)
with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({k: np.array(v) for k, v in params.items()}, f)

# history (for curves — only if tracking per-epoch stats)
with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump(history, f)

# label_probs (for clustering label distribution plot)
np.save(Path(__file__).parent / f"label_probs_{EXP_NAME}.npy", label_probs)
```

All three files are gitignored. `results.jsonl` is the only persisted
experiment output that is committed.

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
