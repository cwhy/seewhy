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
├── logs/                   # Experiment and runner logs (gitignored)
├── lib/
│   ├── __init__.py
│   └── viz.py              # Visualization (adapted for IMLE — no encoder)
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs  ← always use this
    ├── poll_result.py      # Wait for / display results
    ├── gen_viz_clustering.py  # Clustering analysis plots (takes exp_name arg)
    ├── gen_viz_expN.py     # Per-experiment post-training visualizations
    └── tmp/                # Throwaway / generator scripts (not committed)
```

---

## Invariant Rules

These apply to every session without exception:

1. **Always `uv run python`** — never raw `python` or `python3`. The `.venv`
   has CUDA-specific packages; raw python may resolve a different interpreter.

2. **Always use `run_experiments.py`** to launch experiments — never call
   `uv run python experimentsN.py` directly. The runner handles log paths,
   GPU assignment, heartbeat, results table, and optional auto-viz.

3. **Throwaway scripts go in `scripts/tmp/`** — never `python -c "..."` inline.
   Write to `scripts/tmp/<name>.py`, then run with `uv run python`. This keeps
   generator scripts visible and reproducible.

4. **No bash polling loops** — to check progress, read the log file once or
   write a Python script. Never `while true; do ...; sleep N; done`.

---

## Running Experiments

```bash
# Fire and forget — detach to background, return immediately
uv run python projects/imle-gram/scripts/run_experiments.py --bg exp30

# With auto-viz after completion
uv run python projects/imle-gram/scripts/run_experiments.py --bg --viz scripts/gen_viz_clustering.py exp30

# Sequential blocking (useful when chaining multiple)
uv run python projects/imle-gram/scripts/run_experiments.py exp29 exp30

# Parallel across GPUs
uv run python projects/imle-gram/scripts/run_experiments.py --parallel exp29 exp30

# Pin to a specific GPU
uv run python projects/imle-gram/scripts/run_experiments.py --gpu 1 exp30
```

Logs always go to `projects/imle-gram/logs/{exp_name}.log`.

---

## Monitoring

```bash
# Check progress of a running experiment
tail -20 projects/imle-gram/logs/exp30.log

# Poll until result appears in results.jsonl
uv run python projects/imle-gram/scripts/poll_result.py exp30
uv run python projects/imle-gram/scripts/poll_result.py --all
```

---

## Visualization

### During training

`lib/viz.py` functions are called from `experiments*.py` using in-memory
`params` and `history` — nothing is read from disk.

### Post-training

```bash
uv run python projects/imle-gram/scripts/gen_viz_clustering.py exp30
```

Reads `params_expN.pkl` and `history_expN.pkl` from disk and regenerates
all plots without re-running the experiment.

Save these at the end of `__main__` to enable this:

```python
with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({k: np.array(v) for k, v in params.items()}, f)

with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump(history, f)
```

Both files are gitignored. `results.jsonl` is the only committed output.

---

## Diagnostic Scripts (`scripts/tmp/`)

One-off helpers live in `scripts/tmp/` and are not committed. Examples:

| Script | Purpose |
|--------|---------|
| `check_embed_dist.py exp28` | L2 distances between codebook embeddings |
| `check_cosine_sim.py exp27` | Off-diagonal cosine similarities |
| `check_proto_dist.py exp29` | Pixel-space L2 between decoded prototypes |
| `mk_expN.py` | Generator that patches a parent experiment into a new one |

Run all with `uv run python projects/imle-gram/scripts/tmp/<name>.py [exp_name]`.

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
