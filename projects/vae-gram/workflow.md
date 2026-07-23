# Experiment Workflow

This document describes the conventions for running, monitoring, and analysing
experiments in this project. It is intentionally architecture-agnostic — the
same patterns apply to any JAX-based ML experiment project.

For model-specific concepts (architecture, loss, hyperparameters) see
[concepts.md](concepts.md).

---

## Directory Layout

```
project-name/
├── workflow.md          # This file — how to work in the project
├── concepts.md          # Model architecture, loss, key findings
├── experiments1.py      # One file per experiment; numbered sequentially
├── experiments2.py
├── report_1.md          # Written findings after a batch of experiments
├── results.jsonl        # Append-only results log — one JSON object per line
├── params_expN.pkl      # Saved params after each run (never commit large files)
├── lib/                 # Shared code imported by experiments
│   ├── __init__.py
│   └── viz.py           # Shared visualization utilities
└── scripts/             # Standalone utilities (monitoring, analysis, replotting)
    ├── run_experiments.py  # Launch and manage experiment runs
    ├── poll_result.py      # Wait for / display results
    └── replot.py           # Reload saved params and regenerate plots
```

---

## Running Experiments

Always use `uv run` — the system Python does not have the project dependencies:

```bash
uv run python projects/<project-name>/experiments1.py
```

**Never run `uv sync` or `uv run` casually** — it can silently upgrade packages
and break the GPU setup. If GPU stops working after a uv operation:
```bash
uv pip install --force-reinstall nvidia-cudnn-cu13
```

### Single experiment

```bash
uv run python projects/<project-name>/experiments24.py
```

Logs go to the terminal. For background runs, redirect to `/tmp`:

```bash
uv run python projects/<project-name>/experiments24.py > /tmp/exp24.log 2>&1 &
```

### Multiple experiments — use `run_experiments.py`

```bash
# Sequential (safe, recommended — no XLA contention)
uv run python projects/<project-name>/scripts/run_experiments.py exp24 exp25

# Parallel — one experiment per GPU
uv run python projects/<project-name>/scripts/run_experiments.py --parallel exp24 exp25

# Pin to a specific GPU
uv run python projects/<project-name>/scripts/run_experiments.py --gpu 1 exp24
```

**Never use `sleep N` before a second launch as a workaround** — use
`run_experiments.py` sequential mode instead. Sleeping is fragile (wrong
duration) and invisible to future readers.

---

## Monitoring

Use `poll_result.py`. **Never write bash `while ! tail -1 ... | python3 -c "..."`
one-liners** — they are unreadable, fragile, and not reusable.

```bash
# Block until exp24 appears in results.jsonl, then print its metrics
uv run python projects/<project-name>/scripts/poll_result.py exp24

# Print a table of all completed experiments
uv run python projects/<project-name>/scripts/poll_result.py --all

# Custom poll interval
uv run python projects/<project-name>/scripts/poll_result.py exp24 --interval 5
```

---

## JAX Performance

### Use `lax.scan` for the batch loop

Replace the Python `for b in range(num_batches)` loop with `jax.lax.scan`.
This eliminates thousands of Python→XLA dispatch calls per run and lets XLA
see the full epoch as one fused computation.

```python
def make_epoch_fn(optimizer):
    def scan_body(carry, inputs):
        params, opt_state, beta = carry
        batch, z_key = inputs
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, z_key, beta)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state, beta), (loss, *aux)

    def epoch_fn(params, opt_state, Xs_batched, epoch_key, beta):
        num_batches = Xs_batched.shape[0]
        z_keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (params, opt_state, _), stats = jax.lax.scan(
            scan_body, (params, opt_state, beta), (Xs_batched, z_keys)
        )
        return params, opt_state, *[s.mean() for s in stats]

    return jax.jit(epoch_fn)
```

Pre-batch data before calling:
```python
Xs_batched = Xs[:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, -1)
```

### GPU isolation — never launch two JAX experiments on the same GPU simultaneously

XLA compilations from two processes compete for GPU memory and can OOM.
Use `run_experiments.py` sequential mode, or `--parallel` which sets
`CUDA_VISIBLE_DEVICES` to pin each experiment to a different GPU.

### Mixed precision (bf16) — only useful for large models

`bfloat16` only speeds up matmuls when they are large enough to engage tensor
cores (roughly 1024+ dimension). For small models (hidden dim ~256), the
cast overhead dominates and training is **slower**. Do not pursue bf16 unless
the model has large matmuls as its primary bottleneck.

### First epoch is slow — JIT compilation

The first training epoch includes XLA compilation time. Epoch 0 may be 10–20×
slower than subsequent epochs. Do not judge speed from epoch 0.

---

## Experiment File Structure

Every experiment file follows the same layout:

```
module docstring    — hypothesis, what changed, usage line
hyperparameters     — all UPPER_SNAKE_CASE constants at module top
precomputed consts  — JAX arrays that don't change (ROWS, COLS, BIN_CENTERS, …)
utilities           — layer_norm, quantize, n_params, append_result
init_params()
encode() / forward()
decode() / loss_fn()
make_train_step() or make_epoch_fn()   — returns JIT-compiled function
eval helpers        — @jax.jit accuracy/metric functions
train()             — loop with mid-run viz; returns (params, history, elapsed)
__main__            — load data → init → train → visualize → append_result
```

Hyperparameters go at module level in `UPPER_SNAKE_CASE` so they are visible
at a glance and easy to diff between experiments.

---

## Results Logging

Append one JSON object per experiment to `results.jsonl`:

```python
JSONL = Path(__file__).parent / "results.jsonl"

def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
```

Every row must include: all hyperparameters, `n_params`, `time_s`, `epochs`,
final metric values, and the full per-epoch curve lists (`recon_curve`,
`kl_curve`, etc.). Curves are cheap to store and invaluable for re-plotting
without re-running.

---

## Saving and Reloading Params

Always save params at the end of a run:

```python
with open(Path(__file__).parent / "params_exp24.pkl", "wb") as f:
    pickle.dump({k: np.array(v) for k, v in params.items()}, f)
```

To reload in a script:

```python
with open("projects/<project-name>/params_exp24.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}
```

---

## Visualization

Shared visualization functions live in `lib/viz.py`. **Do not inline
visualization code in experiment files** — it diverges across experiments
and is hard to keep consistent.

```python
from lib.viz import save_reconstruction_grid, save_sample_grid, save_learning_curves
```

Format guide:
- **SVG** for charts, learning curves, PCA, anything vector-based
- **PNG** (dpi=150) for pixel images — reconstructions, heatmaps, sample grids

Always log the returned URL:
```python
url = save_matplotlib_figure("exp24_learning_curves", fig, format="svg")
logging.info(f"  learning curves → {url}")
```

---

## Scripts Directory

Put standalone utilities in `scripts/` rather than inlining them as one-off
code in experiment files. Each script must be runnable with:

```bash
uv run python projects/<project-name>/scripts/my_script.py
```

Scripts load saved params, import functions from experiment modules, and
produce analysis or plots without retraining.

---

## Claude Code Settings

Bash permission rules use a **space wildcard**, not a colon:

```json
{ "permissions": { "allow": ["Bash(uv run *)", "Bash(python *)"] } }
```

`Bash(uv run:*)` (colon syntax) is deprecated and **does not match any real
command** — the rule silently does nothing, causing repeated permission prompts
for identical-looking commands.
