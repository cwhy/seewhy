# Experiment Workflow — ema-feature

This project explores Exponential Moving Average (EMA) feature representations
during neural network training on MNIST. The core idea is to maintain a slow-moving
EMA copy of encoder features (or model weights) as stable targets or auxiliary
signals during training.

For algorithmic details (EMA schedule, target network updates, loss formulations)
see [concepts.md](concepts.md) once it exists.

---

## Directory Layout

```
projects/ema-feature/
├── workflow.md             # This file
├── concepts.md             # EMA algorithm, target network, metrics
├── experiments1.py         # One file per experiment; numbered sequentially
├── results.jsonl           # Append-only results log — one JSON object per line
├── params_expN.pkl         # Saved model params after each run (gitignored)
├── history_expN.pkl        # Per-epoch training history (gitignored)
├── logs/                   # Experiment and runner logs (gitignored)
├── lib/
│   ├── __init__.py
│   └── viz.py              # Visualization utilities for this project
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs  ← always use this
    ├── poll_result.py      # Wait for / display results
    ├── gen_viz_expN.py     # Post-training visualizations for experiment N
    └── tmp/                # Throwaway / generator scripts (not committed)
```

---

## Invariant Rules

These apply to every session without exception:

1. **Always `uv run python`** — never raw `python` or `python3`. The `.venv`
   has CUDA-specific packages; raw python may resolve a different interpreter.

2. **Always use `run_experiments.py`** to launch experiments — never call
   `uv run python experimentsN.py` directly. The runner handles log paths,
   GPU assignment, and the results table.

3. **Throwaway scripts go in `scripts/tmp/`** — never `python -c "..."` inline.
   Write to `scripts/tmp/<name>.py`, then run with
   `uv run python projects/ema-feature/scripts/tmp/<name>.py`.

4. **No bash polling loops** — to check progress, read the log file once or
   write a Python script. Never `while true; do ...; sleep N; done`.

---

## Running Experiments

```bash
# Fire and forget — detach to background, return immediately
uv run python projects/ema-feature/scripts/run_experiments.py --bg exp1

# With auto-viz after completion
uv run python projects/ema-feature/scripts/run_experiments.py --bg --viz scripts/gen_viz_exp1.py exp1

# Sequential blocking (safe — no XLA contention)
uv run python projects/ema-feature/scripts/run_experiments.py exp1 exp2

# Parallel across GPUs
uv run python projects/ema-feature/scripts/run_experiments.py --parallel exp1 exp2

# Pin to a specific GPU
uv run python projects/ema-feature/scripts/run_experiments.py --gpu 1 exp1
```

Logs always go to `projects/ema-feature/logs/{exp_name}.log`.

---

## Monitoring

```bash
# Check progress of a running experiment
tail -20 projects/ema-feature/logs/exp1.log

# Poll until result appears in results.jsonl
uv run python projects/ema-feature/scripts/poll_result.py exp1
uv run python projects/ema-feature/scripts/poll_result.py --all
```

---

## JAX Performance

### Use `lax.scan` for the batch loop

Replace `for b in range(num_batches)` with `jax.lax.scan`. This eliminates
thousands of Python→XLA dispatch calls per epoch and lets XLA fuse the full
epoch as one computation.

```python
def make_epoch_fn(optimizer):
    def scan_body(carry, inputs):
        params, ema_params, opt_state = carry
        batch, key = inputs
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, ema_params, batch, key
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_ema = update_ema(ema_params, new_params, EMA_DECAY)
        return (new_params, new_ema, new_opt_state), (loss, *aux)

    def epoch_fn(params, ema_params, opt_state, Xs_batched, epoch_key):
        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(
            jnp.arange(num_batches)
        )
        (params, ema_params, opt_state), stats = jax.lax.scan(
            scan_body, (params, ema_params, opt_state), (Xs_batched, keys)
        )
        return params, ema_params, opt_state, *[s.mean() for s in stats]

    return jax.jit(epoch_fn)
```

### GPU isolation

Never launch two JAX experiments on the same GPU simultaneously. XLA
compilations from two processes compete for memory and can OOM. Use
`run_experiments.py` sequential mode or `--parallel` (which sets
`CUDA_VISIBLE_DEVICES` per process).

### First epoch is slow — JIT compilation

Epoch 0 includes XLA compilation time and may be 10–20× slower than
subsequent epochs. Do not judge speed from epoch 0.

---

## Experiment File Structure

Every experiment file follows the same layout:

```
module docstring    — hypothesis, what changed, EMA_DECAY value, usage line
hyperparameters     — UPPER_SNAKE_CASE: EMA_DECAY, LR, BATCH_SIZE, EPOCHS, …
precomputed consts  — JAX arrays that don't change per run
utilities           — n_params, update_ema, append_result
init_params()       — online params + EMA params (same shape, same init)
encode() / forward()
loss_fn()           — uses both online params and ema_params
make_epoch_fn()     — returns JIT-compiled epoch fn carrying both param trees
eval_metrics()      — accuracy / loss on test set; eval both online and EMA
train()             — epoch loop with periodic viz; returns (params, ema_params, history)
__main__            — load data → init → train → visualize → append_result
```

Hyperparameters go at module level in `UPPER_SNAKE_CASE` so they are visible
at a glance and easy to diff between experiments.

---

## EMA Update

The EMA parameter tree is updated once per step after the optimizer step:

```python
def update_ema(ema_params, new_params, decay):
    return jax.tree.map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params, new_params
    )
```

`EMA_DECAY` is typically in `[0.99, 0.999, 0.9999]`. Higher decay → slower
EMA, more stable but slower to track the online model.

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
`ema_decay`, final metric values (both online and EMA), and per-epoch curve
lists. Curves are cheap to store and invaluable for re-plotting.

---

## Saving and Reloading Params

Save both the online params and EMA params at the end of every run:

```python
with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({
        "online": {k: np.array(v) for k, v in params.items()},
        "ema":    {k: np.array(v) for k, v in ema_params.items()},
    }, f)

with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump(history, f)
```

To reload in a script:

```python
with open("projects/ema-feature/params_exp1.pkl", "rb") as f:
    saved = pickle.load(f)
params     = {k: jnp.array(v) for k, v in saved["online"].items()}
ema_params = {k: jnp.array(v) for k, v in saved["ema"].items()}
```

Both pkl files are gitignored. `results.jsonl` is the only committed output.

---

## Visualization

Shared visualization functions live in `lib/viz.py`. Do not inline
visualization code in experiment files — it diverges and is hard to keep
consistent.

```python
from lib.viz import save_learning_curves, save_ema_vs_online_curves
```

Format guide:
- **SVG** for learning curves, line plots, anything vector-based
- **PNG** (dpi=150) for pixel images — reconstructions, feature grids

Always log the returned URL:

```python
url = save_matplotlib_figure("exp1_learning_curves", fig, format="svg")
logging.info(f"  learning curves → {url}")
```

### Post-training regeneration

```bash
uv run python projects/ema-feature/scripts/gen_viz_exp1.py
```

Reads `params_exp1.pkl` and `history_exp1.pkl` from disk and regenerates
all plots without re-running the experiment.

---

## Diagnostic Scripts (`scripts/tmp/`)

One-off helpers live in `scripts/tmp/` and are not committed. Examples:

| Script | Purpose |
|--------|---------|
| `check_ema_divergence.py exp1` | L2 distance between online and EMA params over time |
| `compare_acc.py exp1 exp2` | Side-by-side online vs EMA accuracy table |
| `mk_expN.py` | Generator that patches a parent experiment into a new one |

Run all with `uv run python projects/ema-feature/scripts/tmp/<name>.py [exp_name]`.
