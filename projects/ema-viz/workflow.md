# Experiment Workflow — ema-viz

Visualising the internal representations learned by the EMA-JEPA Gram-dual encoder
(best SSL model from `projects/ssl/`) to understand what the EMA target encoder captures.

For algorithmic details (model architecture, loss, key hyperparameters) see
[concepts.md](concepts.md).

---

## Directory Layout

```
projects/ema-viz/
├── workflow.md             # This file
├── concepts.md             # Architecture details, what each param does
├── exp1.py                 # Latent space geometry: t-SNE, cluster quality, NN retrieval
├── exp2.py                 # ...
├── results.jsonl           # Append-only results log — one JSON object per line
├── params_expN.pkl         # Saved outputs per experiment (gitignored)
├── logs/                   # Experiment and runner logs (gitignored)
├── lib/
│   ├── __init__.py
│   └── viz.py              # Visualization utilities for this project
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs  ← always use this
    ├── poll_result.py      # Wait for / display results
    └── tmp/                # Throwaway / generator scripts (not committed)
```

---

## Source model

All experiments load the frozen Gram-dual encoder trained in `projects/ssl/`:

```python
import pickle, sys
from pathlib import Path
import jax.numpy as jnp

SSL_DIR = Path(__file__).parent.parent / "ssl"
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

with open(SSL_DIR / "params_exp_ema_1024.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}
```

Encoder hyperparameters (from `ssl/exp_ema_1024.py`):
- `LATENT_DIM = 1024`, `DEC_RANK = 32` (32² = 1024)
- `D_R, D_C, D_V = 6, 6, 4` → `D = 144` per-pixel feature dim
- Input: (B, 784) float32 pixels → output: (B, 1024) embedding

---

## Invariant Rules

These apply to every session without exception:

1. **Always `uv run python`** — never raw `python` or `python3`. The `.venv`
   has CUDA-specific packages; raw python may resolve a different interpreter.

2. **Always use `run_experiments.py`** to launch experiments — never call
   `uv run python expN.py` directly. The runner handles log paths,
   GPU assignment, and the results table.

3. **Throwaway scripts go in `scripts/tmp/`** — never `python -c "..."` inline.
   Write to `scripts/tmp/<name>.py`, then run with
   `uv run python projects/ema-viz/scripts/tmp/<name>.py`.

4. **No bash polling loops** — to check progress, read the log file once or
   write a Python script. Never `while true; do ...; sleep N; done`.

5. **Never retrain the encoder** — all experiments use frozen params from ssl/.
   The encoder is fixed; only the visualisation logic changes between exps.

---

## Running Experiments

```bash
# Fire and forget — detach to background, return immediately
uv run python projects/ema-viz/scripts/run_experiments.py --bg exp1

# Sequential blocking (safe — no XLA contention)
uv run python projects/ema-viz/scripts/run_experiments.py exp1 exp2

# Pin to a specific GPU
uv run python projects/ema-viz/scripts/run_experiments.py --gpu 1 exp1
```

Logs always go to `projects/ema-viz/logs/{exp_name}.log`.

---

## Monitoring

```bash
# Check progress of a running experiment
tail -20 projects/ema-viz/logs/exp1.log

# Poll until result appears in results.jsonl
uv run python projects/ema-viz/scripts/poll_result.py exp1
uv run python projects/ema-viz/scripts/poll_result.py --all
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
        params, opt_state = carry
        batch, key = inputs
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, key
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), (loss, *aux)

    def epoch_fn(params, opt_state, Xs_batched, epoch_key):
        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(
            jnp.arange(num_batches)
        )
        (params, opt_state), stats = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_batched, keys)
        )
        return params, opt_state, *[s.mean() for s in stats]

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
module docstring    — hypothesis, what to visualise, usage line
hyperparameters     — UPPER_SNAKE_CASE constants at module top
load_encoder()      — load frozen params from ssl/params_exp_ema_1024.pkl
encode()            — copy of Gram-dual encode() from ssl/exp_ema_1024.py
visualise()         — main analysis + figure generation
__main__            — load data → encode → visualise → append_result
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

Every row must include: `exp_name`, `description`, `time_s`, and any
quantitative metrics produced (e.g. silhouette score, retrieval accuracy).

---

## Saving and Reloading Outputs

```python
import pickle, numpy as np
from pathlib import Path

with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({"embeddings": np.array(embeddings), ...}, f)
```

To reload in a script:
```python
with open("projects/ema-viz/params_exp1.pkl", "rb") as f:
    data = pickle.load(f)
```

Both pkl files are gitignored. `results.jsonl` is the only committed output.

---

## Visualization

Format guide:
- **SVG** for scatter plots, line plots, anything vector-based
- **PNG** (dpi=150) for pixel images — image grids, heatmaps, reconstructions

Always log the returned URL:

```python
url = save_matplotlib_figure("exp1_tsne", fig, format="svg")
logging.info(f"  t-SNE → {url}")
```

---

## Diagnostic Scripts (`scripts/tmp/`)

One-off helpers live in `scripts/tmp/` and are not committed. Examples:

| Script | Purpose |
|--------|---------|
| `inspect_expN.py` | Print metrics / shapes for a saved run |
| `mk_expN.py` | Generator that patches a parent experiment into a new one |

Run all with `uv run python projects/ema-viz/scripts/tmp/<name>.py [args]`.

---

## Template version used

This project was set up with template **v1**.
