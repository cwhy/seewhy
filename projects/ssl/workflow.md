# Experiment Workflow — SSL

EMA-JEPA self-supervised learning on MNIST/Fashion-MNIST: comparing encoder
architectures (MLP, ConvNet, ViT, Gram+Hadamard) under a momentum target encoder.

For algorithmic details (model architecture, loss, key hyperparameters) see
[concepts.md](concepts.md) *(not yet written — architecture notes are inline in exp docstrings)*.

---

## Status: HALTED (2026-04-01)

Continuing in **`projects/ema-viz/`** — visualising features learned by the best encoder.

**Best encoder found**: Gram-dual (`exp_ema_1024`) — 97.3% MNIST SSL, 83.6% Fashion-MNIST SSL.
Saved params: `projects/ssl/params_exp_ema_1024.pkl` (gitignored, ~12 MB).

**Pareto front at LATENT=1024** (linear probe accuracy):
| Encoder      | MNIST  | F-MNIST |
|---|---|---|
| Gram-dual    | 97.3%  | 83.6%   |
| Gram-single  | 96.9%  | 82.8%   |
| MLP-shallow  | 93.2%  | 85.6%   |
| ViT-patch    | 95.1%  | 80.9%   |
| ConvNet-v1   | 93.9%  | 80.1%   |

**Key finding**: flat pixel masking gives MLP/Gram an unfair advantage. ViT-patch
(patch-level masking) closes 4.9 pp of the gap. Gram encodes strong co-activation
priors well-suited to MNIST strokes but not Fashion-MNIST textures.

---

## Directory Layout

```
projects/ssl/
├── workflow.md             # This file
├── concepts.md             # Algorithm details, metrics, key findings
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
   `uv run python projects/ssl/scripts/tmp/<name>.py`.

4. **No bash polling loops** — to check progress, read the log file once or
   write a Python script. Never `while true; do ...; sleep N; done`.

---

## Running Experiments

```bash
# Fire and forget — detach to background, return immediately
uv run python projects/ssl/scripts/run_experiments.py --bg exp1

# With auto-viz after completion
uv run python projects/ssl/scripts/run_experiments.py --bg --viz scripts/gen_viz_exp1.py exp1

# Sequential blocking (safe — no XLA contention)
uv run python projects/ssl/scripts/run_experiments.py exp1 exp2

# Parallel across GPUs
uv run python projects/ssl/scripts/run_experiments.py --parallel exp1 exp2

# Pin to a specific GPU
uv run python projects/ssl/scripts/run_experiments.py --gpu 1 exp1
```

Logs always go to `projects/ssl/logs/{exp_name}.log`.

---

## Monitoring

```bash
# Check progress of a running experiment
tail -20 projects/ssl/logs/exp1.log

# Poll until result appears in results.jsonl
uv run python projects/ssl/scripts/poll_result.py exp1
uv run python projects/ssl/scripts/poll_result.py --all
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

Pre-batch data before calling:
```python
Xs_batched = Xs[:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, -1)
```

### GPU isolation

Never launch two JAX experiments on the same GPU simultaneously. XLA
compilations from two processes compete for memory and can OOM. Use
`run_experiments.py` sequential mode or `--parallel` (which sets
`CUDA_VISIBLE_DEVICES` per process).

### First epoch is slow — JIT compilation

Epoch 0 includes XLA compilation time and may be 10–20× slower than
subsequent epochs. Do not judge speed from epoch 0.

### Don't close over large arrays in `@jax.jit`

If a `@jax.jit` function closes over a large array, JAX embeds it as a
compile-time constant and allocates it on the GPU at compile time. This
silently OOMs when the function is recompiled (e.g. after switching encoder
architecture mid-eval loop).

```python
# Bad — E_b (~245 MB) captured as constant, OOMs on recompile
@jax.jit
def epoch(W, b, state):
    (W, b), state = jax.lax.scan(step, ((W, b), state), (E_b, Y_b))[0]
    return W, b, state

# Good — pass as arguments
@jax.jit
def epoch(W, b, state, E_b, Y_b):
    (W, b), state = jax.lax.scan(step, ((W, b), state), (E_b, Y_b))[0]
    return W, b, state
```

---

## Experiment File Structure

Every experiment file follows the same layout:

```
module docstring    — hypothesis, what changed, usage line
hyperparameters     — UPPER_SNAKE_CASE constants at module top
precomputed consts  — JAX arrays that don't change per run
utilities           — n_params, append_result, any project-specific helpers
init_params()
encode() / forward()
loss_fn()
make_epoch_fn()     — returns JIT-compiled epoch fn
eval_metrics()      — @jax.jit accuracy / metric functions
train()             — epoch loop with periodic viz; returns (params, history, elapsed)
__main__            — load data → init → train → visualize → append_result
```

Hyperparameters go at module level in `UPPER_SNAKE_CASE` so they are visible
at a glance and easy to diff between experiments.

---

## Results Logging

### Skip-if-done

Check at the top of every `__main__` whether the experiment is already in the
output JSONL and exit early. Makes batch runners idempotent after a crash:

```python
done = set()
if JSONL.exists():
    with open(JSONL) as f:
        for line in f:
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
if EXP_NAME in done:
    logging.info(f"{EXP_NAME} already done — skipping")
    exit(0)
```

### JSONL deduplication

Two concurrent processes both read the `done` set before either has written,
then both write the same rows. After any crash or concurrent run, deduplicate
by key before analysis.

### Append one JSON object per experiment to `results.jsonl`:

```python
JSONL = Path(__file__).parent / "results.jsonl"

def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
```

Every row must include: all hyperparameters, `n_params`, `time_s`, `epochs`,
final metric values, and per-epoch curve lists. Curves are cheap to store and
invaluable for re-plotting without re-running.

---

## Saving and Reloading Params

Save params at the end of every run:

```python
with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({k: np.array(v) for k, v in params.items()}, f)

with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump(history, f)
```

To reload in a script:

```python
with open("projects/ssl/params_exp1.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}
```

Both pkl files are gitignored. `results.jsonl` is the only committed output.

---

## Visualization

Shared visualization functions live in `lib/viz.py`. Do not inline
visualization code in experiment files — it diverges and is hard to keep
consistent.

```python
from lib.viz import save_learning_curves, save_reconstruction_grid
```

Format guide:
- **SVG** for learning curves, line plots, anything vector-based
- **PNG** (dpi=150) for pixel images — reconstructions, heatmaps, sample grids

Always log the returned URL:

```python
url = save_matplotlib_figure("exp1_learning_curves", fig, format="svg")
logging.info(f"  learning curves → {url}")
```

### Post-training regeneration

```bash
uv run python projects/ssl/scripts/gen_viz_exp1.py
```

Reads `params_exp1.pkl` and `history_exp1.pkl` from disk and regenerates
all plots without re-running the experiment.

---

## Evaluation Notes

### Linear probe on transfer datasets

When evaluating a MNIST-trained encoder on Fashion-MNIST, the linear probe
must be trained on **F-MNIST training embeddings** (with F-MNIST labels) and
tested on F-MNIST test embeddings. Do not train the probe on MNIST labels and
apply it to F-MNIST — the two label spaces are completely different (digits vs.
clothing categories), giving ~10% accuracy (chance).

```python
# Wrong — MNIST probe applied to F-MNIST labels (chance performance)
probe_fmnist = linear_probe(params, X_tr_mnist, Y_tr_mnist, X_te_fmnist, Y_te_fmnist)

# Correct — probe trained on F-MNIST train set, same MNIST-trained encoder
probe_fmnist = linear_probe(params, X_tr_fmnist, Y_tr_fmnist, X_te_fmnist, Y_te_fmnist)
```

This applies to any cross-dataset probe evaluation where label spaces differ.

---

## Report Writing

Reports are markdown files with plots linked as R2 URLs (not base64).

### Workflow

1. Write `scripts/gen_report_<name>.py` — generates the `.md` and uploads plots.
2. Write `scripts/tmp/upload_report_<name>.py` — converts the `.md` to HTML and uploads to R2.
3. Always generate the local `.md` first and inspect it before uploading.

### Uploading plots

Use `save_matplotlib_figure()` from `shared_lib.media` — it uploads the figure
to R2 and returns the URL. Reference plots as plain markdown image links:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

url = save_matplotlib_figure("ssl_my_plot", fig, dpi=150)
plt.close(fig)
# In the report body:
# ![My plot](https://media.tanh.xyz/...)
```

**Do not embed base64** (`data:image/png;base64,...`) — it bloats the markdown
file (1+ MB vs ~10 KB) and makes diffs unreadable.

### Uploading the report

```python
# scripts/tmp/upload_report_<name>.py
from shared_lib.report import save_report_file
url = save_report_file("ssl_report_<name>", Path("projects/ssl/report-<name>.md"))
print(url)
```

Do not pass `title=` to `save_report_file` — it injects a duplicate `<h1>`
on top of the markdown's own `# heading`.

---

## Diagnostic Scripts (`scripts/tmp/`)

One-off helpers live in `scripts/tmp/` and are not committed. Examples:

| Script | Purpose |
|--------|---------|
| `inspect_expN.py` | Print metrics / shapes for a saved run |
| `compare_exps.py expA expB` | Side-by-side metric comparison |
| `mk_expN.py` | Generator that patches a parent experiment into a new one |

Run all with `uv run python projects/ssl/scripts/tmp/<name>.py [args]`.

---

## Template version used

This project was set up with template **v1**.
