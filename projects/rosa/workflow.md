# Experiment Workflow — ROSA

A symbolic-streaming MNIST classifier where a non-differentiable statistical
next-token predictor (ROSA, from
[bcml-labs/rosa-plus](https://github.com/bcml-labs/rosa-plus)) sits between a
trainable input module and a trainable output module. Images and labels are
discretized to tokens by an ACT-MLP, ROSA accumulates the full token stream
across all training data, and its predicted next-token is fed back as a
feature for the output module's 10-way classifier.

For algorithmic details (model architecture, ROSA, loss, key hyperparameters)
see [concepts.md](concepts.md). For the experiment roadmap see
[plan.md](plan.md).

---

## Directory Layout

```
projects/rosa/
├── workflow.md             # This file
├── concepts.md             # Architecture, ROSA core, training signal, hyperparams
├── plan.md                 # Multi-experiment roadmap with hypotheses
├── experiments1.py         # One file per experiment; numbered sequentially
├── results.jsonl           # Append-only results log — one JSON object per line
├── params_expN.pkl         # Saved input/output module params + codebook (gitignored)
├── rosa_expN.json          # Saved ROSA SAM + fallback LM (gitignored, large)
├── history_expN.pkl        # Per-epoch training history (gitignored)
├── logs/                   # Experiment and runner logs (gitignored)
├── lib/
│   ├── __init__.py
│   ├── rosa_core.py        # Pure-Python SAM + Witten–Bell fallback (port of rosaplus.py)
│   └── viz.py              # Visualization utilities for this project
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs  ← always use this
    ├── poll_result.py      # Wait for / display results
    ├── gen_viz_expN.py     # Post-training visualizations for experiment N
    └── tmp/                # Throwaway / generator scripts (not committed)
```

---

## Invariant Rules

1. **Always `uv run python`** — never raw `python` or `python3`.

2. **Always use `run_experiments.py`** to launch experiments — never call
   `uv run python experimentsN.py` directly. The runner handles log paths,
   GPU assignment, and the results table.

3. **Throwaway scripts go in `scripts/tmp/`** — never `python -c "..."` inline.

4. **No bash polling loops** — read the log file once or write a Python script.

5. **ROSA is non-JAX**. The SAM is a pure-Python suffix automaton; it lives in
   CPU memory and is queried per-sample. Do not try to JIT it. Treat it like
   an external black-box predictor: gather token IDs from the JAX forward
   pass, query ROSA in Python, and pass ROSA's predicted IDs back into JAX.

---

## Running Experiments

```bash
# Fire and forget — detach to background, return immediately
uv run python projects/rosa/scripts/run_experiments.py --bg exp1

# With auto-viz after completion
uv run python projects/rosa/scripts/run_experiments.py --bg --viz scripts/gen_viz_exp1.py exp1

# Sequential blocking
uv run python projects/rosa/scripts/run_experiments.py exp1 exp2

# Parallel across GPUs
uv run python projects/rosa/scripts/run_experiments.py --parallel exp1 exp2

# Pin to a specific GPU
uv run python projects/rosa/scripts/run_experiments.py --gpu 1 exp1
```

Logs always go to `projects/rosa/logs/{exp_name}.log`.

---

## Monitoring

```bash
tail -20 projects/rosa/logs/exp1.log
uv run python projects/rosa/scripts/poll_result.py exp1
uv run python projects/rosa/scripts/poll_result.py --all
```

---

## JAX Performance

### Why we don't use `lax.scan` over batches here

ROSA is a Python data structure that must be queried *between* the input
module and the output module — the inner step is not a pure JAX function.
The natural shape is:

```
for batch in epoch:
    z, halts          = jit_run_input_module(params, batch)        # JAX
    token_ids         = jit_quantize(codebook, z)                  # JAX
    rosa_preds        = rosa.predict_batch(token_ids, halts)       # Python
    rosa_pred_emb     = jit_lookup(codebook, rosa_preds)           # JAX
    loss, grads       = jit_loss_and_grads(params, batch, z, rosa_pred_emb)
    params, opt_state = jit_update(params, opt_state, grads)
    rosa.commit_batch(token_ids, halts, label_ids)                 # Python
```

Each JAX call is JIT'd individually; the Python ROSA calls break the scan
but they're cheap (suffix-automaton lookup is ~O(1) per token).

### GPU isolation

Never launch two JAX experiments on the same GPU simultaneously. Use
`run_experiments.py` sequential mode or `--parallel` (sets
`CUDA_VISIBLE_DEVICES` per process).

### First epoch is slow

Epoch 0 includes XLA compilation time and may be 10–20× slower than
subsequent epochs.

### Don't close over large arrays in `@jax.jit`

JAX embeds them as compile-time constants. Pass big arrays as arguments,
not closure captures, especially the codebook.

---

## Experiment File Structure

```
module docstring    — hypothesis, what changed, usage line
hyperparameters     — UPPER_SNAKE_CASE constants at module top
precomputed consts  — JAX arrays that don't change per run
utilities           — n_params, append_result
init_params()       — input module + output module + codebook
run_input_act()     — the recurrent ACT loop, returns z[1..K_max], halt_mask
quantize()          — nearest-codebook (forward) + STE (backward)
output_module()     — recent-window pool + rosa-pred-emb → 10-way logits
loss_fn()           — CE digit + commit + ponder
make_step_fn()      — JIT'd one-batch step (no scan — ROSA is in Python)
eval_metrics()      — frozen-ROSA accuracy on test set
train()             — epoch loop with periodic eval; returns (params, rosa, history, elapsed)
__main__            — load data → init → train → visualize → append_result
```

---

## Results Logging

### Skip-if-done

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

### Append one JSON object per experiment to `results.jsonl`:

```python
JSONL = Path(__file__).parent / "results.jsonl"

def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
```

Every row must include: all hyperparameters, `n_params`, `time_s`, `epochs`,
final metric values (`test_acc`, `train_acc`, `codebook_used`,
`rosa_hit_rate`, `sam_state_count`, mean ACT steps per sample/label), and
per-epoch curve lists.

---

## Saving and Reloading State

A ROSA run has three pieces of state:
- **JAX params** (input module + output module + codebook) — `params_expN.pkl`
- **ROSA SAM + fallback LM** — `rosa_expN.json` (uses `ROSAPlus.save()`)
- **Per-epoch history** — `history_expN.pkl`

```python
with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({k: np.array(v) for k, v in params.items()}, f)

rosa.save(Path(__file__).parent / f"rosa_{EXP_NAME}.json")

with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump(history, f)
```

All three are gitignored. `results.jsonl` is the only committed output.

---

## Visualization

Shared visualization functions live in `lib/viz.py`. Format guide:
- **SVG** for learning curves and line plots
- **PNG** (dpi=150) for pixel images — token grids, codebook visualizations

ROSA-specific plots worth standardizing:
- Codebook utilization histogram across training
- ROSA-hit-rate curve (SAM vs. fallback fraction across epochs)
- SAM state count growth across epochs
- Per-digit accuracy with vs. without ROSA's contribution (ablation)
- ACT halt distribution (mean ± std token count per sample, per label)

### Post-training regeneration

```bash
uv run python projects/rosa/scripts/gen_viz_exp1.py
```

Reads `params_exp1.pkl`, `rosa_exp1.json`, and `history_exp1.pkl` and
regenerates plots without re-running the experiment.

---

## Report Writing

Reports are markdown files with plots linked as R2 URLs.

### Workflow

1. Write `scripts/gen_report_<name>.py` — generates the `.md` and uploads plots.
2. Write `scripts/tmp/upload_report_<name>.py` — converts the `.md` to HTML and uploads to R2.
3. Always generate the local `.md` first and inspect it before uploading.

### Uploading plots

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

url = save_matplotlib_figure("rosa_my_plot", fig, dpi=150)
plt.close(fig)
```

**Do not embed base64** — bloats the markdown file and breaks diffs.

### Uploading the report

```python
from shared_lib.report import save_report_file
url = save_report_file("rosa_report_<name>", Path("projects/rosa/report-<name>.md"))
print(url)
```

Do not pass `title=` — duplicates the `<h1>`.
