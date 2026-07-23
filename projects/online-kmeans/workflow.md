# Experiment Workflow — online-kmeans

This project follows the same conventions as imle-gram. This file documents
online-kmeans-specific details. For generic patterns (GPU isolation, JSONL
logging, etc.) see [imle-gram/workflow.md](../imle-gram/workflow.md).

---

## Directory Layout

```
projects/online-kmeans/
├── workflow.md             # This file
├── experiments1.py         # Full-batch K-means (Lloyd's)
├── experiments2.py         # Mini-batch K-means (BATCH_SIZE=256)
├── experiments3.py         # Online K-means (BATCH_SIZE=1)
├── results.jsonl           # Append-only results log
├── params_expN.pkl         # Saved centroids (gitignored)
├── history_expN.pkl        # Per-epoch training history (gitignored)
├── label_probs_expN.npy    # Per-cluster label distributions (gitignored)
├── logs/                   # Experiment and runner logs (gitignored)
├── lib/
│   ├── __init__.py
│   └── viz.py              # Visualization (centroid grid, learning curves)
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs  ← always use this
    ├── poll_result.py      # Wait for / display results
    ├── gen_viz_clustering.py  # Clustering analysis plots (takes exp_name arg)
    └── tmp/                # Throwaway / diagnostic scripts (not committed)
```

---

## Invariant Rules

1. **Always `uv run python`** — never raw `python` or `python3`.

2. **Always use `run_experiments.py`** to launch experiments.

3. **Throwaway scripts go in `scripts/tmp/`** — never `python -c "..."` inline.

4. **No bash polling loops** — read the log file once or write a Python script.

---

## Running Experiments

```bash
# Fire and forget
uv run python projects/online-kmeans/scripts/run_experiments.py --bg exp1

# With auto-viz after completion
uv run python projects/online-kmeans/scripts/run_experiments.py --bg --viz scripts/gen_viz_clustering.py exp1

# Sequential blocking
uv run python projects/online-kmeans/scripts/run_experiments.py exp1 exp2 exp3

# Parallel across GPUs
uv run python projects/online-kmeans/scripts/run_experiments.py --parallel exp1 exp2 exp3
```

Logs always go to `projects/online-kmeans/logs/{exp_name}.log`.

---

## Monitoring

```bash
# Check progress
tail -20 projects/online-kmeans/logs/exp1.log

# Poll until result appears
uv run python projects/online-kmeans/scripts/poll_result.py exp1
uv run python projects/online-kmeans/scripts/poll_result.py --all
```

---

## Visualization

### Post-training

```bash
uv run python projects/online-kmeans/scripts/gen_viz_clustering.py exp1
```

Reads `params_expN.pkl` and `history_expN.pkl` from disk. Produces:
- Per-epoch cluster usage line plot
- Inertia curve (train + eval)
- Label accuracy (hard majority-vote + Bayes soft)
- Per-cluster label distribution grid
- Centroid image grid (28×28)
- Centroid cosine similarity heatmap
- Single HTML page assembling all plots

---

## Experiment Comparison

| Exp  | Algorithm        | BATCH_SIZE | Update rule                      |
|------|------------------|-----------|----------------------------------|
| exp1 | Lloyd's K-means  | full      | exact centroid mean each epoch   |
| exp2 | Mini-batch K-means | 256     | count-based incremental mean     |
| exp3 | Online K-means   | 1         | count-based incremental mean     |

All use:
- K-means++ initialisation
- N_CODES=128
- Dead cluster reinitialisation (one random training sample per dead cluster/epoch)

---

## Algorithm Notes

### Count-based incremental update (exp2, exp3)

```
counts[c] += n_c
lr = n_c / counts[c]
centroids[c] += lr * (batch_mean_c - centroids[c])
```

Counts accumulate across the full training run. The effective lr decays as
1/t, ensuring convergence to the true centroid mean — equivalent to computing
the running mean of all samples assigned to cluster c.

### Dead cluster handling

After each epoch, any cluster with zero assignments is reinitialised from a
random training sample.  This prevents mode collapse without restarting the
full K-means++ init.

### History format

Each experiment saves:

```python
with open(f"history_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump(history, f)

with open(f"params_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({"centroids": np.array(centroids)}, f)

np.save(f"label_probs_{EXP_NAME}.npy", label_probs)
```

`params["centroids"]` is `(N_CODES, 784)` float32.
`decode_prototypes(params)` returns `params["centroids"].reshape(N_CODES, 28, 28)`.
