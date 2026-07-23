# Experiment Workflow — component-analysis

This project follows the same conventions as imle-gram. See
[imle-gram/workflow.md](../imle-gram/workflow.md) for the generic workflow
(GPU isolation, JSONL logging, etc.).

For algorithmic details (PCA, MCCA, metrics) see [concepts.md](concepts.md).

---

## Directory Layout

```
projects/component-analysis/
├── workflow.md             # This file
├── concepts.md             # PCA, MCCA algorithm details, metrics
├── experiments1.py         # PCA: global + per-digit (50 components)
├── experiments2.py         # MCCA: multiset CCA over 10 digit datasets
├── results.jsonl           # Append-only results log
├── pca_results_expN.pkl    # Full PCA results (gitignored)
├── mcca_results_expN.pkl   # Full MCCA results (gitignored)
├── logs/                   # Experiment and runner logs (gitignored)
├── lib/
│   ├── __init__.py
│   ├── pca.py              # Pure-function PCA (fit, project, reconstruct)
│   ├── cca.py              # Pure-function MCCA (fit, project, canonical_correlations)
│   ├── viz.py              # Static SVG plots (component images, scatter, variance curves)
│   └── interactive.py      # Interactive HTML pages (scatter with hover, component grid)
└── scripts/
    ├── run_experiments.py  # Launch and manage experiment runs  ← always use this
    ├── poll_result.py      # Wait for / display results
    ├── gen_viz_exp1.py     # Post-training viz for exp1 (PCA)
    ├── gen_viz_exp2.py     # Post-training viz for exp2 (MCCA)
    └── tmp/                # Throwaway / generator scripts (not committed)
```

---

## Invariant Rules

These apply to every session without exception:

1. **Always `uv run python`** — never raw `python` or `python3`.

2. **Always use `run_experiments.py`** to launch experiments — never call
   `uv run python experimentsN.py` directly.

3. **Throwaway / one-off scripts go in `scripts/tmp/`** — this includes smoke
   tests, diagnostic checks, data inspection, and generator scripts.
   **Never** use `python -c "..."` or `python3 -c "..."` inline.
   Always write to `scripts/tmp/<name>.py` and run with
   `uv run python projects/component-analysis/scripts/tmp/<name>.py`.

4. **No bash polling loops** — to check progress, read the log file once or
   write a Python script. Never `while true; do ...; sleep N; done`.

---

## Running Experiments

```bash
# PCA analysis (exp1)
uv run python projects/component-analysis/scripts/run_experiments.py exp1

# MCCA analysis (exp2)
uv run python projects/component-analysis/scripts/run_experiments.py exp2

# Both sequentially with auto-viz
uv run python projects/component-analysis/scripts/run_experiments.py --viz scripts/gen_viz_exp1.py exp1
uv run python projects/component-analysis/scripts/run_experiments.py --viz scripts/gen_viz_exp2.py exp2

# Background (fire and forget)
uv run python projects/component-analysis/scripts/run_experiments.py --bg exp1

# Both in parallel
uv run python projects/component-analysis/scripts/run_experiments.py --parallel exp1 exp2
```

Logs always go to `projects/component-analysis/logs/{exp_name}.log`.

---

## Monitoring

```bash
# Check progress of a running experiment
tail -20 projects/component-analysis/logs/exp1.log

# Poll until result appears in results.jsonl
uv run python projects/component-analysis/scripts/poll_result.py exp1
uv run python projects/component-analysis/scripts/poll_result.py --all
```

---

## Visualization

### Three output types

Every experiment produces three kinds of visual output:

| Type | How | Result |
|------|-----|--------|
| Static SVG | `lib/viz.py` functions → `save_svg` | Individual figure URLs |
| Figures page | collect `(caption, url)` pairs → `save_figures_page` | Tailwind HTML grid with click-through detail views |
| Interactive HTML | `lib/interactive.py` functions → `save_html` | Self-contained page with pan/zoom, hover, toggles |

### Static SVG plots (`lib/viz.py`)

Each function saves one matplotlib figure as SVG via `shared_lib.media.save_svg`
and returns the URL. Use these as the building blocks for a figures page.

```python
from shared_lib.media import save_svg
import matplotlib.pyplot as plt

fig, ax = plt.subplots(...)
# ... draw ...
url = save_svg(f"{exp_name}_{tag}", fig)
plt.close(fig)
```

All `lib/viz.py` functions follow this pattern and return the SVG URL.

### Assembling a figures page

Collect `(caption, url)` pairs and pass to `save_figures_page` at the end.
The result is a responsive Tailwind HTML page where each figure links to a
full-page detail view.

```python
from shared_lib.html import save_figures_page, save_flat_grid

figures = []
figures.append(("Global PCA components", save_component_images(...)))
figures.append(("Explained variance",    save_variance_curve(...)))
figures.append(("2D projection",         save_projection_2d(...)))

page_url = save_figures_page(f"{exp_name}_analysis", f"{exp_name} — Analysis", figures)
print(page_url)
```

Use `save_flat_grid` instead if you don't want per-figure detail pages
(e.g. for a grid of many small thumbnails).

### Interactive HTML pages (`lib/interactive.py`)

For scatter plots with hover and component inspection, use the two helpers:

```python
from lib.interactive import save_projection_interactive, save_components_interactive

# Interactive 2D scatter — pan/zoom, per-digit toggle, hover thumbnail
url = save_projection_interactive(
    scores[:, :2], labels, exp_name, tag,
    images=X_pixels,              # (N, 784) for hover thumbnails (optional)
    component_images=components,  # (K, 784) shown in sidebar (optional)
    component_labels=[...],
    xlabel="PC 1", ylabel="PC 2",
)

# Clickable component grid — click to magnify, metadata in overlay
url = save_components_interactive(
    components, exp_name, tag,
    labels=[f"PC {i+1}" for i in range(K)],
    metadata=[{"evr": ..., "cumulative_evr": ...} for i in range(K)],
)
```

Both functions embed all data as JSON directly in the HTML, so the pages are
fully self-contained (no server needed).

### Post-experiment regeneration

```bash
uv run python projects/component-analysis/scripts/gen_viz_exp1.py
uv run python projects/component-analysis/scripts/gen_viz_exp2.py
```

These read `pca_results_expN.pkl` / `mcca_results_expN.pkl` from disk and
regenerate all plots. Save these at the end of `__main__` to enable this:

```python
with open(Path(__file__).parent / f"pca_results_{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({...}, f)
```

Both pkl files are gitignored. `results.jsonl` is the only committed output.

### Adding a new plot

1. If it's a one-off: write to `scripts/tmp/`, call `save_svg` + `save_figures_page` directly.
2. If it belongs to an existing experiment: add to `gen_viz_expN.py` and append to `figures`.
3. If the plot type is reusable: add a pure function to `lib/viz.py` (returns SVG URL)
   or `lib/interactive.py` (returns HTML URL).

---

## Lib API summary

### `lib/pca.py`

```python
result = fit_pca(X, n_components)          # (N,D) → PCAResult
scores = project(X, result)                # (N,D) → (N, n_components)
X_hat  = reconstruct(scores, result)       # (N, n_components) → (N, D)
mse    = reconstruction_error(X, result)   # float
```

### `lib/cca.py`

```python
result    = fit_mcca(Xs, n_components, reg, seed)   # list[(N_k,D)] → MCCAResult
scores_k  = project_class(X, k, result)             # class k → (N, n_components)
scores    = project_all(Xs, result)                 # list → list of scores
mean_corr = canonical_correlations(scores_list)     # (n_components,)
```

### `lib/viz.py` — static SVG plots (all return SVG URL)

```python
save_component_images(components, exp_name, tag, n_show, ...)
save_component_comparison(components_dict, exp_name, tag, n_show, ...)
save_projection_2d(scores, labels, exp_name, tag, ...)
save_projection_grid(scores_list, labels_list, titles, exp_name, tag, ...)
save_variance_curve(global_evr, exp_name, tag, per_class_evr, ...)
save_eigenvalue_curve(eigenvalues, exp_name, tag, ...)
save_canonical_correlation_heatmap(corr_matrix, exp_name, tag, ...)
```

All use `shared_lib.media.save_svg` internally and return the SVG URL.

### `lib/interactive.py` — interactive HTML pages (all return HTML URL)

```python
save_projection_interactive(scores, labels, exp_name, tag,
    images=None,             # (N, 784) hover thumbnails
    component_images=None,   # (K, 784) sidebar
    component_labels=None,   # list of K strings
    xlabel=..., ylabel=..., title=...,
)
save_components_interactive(components, exp_name, tag,
    labels=None,    # list of K strings
    metadata=None,  # list of K dicts shown in click overlay
    title=...,
)
```

Data is embedded as JSON in the HTML — pages are fully self-contained.

### `shared_lib.html` — page assembly

```python
from shared_lib.html import save_figures_page, save_flat_grid, save_html

save_figures_page(name, title, figures)  # figures = [(caption, svg_url), ...]
save_flat_grid(name, title, figures)     # same but no per-figure detail links
save_html(name, html_str)                # upload arbitrary HTML string
```

---

## Experiment Structure

```
module docstring    — algorithm description, what changed, usage line
hyperparameters     — UPPER_SNAKE_CASE
data loading        — load_supervised_1d("mnist")
analysis            — fit_pca / fit_mcca (pure functions from lib/)
save results        — pca_results_expN.pkl / mcca_results_expN.pkl
visualize           — lib/viz.py functions
append_result()     — write to results.jsonl
```
