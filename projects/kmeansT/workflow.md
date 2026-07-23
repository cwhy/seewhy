# Experiment Workflow — kmeansT

**Concept**: cluster *features* (pixels) rather than samples. For MNIST
28×28, each of the 784 pixels is treated as a point in R^N described by
its activation value across N training samples. K-means groups pixels that
co-activate together.

---

## Directory Layout

```
projects/kmeansT/
├── workflow.md                  # This file
├── feature-clustering.py        # Main experiment script
└── latest.svg                   # Local copy of last generated SVG (gitignored)
```

---

## Invariant Rules

1. **Always `uv run python`** — never raw `python` or `python3`.

2. **Throwaway / one-off scripts go in `scripts/tmp/`** inside this
   directory. Never use `python -c "..."` inline.

3. **No bash polling loops** — read a log file once or write a Python
   script.

---

## Running Experiments

```bash
# Feature clustering across K ∈ {10,16,32,48,64} and 6 sample subsets
uv run python projects/kmeansT/feature-clustering.py
```

Output:
- SVG uploaded to R2 (or saved locally to `outputs/yy-mm-dd/`) via
  `shared_lib.media.save_media`
- HTML wrapper page uploaded via `shared_lib.html.save_html`
- Local copy written to `projects/kmeansT/latest.svg`

---

## Upload Utilities

All outputs go through `shared_lib`:

```python
from shared_lib.media import save_media
from shared_lib.html  import save_html

# Upload a raw SVG string
url = save_media("my_file.svg", svg_str.encode(), "image/svg+xml")

# Upload an HTML page
url = save_html("my_page", html_str)
```

Both try R2 first and fall back to `outputs/yy-mm-dd/` locally.
See [component-analysis/workflow.md](../component-analysis/workflow.md)
for the full shared_lib API (save_svg, save_figures_page, etc.).

---

## Varying the Experiment

Key constants at the top of `feature-clustering.py`:

| Constant    | Default                  | Meaning                        |
|-------------|--------------------------|--------------------------------|
| `K_VALUES`  | `[10, 16, 32, 48, 64]`   | Cluster counts to sweep        |
| `N_ITER`    | `40`                     | Max K-means iterations         |
| `CELL`      | `12`                     | SVG px per pixel cell          |

Add new subsets in `make_subsets()` — returns `(label, X_subset)` pairs
where `X_subset.T` is the feature matrix passed to k-means.
