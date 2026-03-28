"""
Visualization utilities for component-analysis experiments.

All functions produce a single matplotlib figure, save it as an SVG via
shared_lib.media.save_svg, and return the URL.  Callers collect
(caption, url) pairs and pass them to save_figures_page / save_flat_grid
from shared_lib.html to build the assembled HTML page.

Assumes 28×28 images (MNIST). Pass img_shape to override.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_svg


# ── Component image grid ───────────────────────────────────────────────────────

def _normalise_component(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def save_component_images(
    components: np.ndarray,
    exp_name: str,
    tag: str,
    n_show: int = 20,
    img_shape: tuple[int, int] = (28, 28),
    title: str | None = None,
) -> str:
    """
    Display the first n_show rows of components as grayscale images.
    Returns the SVG URL.
    """
    n = min(n_show, len(components))
    cols = min(10, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            img = _normalise_component(components[i]).reshape(img_shape)
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_title(f"PC {i+1}", fontsize=7)
        ax.axis("off")

    fig.suptitle(title or f"{exp_name} — {tag} components", fontsize=10)
    plt.tight_layout()
    url = save_svg(f"{exp_name}_{tag}_components", fig)
    plt.close(fig)
    logging.info(f"  components grid → {url}")
    return url


def save_component_comparison(
    components_dict: dict[str, np.ndarray],
    exp_name: str,
    tag: str,
    n_show: int = 10,
    img_shape: tuple[int, int] = (28, 28),
) -> str:
    """
    Show components from multiple sources side-by-side.
    Each row = one source; columns = components 1..n_show.
    Returns the SVG URL.
    """
    sources = list(components_dict.items())
    n_rows = len(sources)
    n_cols = n_show

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.5))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for r, (label, comps) in enumerate(sources):
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(comps):
                img = _normalise_component(comps[c]).reshape(img_shape)
                ax.imshow(img, cmap="gray", interpolation="nearest")
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=90, labelpad=4)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{exp_name} — {tag}", fontsize=10)
    plt.tight_layout()
    url = save_svg(f"{exp_name}_{tag}_comparison", fig)
    plt.close(fig)
    logging.info(f"  comparison      → {url}")
    return url


# ── 2D projection scatter ──────────────────────────────────────────────────────

_DIGIT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def save_projection_2d(
    scores: np.ndarray,
    labels: np.ndarray,
    exp_name: str,
    tag: str,
    alpha: float = 0.3,
    n_max: int = 5000,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    title: str | None = None,
) -> str:
    """2D scatter of (N, ≥2) scores coloured by digit label. Returns SVG URL."""
    if len(scores) > n_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(scores), n_max, replace=False)
        scores, labels = scores[idx], labels[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    for d in range(10):
        mask = labels == d
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   c=[_DIGIT_COLORS[d]], label=str(d),
                   alpha=alpha, s=8, linewidths=0)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title or f"{exp_name} — {tag}")
    ax.legend(title="digit", ncol=2, fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    url = save_svg(f"{exp_name}_{tag}_scatter2d", fig)
    plt.close(fig)
    logging.info(f"  2D scatter       → {url}")
    return url


def save_projection_grid(
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    titles: list[str],
    exp_name: str,
    tag: str,
    alpha: float = 0.3,
    n_max: int = 2000,
) -> str:
    """Grid of 2D scatter plots. Returns SVG URL."""
    K = len(scores_list)
    cols = min(5, K)
    rows = (K + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.array(axes).reshape(rows, cols)

    for i, (scores, labels, ttl) in enumerate(zip(scores_list, labels_list, titles)):
        ax = axes[i // cols, i % cols]
        if len(scores) > n_max:
            rng = np.random.default_rng(i)
            idx = rng.choice(len(scores), n_max, replace=False)
            scores, labels = scores[idx], labels[idx]
        for d in np.unique(labels):
            mask = labels == d
            ax.scatter(scores[mask, 0], scores[mask, 1],
                       c=[_DIGIT_COLORS[int(d)]], label=str(d),
                       alpha=alpha, s=6, linewidths=0)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.15)

    for i in range(K, rows * cols):
        axes[i // cols, i % cols].axis("off")

    fig.suptitle(f"{exp_name} — {tag}", fontsize=11)
    plt.tight_layout()
    url = save_svg(f"{exp_name}_{tag}_grid", fig)
    plt.close(fig)
    logging.info(f"  projection grid → {url}")
    return url


# ── Explained variance / eigenvalue curves ────────────────────────────────────

def save_variance_curve(
    global_evr: np.ndarray,
    exp_name: str,
    tag: str,
    per_class_evr: dict[int, np.ndarray] | None = None,
    title: str | None = None,
) -> str:
    """Cumulative explained variance curve. Returns SVG URL."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    if per_class_evr:
        for d, evr in per_class_evr.items():
            ax.plot(np.arange(1, len(evr) + 1), evr,
                    color=_DIGIT_COLORS[d], alpha=0.5, linewidth=1)
    ax.plot(np.arange(1, len(global_evr) + 1), global_evr,
            color="black", linewidth=2, label="global")
    ax.set_xlabel("Component"); ax.set_ylabel("EVR")
    ax.set_title("Per-component EVR")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    if per_class_evr:
        for d, evr in per_class_evr.items():
            ax.plot(np.arange(1, len(evr) + 1), evr.cumsum(),
                    color=_DIGIT_COLORS[d], alpha=0.5, linewidth=1, label=str(d))
    ax.plot(np.arange(1, len(global_evr) + 1), global_evr.cumsum(),
            color="black", linewidth=2, label="global")
    ax.set_xlabel("n_components"); ax.set_ylabel("Cumulative EVR")
    ax.set_title("Cumulative explained variance")
    ax.grid(True, alpha=0.3)
    ax.legend(title="digit", ncol=3, fontsize=7)

    fig.suptitle(title or f"{exp_name} — {tag} variance", fontsize=11)
    plt.tight_layout()
    url = save_svg(f"{exp_name}_{tag}_variance", fig)
    plt.close(fig)
    logging.info(f"  variance curve  → {url}")
    return url


def save_eigenvalue_curve(
    eigenvalues: np.ndarray,
    exp_name: str,
    tag: str,
    title: str | None = None,
) -> str:
    """MCCA eigenvalue scree plot. Returns SVG URL."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, len(eigenvalues) + 1), eigenvalues, color="steelblue")
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title or f"{exp_name} — {tag} eigenvalues")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    url = save_svg(f"{exp_name}_{tag}_eigenvalues", fig)
    plt.close(fig)
    logging.info(f"  eigenvalue scree → {url}")
    return url


# ── Pairwise canonical correlation heatmap ────────────────────────────────────

def save_data_pca_grid(
    X_d: np.ndarray,
    U: np.ndarray,
    mean: np.ndarray,
    exp_name: str,
    tag: str,
    n_show: int = 8,
    n_exemplars: int = 4,
    img_shape: tuple[int, int] = (28, 28),
    title: str | None = None,
) -> str:
    """
    Visualize data-dimension PCA components for one digit class.

    For each of the first n_show components u_k in U (N, K):
      - Left n_exemplars cols: images with the most POSITIVE loading in u_k
      - Centre col: pixel loading image = (X_d - mean)^T u_k  reshaped to img_shape
      - Right n_exemplars cols: images with the most NEGATIVE loading in u_k

    X_d : (N, D) raw images for one digit class
    U   : (N, K) left singular vectors (one column per component)
    mean: (D,)   per-pixel mean used during SVD
    Returns the SVG URL.
    """
    K = min(n_show, U.shape[1])
    Xc = X_d - mean                               # (N, D)
    pixel_loadings = Xc.T @ U[:, :K]             # (D, K)

    n_cols = n_exemplars + 1 + n_exemplars
    fig, axes = plt.subplots(K, n_cols, figsize=(n_cols * 1.3, K * 1.4))
    axes = np.array(axes).reshape(K, n_cols)

    center_col = n_exemplars  # index of the loading-image column

    for k in range(K):
        scores_k = U[:, k]                        # (N,)
        order = np.argsort(scores_k)              # ascending

        # Positive exemplars (highest scores) — rightmost columns
        pos_idx = order[-(n_exemplars):][::-1]
        neg_idx = order[:n_exemplars]

        for e, idx in enumerate(neg_idx):
            ax = axes[k, e]
            ax.imshow(X_d[idx].reshape(img_shape), cmap="gray", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_title(f"neg {e+1}", fontsize=6)

        # Centre: pixel loading image
        ax = axes[k, center_col]
        loading_img = _normalise_component(pixel_loadings[:, k]).reshape(img_shape)
        ax.imshow(loading_img, cmap="RdBu_r", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"PC {k+1}", fontsize=7, rotation=0, labelpad=22, va="center")
        if k == 0:
            ax.set_title("loading", fontsize=6)

        for e, idx in enumerate(pos_idx):
            ax = axes[k, center_col + 1 + e]
            ax.imshow(X_d[idx].reshape(img_shape), cmap="gray", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_title(f"pos {e+1}", fontsize=6)

    fig.suptitle(title or f"{exp_name} — {tag} data-PCA", fontsize=10)
    plt.tight_layout()
    url = save_svg(f"{exp_name}_{tag}_data_pca_grid", fig)
    plt.close(fig)
    logging.info(f"  data pca grid   → {url}")
    return url


def save_canonical_correlation_heatmap(
    corr_matrix: np.ndarray,
    exp_name: str,
    tag: str,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str | None = None,
) -> str:
    """Heatmap of a correlation or similarity matrix. Returns SVG URL."""
    fig, ax = plt.subplots(figsize=(max(5, corr_matrix.shape[1] * 0.6),
                                     max(4, corr_matrix.shape[0] * 0.6)))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8)

    ax.set_title(title or f"{exp_name} — {tag}")
    plt.tight_layout()
    url = save_svg(f"{exp_name}_{tag}_corrmap", fig)
    plt.close(fig)
    logging.info(f"  corr heatmap    → {url}")
    return url
