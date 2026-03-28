"""
exp7 — K-Subspaces clustering + nearest-subspace classification on MNIST.

Fits K-Subspaces unsupervised (no labels) for K in K_VALUES.
After fitting, each cluster is labelled by majority vote of training points.
Test images are classified by nearest subspace (min reconstruction error).

Usage:
    uv run python projects/component-analysis/scripts/run_experiments.py exp7
"""

import json, time, sys, pickle, logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_1d
from shared_lib.media import save_svg
from shared_lib.html import save_figures_page

sys.path.insert(0, str(Path(__file__).parent))
from lib.ksubspaces import fit_ksubspaces, predict
from lib.viz import save_component_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME     = "exp7"
K_VALUES     = [10, 20, 50, 100, 200]
N_COMPONENTS = 20    # subspace dimensionality (same for all K)
MAX_ITER     = 40
N_RESTARTS   = 1
JSONL        = Path(__file__).parent / "results.jsonl"


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def majority_labels(assignments: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """For each cluster, return the most common digit label."""
    labels = np.zeros(K, dtype=np.int32)
    for c in range(K):
        mask = assignments == c
        if mask.sum() == 0:
            labels[c] = -1
        else:
            labels[c] = np.bincount(y[mask], minlength=10).argmax()
    return labels


def purity(assignments: np.ndarray, y: np.ndarray, K: int) -> float:
    """Fraction of training points that agree with their cluster's majority label."""
    cluster_labels = majority_labels(assignments, y, K)
    correct = (cluster_labels[assignments] == y).sum()
    return float(correct) / len(y)


def confusion_matrix_fig(y_true, y_pred, title):
    conf = np.zeros((10, 10), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    acc = (y_true == y_pred).mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    fontsize=7,
                    color="white" if conf[i, j] > conf.max() * 0.6 else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}  (acc={acc*100:.2f}%)")
    plt.tight_layout()
    return fig, acc


if __name__ == "__main__":
    t0 = time.perf_counter()

    # ── Load data ─────────────────────────────────────────────────────────────
    logging.info("Loading MNIST...")
    data = load_supervised_1d("mnist")
    X_tr = np.array(data.X,      dtype=np.float64)
    y_tr = np.array(data.y,      dtype=np.int32)
    X_te = np.array(data.X_test, dtype=np.float64)
    y_te = np.array(data.y_test, dtype=np.int32)
    logging.info(f"Train {X_tr.shape}  Test {X_te.shape}")

    # ── Sweep K ───────────────────────────────────────────────────────────────
    run_results = {}
    figures     = []

    accs_train_purity = []
    accs_test         = []

    for K in K_VALUES:
        t_k = time.perf_counter()
        logging.info(f"\n── K={K} ({'─'*40})")

        result = fit_ksubspaces(
            X_tr, K, N_COMPONENTS,
            max_iter=MAX_ITER, n_restarts=N_RESTARTS, seed=42,
        )
        logging.info(
            f"  n_iter={result.n_iter}  "
            f"total_err={result.total_error:.1f}  "
            f"cluster sizes: min={np.bincount(result.assignments).min()}  "
            f"max={np.bincount(result.assignments).max()}"
        )

        # Majority-vote labelling
        cluster_labels = majority_labels(result.assignments, y_tr, K)
        train_purity   = purity(result.assignments, y_tr, K)
        logging.info(f"  train purity: {train_purity:.4f}")

        # Classify test set
        te_asgn, _ = predict(X_te, result)
        y_pred      = cluster_labels[te_asgn]
        test_acc    = float((y_pred == y_te).mean())
        logging.info(f"  test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

        accs_train_purity.append(train_purity)
        accs_test.append(test_acc)

        # Confusion matrix
        fig, _ = confusion_matrix_fig(y_te, y_pred, f"K={K} confusion")
        url = save_svg(f"{EXP_NAME}_K{K}_confusion", fig)
        plt.close(fig)
        figures.append((f"K={K}  acc={test_acc*100:.2f}%  purity={train_purity*100:.1f}%", url))

        # Component images for each cluster (first 20 clusters, first 8 components)
        n_show_clusters = min(K, 20)
        # Flatten: (n_show_clusters * N_COMPONENTS, D) — one row of N_COMPONENTS eigenimages per cluster
        comp_flat = result.components[:n_show_clusters].reshape(-1, X_tr.shape[1])
        url_comp = save_component_images(
            comp_flat, EXP_NAME, f"K{K}_components",
            n_show=min(n_show_clusters * N_COMPONENTS, 200),
            title=f"K={K} subspace components (first {n_show_clusters} clusters × {N_COMPONENTS} dims)",
        )
        figures.append((f"K={K} subspace components", url_comp))

        run_results[K] = {
            "train_purity": round(train_purity, 5),
            "test_acc":     round(test_acc, 5),
            "n_iter":       result.n_iter,
            "time_s":       round(time.perf_counter() - t_k, 1),
        }

    # ── Accuracy vs K plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_VALUES, [v * 100 for v in accs_test],
            marker="o", linewidth=2, label="Test accuracy")
    ax.plot(K_VALUES, [v * 100 for v in accs_train_purity],
            marker="s", linewidth=2, linestyle="--", label="Train purity")
    ax.set_xlabel("K (number of subspaces)")
    ax.set_ylabel("%")
    ax.set_title(f"{EXP_NAME} — K-Subspaces: accuracy & purity vs K  (n_comp={N_COMPONENTS})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for k, acc, pur in zip(K_VALUES, accs_test, accs_train_purity):
        ax.annotate(f"{acc*100:.1f}%", (k, acc * 100), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    plt.tight_layout()
    url_curve = save_svg(f"{EXP_NAME}_acc_vs_K", fig)
    plt.close(fig)
    figures.insert(0, ("Accuracy & purity vs K", url_curve))

    # ── Figures page ──────────────────────────────────────────────────────────
    page_url = save_figures_page(
        f"{EXP_NAME}_ksubspaces",
        f"{EXP_NAME} — K-Subspaces Classification (n_comp={N_COMPONENTS})",
        figures,
    )
    logging.info(f"\nFigures page → {page_url}")

    # ── Save pkl ──────────────────────────────────────────────────────────────
    with open(Path(__file__).parent / f"ksubspaces_results_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(run_results, f)

    # ── Log result ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    append_result({
        "experiment": EXP_NAME,
        "name": f"K-Subspaces sweep K={K_VALUES} n_comp={N_COMPONENTS}",
        "results_by_K": run_results,
        "best_K":  max(run_results, key=lambda k: run_results[k]["test_acc"]),
        "best_acc": max(r["test_acc"] for r in run_results.values()),
        "page_url": page_url,
        "time_s": round(elapsed, 1),
    })
    logging.info(f"Done. {elapsed:.1f}s")
