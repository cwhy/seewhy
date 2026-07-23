"""
exp6 — Prototype nearest-neighbour classifier from data-PCA exemplars.

For each digit class d and each of the first N_COMPONENTS data-PCA components
we build two prototype images:
  - pos prototype: mean of the TOP_K training images by loading score u_k[i]
  - neg prototype: mean of the BOTTOM_K training images by loading score u_k[i]

This gives  10 × N_COMPONENTS × 2  prototypes in total.

At test time each image is classified by the label of its nearest prototype
(L2 distance in pixel space).

Requires: data_pca_results_exp5.pkl produced by exp5.

Usage:
    uv run python projects/component-analysis/scripts/run_experiments.py exp6
"""

import json, time, sys, pickle, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_1d
from shared_lib.html import save_figures_page

sys.path.insert(0, str(Path(__file__).parent))
from lib.viz import save_component_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME     = "exp6"
N_COMPONENTS = 16   # components per digit to use
TOP_K        = 4    # exemplars averaged per side
JSONL        = Path(__file__).parent / "results.jsonl"
PKL_EXP5     = Path(__file__).parent / "data_pca_results_exp5.pkl"


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def build_prototypes(per_digit: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
        prototypes : (P, 784)  where P = 10 * N_COMPONENTS * 2
        proto_labels: (P,)     digit label of each prototype
    """
    proto_list, label_list = [], []
    for d in range(10):
        r = per_digit[d]
        X_d = r["X_d"]       # (N_d, 784)
        U   = r["U"]         # (N_d, N_COMPONENTS)
        for k in range(N_COMPONENTS):
            scores = U[:, k]
            order  = np.argsort(scores)
            neg_mean = X_d[order[:TOP_K]].mean(axis=0)
            pos_mean = X_d[order[-TOP_K:]].mean(axis=0)
            proto_list.append(neg_mean)
            proto_list.append(pos_mean)
            label_list.append(d)
            label_list.append(d)
    return np.stack(proto_list), np.array(label_list)


def nn_predict(X_test: np.ndarray, prototypes: np.ndarray,
               proto_labels: np.ndarray, batch: int = 500) -> np.ndarray:
    """Batched L2 nearest-neighbour prediction."""
    preds = np.empty(len(X_test), dtype=np.int32)
    for i in range(0, len(X_test), batch):
        Xb = X_test[i:i + batch]                         # (B, 784)
        # squared L2: ||x - p||^2 = ||x||^2 + ||p||^2 - 2 x·p^T
        dists = (
            (Xb ** 2).sum(axis=1, keepdims=True)
            + (prototypes ** 2).sum(axis=1)
            - 2 * Xb @ prototypes.T
        )                                                  # (B, P)
        preds[i:i + batch] = proto_labels[dists.argmin(axis=1)]
    return preds


if __name__ == "__main__":
    t0 = time.perf_counter()

    # ── Load exp5 results ─────────────────────────────────────────────────────
    logging.info(f"Loading {PKL_EXP5}...")
    with open(PKL_EXP5, "rb") as f:
        exp5 = pickle.load(f)
    per_digit = exp5["per_digit"]

    # ── Build prototypes ──────────────────────────────────────────────────────
    logging.info(f"Building prototypes ({N_COMPONENTS} components × 2 sides × 10 digits)...")
    prototypes, proto_labels = build_prototypes(per_digit)
    logging.info(f"  {len(prototypes)} prototypes  shape={prototypes.shape}")

    # ── Load test set ─────────────────────────────────────────────────────────
    logging.info("Loading MNIST test set...")
    data  = load_supervised_1d("mnist")
    X_te  = np.array(data.X_test, dtype=np.float64)
    y_te  = np.array(data.y_test, dtype=np.int32)

    # ── Nearest-neighbour classification ──────────────────────────────────────
    logging.info("Classifying test set...")
    preds = nn_predict(X_te, prototypes, proto_labels)

    overall_acc = float((preds == y_te).mean())
    logging.info(f"  Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    per_digit_acc = {}
    for d in range(10):
        mask = y_te == d
        acc_d = float((preds[mask] == y_te[mask]).mean())
        per_digit_acc[d] = acc_d
        logging.info(f"  Digit {d}: {acc_d:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared_lib.media import save_svg

    conf = np.zeros((10, 10), dtype=np.int32)
    for true, pred in zip(y_te, preds):
        conf[true, pred] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    fontsize=7, color="black" if conf[i, j] < conf.max() * 0.6 else "white")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{EXP_NAME} — Confusion matrix  (acc={overall_acc*100:.2f}%)")
    plt.tight_layout()
    conf_url = save_svg(f"{EXP_NAME}_confusion", fig)
    plt.close(fig)
    logging.info(f"  confusion matrix → {conf_url}")

    # ── Prototype grid (one row per digit) ────────────────────────────────────
    # Reshape to (10, N_COMPONENTS*2, 784), show first 16 protos per digit
    figures = []
    figures.append(("Confusion matrix", conf_url))

    for d in range(10):
        start = d * N_COMPONENTS * 2
        protos_d = prototypes[start:start + N_COMPONENTS * 2]   # (32, 784)
        url = save_component_images(
            protos_d, EXP_NAME, f"protos_digit{d}",
            n_show=N_COMPONENTS * 2,
            title=f"Digit {d} prototypes — neg/pos pairs for components 1–{N_COMPONENTS}",
        )
        figures.append((f"Digit {d} prototypes", url))

    from shared_lib.html import save_figures_page
    page_url = save_figures_page(
        f"{EXP_NAME}_nn_classifier",
        f"{EXP_NAME} — Data-PCA Prototype NN Classifier",
        figures,
    )
    logging.info(f"Figures page → {page_url}")

    # ── Log result ─────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    append_result({
        "experiment": EXP_NAME,
        "name": f"Data-PCA NN classifier  n_comp={N_COMPONENTS} top_k={TOP_K}",
        "n_prototypes": len(prototypes),
        "overall_acc": round(overall_acc, 5),
        "per_digit_acc": {str(d): round(per_digit_acc[d], 5) for d in range(10)},
        "page_url": page_url,
        "time_s": round(elapsed, 1),
    })
    logging.info(f"Done. {elapsed:.1f}s")
