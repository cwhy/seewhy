"""
kmeansT — Exp 3: Second-round kmeansT using centroid profiles.

After exp2 we have N_CODES=128 sample-cluster centroids in pixel space (128, 784).
Each pixel i now has a 128-dim profile: its value in each cluster prototype.
We use *that* matrix (784, 128) as the feature matrix for a new round of kmeansT.

Intuition: pixels that respond similarly across all 128 cluster prototypes are
grouped together — this is a data-driven, structure-aware pixel grouping rather
than one based on raw co-activation statistics.

Pipeline:
  Round 1 (exp2):  kmeansT(X.T)           → pixel_assignments_1 (784,)
                   Lloyd's(F_1)            → centroids_1 (128, 48), decoded to (128, 784)

  Round 2 (exp3):  kmeansT(centroids_1.T) → pixel_assignments_2 (784,)
                   Lloyd's(F_2)            → centroids_2 (128, K2)

Comparison: visualise pixel maps and centroid grids from both rounds side by side.

Usage:
    uv run python projects/kmeansT/experiments3.py
"""

import importlib.util, json, logging, pickle, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import colorsys, io

ROOT    = Path(__file__).resolve().parents[2]
HERE    = Path(__file__).resolve().parent
OKMEANS = Path(__file__).resolve().parents[1] / "online-kmeans"

sys.path.insert(0, str(ROOT))
from shared_lib.datasets import load_supervised_image
from shared_lib.html import save_figures_page
from shared_lib.media import save_media, save_matplotlib_figure

sys.path.insert(0, str(HERE))
from lib.kmeans import fit as kmeans_fit, extract_features

_viz_spec = importlib.util.spec_from_file_location("okmeans_viz", OKMEANS / "lib" / "viz.py")
_viz = importlib.util.module_from_spec(_viz_spec)
_viz_spec.loader.exec_module(_viz)
save_centroid_grid   = _viz.save_centroid_grid
save_learning_curves = _viz.save_learning_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp3"

# ── Hyperparameters ──────────────────────────────────────────────────────────
K_PIXEL2   = 48     # pixel clusters for round 2
N_CODES    = 128    # sample clusters
MAX_EPOCHS = 30
EVAL_EVERY = 5
KMEANS_ITER = 40
SEED       = 42

N_MNIST_CLASSES = 10
RESULTS_PATH = Path(__file__).parent / "results.jsonl"
# ────────────────────────────────────────────────────────────────────────────


def append_result(row: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")


# ── Centroid decode/encode ────────────────────────────────────────────────────

def decode_to_pixels(feat_centroids: np.ndarray, pixel_assignments: np.ndarray, K_pixel: int) -> np.ndarray:
    """Expand (N_CODES, K_pixel) → (N_CODES, 784) in [0, 1]."""
    out = np.empty((len(feat_centroids), 784), dtype=np.float32)
    for k in range(K_pixel):
        mask = pixel_assignments == k
        out[:, mask] = feat_centroids[:, k:k+1]
    return out   # [0, 1] range


# ── JAX Lloyd's (same as exp2) ────────────────────────────────────────────────

@jax.jit
def sq_dists(A, B):
    return (A**2).sum(1, keepdims=True) + (B**2).sum(1) - 2.0 * (A @ B.T)


def lloyd_step(centroids, X, n_codes):
    dists  = sq_dists(X, centroids)
    assign = jnp.argmin(dists, axis=1)
    sums   = jnp.zeros_like(centroids).at[assign].add(X)
    counts = jnp.zeros(n_codes).at[assign].add(1.0)
    alive  = counts > 0
    new_c  = jnp.where(alive[:, None],
                       sums / jnp.where(alive[:, None], counts[:, None], 1.0),
                       centroids)
    return new_c, assign, counts


def kmeans_plus_plus(X, n_clusters, key):
    idx    = jax.random.randint(key, (), 0, len(X))
    center = X[idx]
    min_d2 = ((X - center) ** 2).sum(1)
    centers = [center]
    for _ in range(1, n_clusters):
        key, subkey = jax.random.split(key)
        p   = jnp.clip(min_d2, 0.0, None)
        p   = p / p.sum()
        idx = jax.random.choice(subkey, len(X), p=p)
        center = X[idx]
        centers.append(center)
        min_d2 = jnp.minimum(min_d2, ((X - center) ** 2).sum(1))
    return jnp.stack(centers)


def label_accuracy(centroids, X_te, Y_te):
    dists  = np.array(sq_dists(X_te, centroids))
    assign = np.argmin(dists, axis=1)
    label_probs = np.zeros((N_CODES, N_MNIST_CLASSES))
    for c in range(N_CODES):
        mask = assign == c
        if mask.sum() > 0:
            label_probs[c] = np.bincount(Y_te[mask], minlength=N_MNIST_CLASSES) / mask.sum()
        else:
            label_probs[c] = 1.0 / N_MNIST_CLASSES
    majority = label_probs.argmax(axis=1)
    hard_acc = float((majority[assign] == Y_te).mean())
    scale = dists.mean()
    p_c   = np.exp(-dists / (scale + 1e-8))
    p_c  /= p_c.sum(axis=1, keepdims=True)
    soft_acc = float(((p_c @ label_probs).argmax(1) == Y_te).mean())
    return hard_acc, soft_acc, label_probs


def train(centroids, X_tr, X_te, Y_te, key, subtitle=""):
    history = {
        "match_loss": [], "match_loss_eval": [],
        "cluster_usage": [], "label_acc_hard": [], "label_acc_soft": [],
    }
    t0 = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_tr))
        Xs   = X_tr[perm]
        centroids, assign, counts = lloyd_step(centroids, Xs, N_CODES)
        assign_np = np.array(assign)
        counts_np = np.array(counts)
        dead = np.where(counts_np == 0)[0]
        if len(dead):
            key, subkey = jax.random.split(key)
            ridx = jax.random.randint(subkey, (len(dead),), 0, len(X_tr))
            centroids = centroids.at[dead].set(X_tr[np.array(ridx)])
        dists_tr   = np.array(sq_dists(Xs, centroids))
        usage      = np.bincount(assign_np, minlength=N_CODES)
        n_active   = int((usage > 0).sum())
        train_loss = float(np.sqrt(np.maximum(dists_tr[np.arange(len(Xs)), assign_np], 0)).mean())
        history["match_loss"].append(train_loss)
        history["cluster_usage"].append(usage.tolist())
        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_d   = np.array(sq_dists(X_te, centroids))
            ea       = np.argmin(eval_d, axis=1)
            el       = float(np.sqrt(np.maximum(eval_d[np.arange(len(X_te)), ea], 0)).mean())
            hard_acc, soft_acc, _ = label_accuracy(centroids, X_te, Y_te)
            history["match_loss_eval"].append((epoch, el))
            history["label_acc_hard"].append((epoch, hard_acc))
            history["label_acc_soft"].append((epoch, soft_acc))
            eval_str = f"  eval_loss={el:.4f}  lbl_hard={hard_acc:.1%}  lbl_bayes={soft_acc:.1%}"
        logging.info(f"epoch {epoch:3d}  train_loss={train_loss:.4f}"
                     f"  active={n_active}/{N_CODES}  dead={len(dead)}{eval_str}"
                     f"  {time.perf_counter()-t0:.1f}s")
    return centroids, history, time.perf_counter() - t0


# ── Pixel-map comparison plot ─────────────────────────────────────────────────

def save_pixel_map_comparison(
    px_assign_1: np.ndarray, K1: int, label1: str,
    px_assign_2: np.ndarray, K2: int, label2: str,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, assignments, K, label in [
        (axes[0], px_assign_1, K1, label1),
        (axes[1], px_assign_2, K2, label2),
    ]:
        colors = np.array([colorsys.hsv_to_rgb(i / K, 0.7, 0.85) for i in range(K)])
        img = colors[assignments].reshape(28, 28, 3)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle("Pixel cluster maps: round 1 vs round 2", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    url = save_media("kmeansT_exp3_pixel_maps.svg", buf, "image/svg+xml")
    plt.close(fig)
    return url


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")

    # ── Load exp2 results ─────────────────────────────────────────────────
    logging.info("Loading exp2 params …")
    params2 = pickle.load(open(HERE / "params_exp2.pkl", "rb"))
    feat_centroids_1 = params2["centroids"]        # (N_CODES, K_pixel1)  in [0,1]
    px_assign_1      = params2["pixel_assignments"] # (784,)
    K_pixel1         = params2["K_pixel"]

    logging.info(f"  exp2 centroids: {feat_centroids_1.shape}  K_pixel1={K_pixel1}")

    # ── Round 2 pixel features: centroid profile matrix ───────────────────
    # Decode feature centroids → pixel-space (N_CODES, 784) then transpose
    C_px = decode_to_pixels(feat_centroids_1, px_assign_1, K_pixel1)  # (128, 784) in [0,1]
    pixel_feature_matrix = C_px.T   # (784, 128) — each pixel × N_CODES centroid values
    logging.info(f"Round-2 pixel feature matrix: {pixel_feature_matrix.shape}")

    # ── kmeansT round 2 ───────────────────────────────────────────────────
    logging.info(f"Round 2 kmeansT — clustering 784 pixels in {N_CODES}-dim centroid space …")
    px_assign_2, _ = kmeans_fit(
        pixel_feature_matrix, K_PIXEL2, n_iter=KMEANS_ITER, seed=SEED
    )
    counts2 = np.bincount(px_assign_2, minlength=K_PIXEL2)
    logging.info(f"  pixel cluster sizes  min={counts2.min()}  max={counts2.max()}  mean={counts2.mean():.1f}")

    # ── Load MNIST, extract round-2 features ──────────────────────────────
    logging.info("Loading MNIST …")
    data       = load_supervised_image("mnist")
    X_train_px = data.X.reshape(data.n_samples,           -1).astype(np.float32) / 255.0
    X_test_px  = data.X_test.reshape(data.n_test_samples, -1).astype(np.float32) / 255.0
    Y_test     = np.array(data.y_test, dtype=np.int32)

    logging.info(f"Extracting {K_PIXEL2}-dim round-2 features …")
    F2_train = extract_features(X_train_px, px_assign_2, K_PIXEL2)
    F2_test  = extract_features(X_test_px,  px_assign_2, K_PIXEL2)
    logging.info(f"  F2_train {F2_train.shape}  F2_test {F2_test.shape}")

    # ── Lloyd's round 2 ───────────────────────────────────────────────────
    X_tr = jnp.array(F2_train)
    X_te = jnp.array(F2_test)

    key = jax.random.PRNGKey(SEED)
    logging.info(f"K-means++ init (N_CODES={N_CODES}) …")
    key, subkey = jax.random.split(key)
    centroids2 = kmeans_plus_plus(X_tr, N_CODES, subkey)

    centroids2, history2, elapsed = train(centroids2, X_tr, X_te, Y_test, key)

    # ── Save ──────────────────────────────────────────────────────────────
    hist_path = HERE / f"history_{EXP_NAME}.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history2, f)

    params3 = {
        "centroids":         np.array(centroids2),
        "pixel_assignments": px_assign_2,
        "K_pixel":           K_PIXEL2,
    }
    params_path = HERE / f"params_{EXP_NAME}.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(params3, f)
    logging.info(f"Saved params → {params_path}")

    # ── Eval ──────────────────────────────────────────────────────────────
    final_d  = np.array(sq_dists(X_te, centroids2))
    final_a  = np.argmin(final_d, axis=1)
    final_l  = float(np.sqrt(np.maximum(final_d[np.arange(len(F2_test)), final_a], 0)).mean())
    hard_acc, soft_acc, label_probs = label_accuracy(centroids2, X_te, Y_test)
    np.save(HERE / f"label_probs_{EXP_NAME}.npy", label_probs)

    # ── Visualize ─────────────────────────────────────────────────────────
    figures = []

    # Pixel map comparison
    url = save_pixel_map_comparison(
        px_assign_1, K_pixel1, f"Round 1 (K={K_pixel1}, raw pixels)",
        px_assign_2, K_PIXEL2, f"Round 2 (K={K_PIXEL2}, centroid profiles)",
    )
    figures.append(("Pixel cluster maps: round 1 vs round 2", url))
    logging.info(f"  pixel maps → {url}")

    # Round-2 centroid grid (decoded to 28×28)
    C2_px = decode_to_pixels(np.array(centroids2), px_assign_2, K_PIXEL2)  # (128, 784) [0,1]
    url = save_centroid_grid(C2_px * 255.0, EXP_NAME, tag="final")
    figures.append(("Round-2 cluster centroids (28×28)", url))

    # Load round-1 centroid grid for comparison
    C1_px = decode_to_pixels(feat_centroids_1, px_assign_1, K_pixel1)
    url_r1 = save_centroid_grid(C1_px * 255.0, "exp2", tag="r1_compare")
    figures.append(("Round-1 cluster centroids (28×28)", url_r1))

    # Learning curves
    url = save_learning_curves(
        history2, EXP_NAME,
        subtitle=f"Round 2 Lloyd's  K_pixel={K_PIXEL2}  N_CODES={N_CODES}",
    )
    figures.append(("Round-2 learning curves", url))

    page_url = save_figures_page(
        f"kmeansT_{EXP_NAME}",
        "kmeansT Exp 3 — Round-2 kmeansT via centroid profiles",
        figures,
    )
    logging.info(f"Figures page → {page_url}")

    # ── Comparison summary ────────────────────────────────────────────────
    hist1 = pickle.load(open(HERE / "history_exp2.pkl", "rb"))
    acc1_hard = hist1["label_acc_hard"][-1][1] if hist1["label_acc_hard"] else None
    acc1_soft = hist1["label_acc_soft"][-1][1] if hist1["label_acc_soft"] else None

    append_result({
        "experiment": EXP_NAME,
        "name": f"round2-lloyd  K_pixel={K_PIXEL2}  N_CODES={N_CODES}",
        "K_pixel": K_PIXEL2, "n_codes": N_CODES,
        "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_match_l2": round(final_l, 6),
        "final_label_acc_hard": round(hard_acc, 6),
        "final_label_acc_soft": round(soft_acc, 6),
    })

    logging.info("── Comparison ───────────────────────────────────────────")
    logging.info(f"  Round 1 (exp2):  lbl_hard={acc1_hard:.1%}  lbl_bayes={acc1_soft:.1%}" if acc1_hard else "  Round 1: n/a")
    logging.info(f"  Round 2 (exp3):  lbl_hard={hard_acc:.1%}  lbl_bayes={soft_acc:.1%}")
    logging.info(f"  Elapsed: {elapsed:.1f}s")
