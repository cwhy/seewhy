"""
kmeansT — Exp 2: Lloyd's K-means on pixel-cluster features.

Two-stage pipeline:
  Stage 1 (kmeansT)  — cluster the 784 *pixels* into K_PIXEL groups using
                        their activation patterns across training samples.
                        Each pixel gets a cluster id; each image becomes a
                        K_PIXEL-dim vector of per-cluster mean pixel values.

  Stage 2 (Lloyd's)  — cluster the 60k *samples* in that K_PIXEL-dim space
                        with N_CODES centroids, identical to online-kmeans exp1.

Centroid decoding: a K_PIXEL-dim centroid is expanded back to 28×28 by
assigning each pixel the centroid value of its pixel-cluster.  This gives
a piecewise-constant reconstruction that can be visualised with the
online-kmeans centroid grid.

Usage:
    uv run python projects/kmeansT/experiments2.py
"""

import importlib.util, json, logging, pickle, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT   = Path(__file__).resolve().parents[2]
HERE   = Path(__file__).resolve().parent
OKMEANS = Path(__file__).resolve().parents[1] / "online-kmeans"

sys.path.insert(0, str(ROOT))
from shared_lib.datasets import load_supervised_image
from shared_lib.html import save_figures_page

sys.path.insert(0, str(HERE))
from lib.kmeans import fit as kmeans_fit, extract_features

# Load online-kmeans viz directly to avoid lib package name collision
_viz_spec = importlib.util.spec_from_file_location(
    "okmeans_viz", OKMEANS / "lib" / "viz.py"
)
_viz = importlib.util.module_from_spec(_viz_spec)
_viz_spec.loader.exec_module(_viz)
save_centroid_grid  = _viz.save_centroid_grid
save_learning_curves = _viz.save_learning_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp2"

# ── Hyperparameters ──────────────────────────────────────────────────────────
K_PIXEL    = 48     # pixel clusters (kmeansT stage) — best from exp1
N_CODES    = 128    # sample clusters (Lloyd's stage)
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


# ── Centroid decode: K_PIXEL-dim → 28×28 pixel image ────────────────────────

def decode_centroids(feat_centroids: np.ndarray, pixel_assignments: np.ndarray) -> np.ndarray:
    """
    Expand feature-space centroids back to pixel space.

    feat_centroids:    (N_CODES, K_PIXEL) — centroid in feature space
    pixel_assignments: (784,)             — cluster id for each pixel

    Returns (N_CODES, 784) float32 in [0, 255] for save_centroid_grid.
    """
    out = np.empty((len(feat_centroids), 784), dtype=np.float32)
    for k in range(K_PIXEL):
        mask = pixel_assignments == k
        out[:, mask] = feat_centroids[:, k:k+1]   # broadcast over pixels
    return out * 255.0   # scale to [0, 255] for viz


# ── JAX Lloyd's step (mirrors online-kmeans exp1) ────────────────────────────

@jax.jit
def sq_dists(A, B):
    """Squared L2 distances: (n, d) × (m, d) → (n, m)."""
    return (A**2).sum(1, keepdims=True) + (B**2).sum(1) - 2.0 * (A @ B.T)


def lloyd_step(centroids, X, n_codes):
    dists  = sq_dists(X, centroids)                         # (N, K)
    assign = jnp.argmin(dists, axis=1)                      # (N,)
    sums   = jnp.zeros_like(centroids).at[assign].add(X)   # (K, D)
    counts = jnp.zeros(n_codes).at[assign].add(1.0)        # (K,)
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


# ── Label accuracy ────────────────────────────────────────────────────────────

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


# ── Training loop ─────────────────────────────────────────────────────────────

def train(centroids, X_tr, X_te, Y_te, key):
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

        # Dead-cluster reinit
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
                     f"  active={n_active}/{N_CODES}  dead_reinit={len(dead)}{eval_str}"
                     f"  {time.perf_counter()-t0:.1f}s")

    return centroids, history, time.perf_counter() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")

    # ── Stage 1: learn pixel clusters ────────────────────────────────────
    logging.info(f"Loading MNIST …")
    data    = load_supervised_image("mnist")
    X_train_px = data.X.reshape(data.n_samples,           -1).astype(np.float32) / 255.0
    X_test_px  = data.X_test.reshape(data.n_test_samples, -1).astype(np.float32) / 255.0
    Y_test     = np.array(data.y_test, dtype=np.int32)

    logging.info(f"Stage 1: kmeansT — learning {K_PIXEL} pixel clusters …")
    X_T = X_train_px.T                           # (784, N)
    pixel_assignments, _ = kmeans_fit(X_T, K_PIXEL, n_iter=KMEANS_ITER, seed=SEED)
    counts = np.bincount(pixel_assignments, minlength=K_PIXEL)
    logging.info(f"  pixel cluster sizes  min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")

    # ── Stage 2: extract K_PIXEL-dim features ────────────────────────────
    logging.info(f"Extracting {K_PIXEL}-dim features …")
    F_train = extract_features(X_train_px, pixel_assignments, K_PIXEL)   # (N, K_PIXEL)
    F_test  = extract_features(X_test_px,  pixel_assignments, K_PIXEL)   # (N_te, K_PIXEL)
    logging.info(f"  F_train {F_train.shape}  F_test {F_test.shape}")

    # Convert to JAX arrays on GPU
    X_tr = jnp.array(F_train)
    X_te = jnp.array(F_test)

    # ── Stage 2: Lloyd's K-means in feature space ─────────────────────────
    key = jax.random.PRNGKey(SEED)
    logging.info(f"K-means++ initialisation  (N_CODES={N_CODES}) …")
    key, subkey = jax.random.split(key)
    centroids = kmeans_plus_plus(X_tr, N_CODES, subkey)
    logging.info(f"  Initialised {N_CODES} centroids  shape={centroids.shape}")

    centroids, history, elapsed = train(centroids, X_tr, X_te, Y_test, key)

    # ── Save params / history ─────────────────────────────────────────────
    here = Path(__file__).parent

    hist_path = here / f"history_{EXP_NAME}.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    logging.info(f"History  → {hist_path}")

    params = {
        "centroids":        np.array(centroids),          # (N_CODES, K_PIXEL)
        "pixel_assignments": pixel_assignments,            # (784,)
        "K_pixel":          K_PIXEL,
    }
    params_path = here / f"params_{EXP_NAME}.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    logging.info(f"Params   → {params_path}")

    # ── Eval ──────────────────────────────────────────────────────────────
    final_d  = np.array(sq_dists(X_te, centroids))
    final_a  = np.argmin(final_d, axis=1)
    final_l  = float(np.sqrt(np.maximum(final_d[np.arange(len(F_test)), final_a], 0)).mean())
    hard_acc, soft_acc, label_probs = label_accuracy(centroids, X_te, Y_test)
    np.save(here / f"label_probs_{EXP_NAME}.npy", label_probs)

    # ── Visualize ─────────────────────────────────────────────────────────
    # Decode centroids from K_PIXEL-dim back to 28×28 pixel images
    centroids_px = decode_centroids(np.array(centroids), pixel_assignments)   # (N_CODES, 784) in [0,255]

    figures = []

    url = save_centroid_grid(centroids_px, EXP_NAME, tag="final")
    figures.append(("Cluster centroids (decoded to 28×28)", url))

    url = save_learning_curves(
        history, EXP_NAME,
        subtitle=f"Lloyd's on kmeansT features  K_pixel={K_PIXEL}  N_CODES={N_CODES}",
    )
    figures.append(("Learning curves", url))

    page_url = save_figures_page(
        f"kmeansT_{EXP_NAME}",
        f"kmeansT Exp 2 — Lloyd's in {K_PIXEL}-dim pixel-cluster space",
        figures,
    )
    logging.info(f"Figures page → {page_url}")

    # ── Log result ────────────────────────────────────────────────────────
    append_result({
        "experiment": EXP_NAME,
        "name": f"lloyd-kmeans on kmeansT features  K_pixel={K_PIXEL}  N_CODES={N_CODES}",
        "K_pixel": K_PIXEL, "n_codes": N_CODES,
        "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_match_l2": round(final_l, 6),
        "final_label_acc_hard": round(hard_acc, 6),
        "final_label_acc_soft": round(soft_acc, 6),
    })
    logging.info(f"Done.  loss={final_l:.4f}  lbl_hard={hard_acc:.1%}"
                 f"  lbl_bayes={soft_acc:.1%}  {elapsed:.1f}s")
