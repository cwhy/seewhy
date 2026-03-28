"""
exp5 — Spherical K-means (cosine similarity) on raw MNIST pixels.

Pixels are L2-normalised to unit vectors before any computation.
Assignment uses argmax of dot products (= argmin cosine distance on the sphere).
Centroid update: mean of assigned unit vectors, then re-normalise to unit norm.

This is "spherical K-means" — removes brightness bias; clusters by shape/texture.

Compare with:
  exp1 — same N_CODES=128, Euclidean distance

Usage:
    uv run python projects/online-kmeans/experiments5.py
"""

import json, time, sys, pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_image

sys.path.insert(0, str(Path(__file__).parent))
from lib.viz import save_centroid_grid, save_learning_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp5"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_CODES    = 128
MAX_EPOCHS = 30
EVAL_EVERY = 5
SEED       = 42

N_MNIST_CLASSES = 10
JSONL = Path(__file__).parent / "results.jsonl"


# ── Utilities ──────────────────────────────────────────────────────────────────

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

def decode_prototypes(params):
    """Return centroids (unit sphere) de-normalised for display, shaped (N_CODES, 28, 28)."""
    c = np.array(params["centroids"])              # (K, 784), unit norm
    # Scale to [0, 255] for display
    c_min, c_max = c.min(1, keepdims=True), c.max(1, keepdims=True)
    c_disp = (c - c_min) / np.where(c_max - c_min > 0, c_max - c_min, 1.0) * 255.0
    return c_disp.reshape(N_CODES, 28, 28)


# ── Normalisation ──────────────────────────────────────────────────────────────

@jax.jit
def l2_normalise(X):
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    return X / jnp.where(norms > 0, norms, 1.0)


# ── Core GPU functions ─────────────────────────────────────────────────────────

@jax.jit
def spherical_lloyd_step(centroids, X_norm):
    """
    One Lloyd's step on the unit sphere.

    centroids: (K, D) unit vectors
    X_norm:    (N, D) unit vectors

    Assignment:  argmax dot product (= argmin cosine distance)
    Update:      mean of assigned unit vectors, re-normalised
    """
    dots   = X_norm @ centroids.T                              # (N, K) cosine sims
    assign = jnp.argmax(dots, axis=1)                         # (N,)

    sums   = jnp.zeros_like(centroids).at[assign].add(X_norm) # (K, D)
    counts = jnp.zeros(N_CODES).at[assign].add(1.0)           # (K,)
    alive  = counts > 0

    means  = jnp.where(alive[:, None],
                       sums / jnp.where(alive[:, None], counts[:, None], 1.0),
                       centroids)
    # Re-normalise to unit sphere
    norms  = jnp.linalg.norm(means, axis=1, keepdims=True)
    new_c  = means / jnp.where(norms > 0, norms, 1.0)
    return new_c, assign, counts, dots


# ── K-means++ initialisation on unit sphere ────────────────────────────────────

def kmeans_plus_plus(X_norm, n_clusters, key):
    """K-means++ with cosine distance: 1 - dot product (for unit vectors)."""
    idx    = jax.random.randint(key, (), 0, len(X_norm))
    center = X_norm[idx]
    # cosine distance = 1 - similarity; use as selection weight
    min_cd = 1.0 - (X_norm @ center)              # (N,)
    centers = [center]
    for _ in range(1, n_clusters):
        key, subkey = jax.random.split(key)
        p   = jnp.clip(min_cd, 0.0, None)
        p   = p / p.sum()
        idx = jax.random.choice(subkey, len(X_norm), p=p)
        center = X_norm[idx]
        centers.append(center)
        new_cd = 1.0 - (X_norm @ center)
        min_cd = jnp.minimum(min_cd, new_cd)
    return jnp.stack(centers)    # (K, D) — already unit norm (rows of X_norm)


# ── Label accuracy using cosine assignment ────────────────────────────────────

def label_accuracy(centroids, X_norm, Y_te):
    dots   = np.array(X_norm @ centroids.T)    # (N, K)
    assign = np.argmax(dots, axis=1)

    label_probs = np.zeros((N_CODES, N_MNIST_CLASSES))
    for c in range(N_CODES):
        mask = assign == c
        if mask.sum() > 0:
            label_probs[c] = np.bincount(Y_te[mask], minlength=N_MNIST_CLASSES) / mask.sum()
        else:
            label_probs[c] = 1.0 / N_MNIST_CLASSES

    majority = label_probs.argmax(axis=1)
    hard_acc = float((majority[assign] == Y_te).mean())

    # Bayes: weight by cosine similarity
    sim = np.clip(dots, 0, None)
    sim_sum = sim.sum(1, keepdims=True)
    p_c = sim / np.where(sim_sum > 0, sim_sum, 1.0)
    soft_acc = float(((p_c @ label_probs).argmax(1) == Y_te).mean())

    return hard_acc, soft_acc, label_probs


# ── Training ───────────────────────────────────────────────────────────────────

def train(centroids, X_norm, X_norm_te, Y_te, key):
    history = {
        "match_loss": [], "match_loss_eval": [],
        "cluster_usage": [], "label_acc_hard": [], "label_acc_soft": [],
    }
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_norm))
        Xs   = X_norm[perm]

        centroids, assign, counts, dots = spherical_lloyd_step(centroids, Xs)
        assign_np = np.array(assign)
        counts_np = np.array(counts)

        # Dead-cluster reinit
        dead = np.where(counts_np == 0)[0]
        if len(dead):
            key, subkey = jax.random.split(key)
            ridx      = jax.random.randint(subkey, (len(dead),), 0, len(X_norm))
            centroids = centroids.at[dead].set(X_norm[np.array(ridx)])

        # Train loss: mean cosine distance = 1 - mean cosine similarity
        dots_np   = np.array(dots)
        train_cos = float(dots_np[np.arange(len(Xs)), assign_np].mean())
        train_loss = 1.0 - train_cos    # cosine distance ∈ [0, 2]
        usage      = np.bincount(assign_np, minlength=N_CODES)
        n_active   = int((usage > 0).sum())
        history["match_loss"].append(train_loss)
        history["cluster_usage"].append(usage.tolist())

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            dots_te   = np.array(X_norm_te @ centroids.T)
            ea        = np.argmax(dots_te, axis=1)
            eval_loss = float(1.0 - dots_te[np.arange(len(X_norm_te)), ea].mean())
            hard_acc, soft_acc, _ = label_accuracy(centroids, X_norm_te, Y_te)
            history["match_loss_eval"].append((epoch, eval_loss))
            history["label_acc_hard"].append((epoch, hard_acc))
            history["label_acc_soft"].append((epoch, soft_acc))
            eval_str = (f"  eval_cos_dist={eval_loss:.4f}"
                        f"  lbl_hard={hard_acc:.1%}  lbl_bayes={soft_acc:.1%}")

        logging.info(f"epoch {epoch:3d}  cos_dist={train_loss:.4f}  cos_sim={train_cos:.4f}"
                     f"  active={n_active}/{N_CODES}  dead_reinit={len(dead)}{eval_str}"
                     f"  {time.perf_counter()-t0:.1f}s")

    return centroids, history, time.perf_counter() - t0


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = l2_normalise(data.X.reshape(data.n_samples,           784).astype(jnp.float32))
    X_test  = l2_normalise(data.X_test.reshape(data.n_test_samples, 784).astype(jnp.float32))
    Y_test  = np.array(data.y_test).astype(np.int32)
    logging.info(f"Data normalised to unit sphere.")

    key = jax.random.PRNGKey(SEED)
    logging.info("K-means++ (cosine) initialisation …")
    key, subkey = jax.random.split(key)
    centroids = kmeans_plus_plus(X_train, N_CODES, subkey)
    logging.info(f"Initialised {N_CODES} centroids  ({centroids.size:,} params)")

    centroids, history, elapsed = train(centroids, X_train, X_test, Y_test, key)

    hist_path = Path(__file__).parent / f"history_{EXP_NAME}.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    params = {"centroids": np.array(centroids)}
    params_path = Path(__file__).parent / f"params_{EXP_NAME}.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(params, f)

    save_learning_curves(history, EXP_NAME,
                         subtitle=f"Spherical K-means (cosine)  N_CODES={N_CODES}")
    save_centroid_grid(decode_prototypes(params), EXP_NAME, tag="final")

    dots_te   = np.array(X_test @ centroids.T)
    final_a   = np.argmax(dots_te, axis=1)
    final_l   = float(1.0 - dots_te[np.arange(len(X_test)), final_a].mean())
    hard_acc, soft_acc, label_probs = label_accuracy(centroids, X_test, Y_test)

    np.save(Path(__file__).parent / f"label_probs_{EXP_NAME}.npy", label_probs)

    append_result({
        "experiment": EXP_NAME,
        "name": f"spherical-kmeans N_CODES={N_CODES}",
        "n_params": int(centroids.size), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_match_l2": round(final_l, 6),
        "final_label_acc_hard": round(hard_acc, 6),
        "final_label_acc_soft": round(soft_acc, 6),
        "n_codes": N_CODES, "batch_size": "full",
    })
    logging.info(f"Done. cos_dist={final_l:.4f}  lbl_hard={hard_acc:.1%}"
                 f"  lbl_bayes={soft_acc:.1%}  {elapsed:.1f}s")
