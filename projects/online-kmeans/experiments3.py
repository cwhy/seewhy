"""
exp3 — Small mini-batch K-means (BATCH_SIZE=32) on raw MNIST pixels.

Identical algorithm to exp2 (count-based incremental centroid update via JAX
immutable scatter), but uses a much smaller batch.  Represents the "more
online" end of the spectrum without the pure-Python per-sample bottleneck.

Note: true BATCH_SIZE=1 requires Numba or Cython to be tractable in Python
(Python loop overhead ~1ms/sample → ~98 min for 100 epochs × 60K samples).
BATCH_SIZE=32 preserves the online qualitative behaviour while running fast
via vectorised GPU kernels.

Compare with:
  exp1 — full-batch Lloyd's
  exp2 — mini-batch K-means (BATCH_SIZE=256)

Usage:
    uv run python projects/online-kmeans/experiments3.py
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

EXP_NAME   = "exp3"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_CODES    = 128
BATCH_SIZE = 32
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
    """Return centroids shaped (N_CODES, 28, 28) for visualisation."""
    return np.array(params["centroids"]).reshape(N_CODES, 28, 28)


# ── Core GPU functions ─────────────────────────────────────────────────────────

@jax.jit
def sq_dists(A, B):
    return (A**2).sum(1, keepdims=True) + (B**2).sum(1) - 2.0 * (A @ B.T)


@jax.jit
def batch_step(centroids, counts, batch):
    """One mini-batch step.  Returns (new_centroids, new_counts, assign)."""
    dists    = sq_dists(batch, centroids)
    assign   = jnp.argmin(dists, axis=1)
    b_counts = jnp.zeros(N_CODES).at[assign].add(1.0)
    b_sums   = jnp.zeros_like(centroids).at[assign].add(batch)

    new_counts = counts + b_counts
    lr     = jnp.where(b_counts > 0, b_counts / new_counts, 0.0)[:, None]
    b_means = jnp.where(b_counts[:, None] > 0,
                        b_sums / jnp.where(b_counts[:, None] > 0, b_counts[:, None], 1.0),
                        centroids)
    new_centroids = centroids + lr * (b_means - centroids)
    return new_centroids, new_counts, assign


# ── K-means++ initialisation (GPU) ────────────────────────────────────────────

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


# ── Label accuracy (host) ──────────────────────────────────────────────────────

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


# ── Training ───────────────────────────────────────────────────────────────────

def train(centroids, X_tr, X_te, Y_te, key):
    history = {
        "match_loss": [], "match_loss_eval": [],
        "cluster_usage": [], "label_acc_hard": [], "label_acc_soft": [],
    }
    counts = jnp.ones(N_CODES)
    t0 = time.perf_counter()
    num_batches = len(X_tr) // BATCH_SIZE

    for epoch in range(MAX_EPOCHS):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_tr))
        Xs   = X_tr[perm]

        epoch_usage = np.zeros(N_CODES, dtype=np.int64)
        epoch_loss  = 0.0

        for b in range(num_batches):
            batch = Xs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            centroids, counts, assign = batch_step(centroids, counts, batch)
            assign_np    = np.array(assign)
            epoch_usage += np.bincount(assign_np, minlength=N_CODES)

            batch_np = np.array(batch)
            c_np     = np.array(centroids)
            d2       = ((batch_np - c_np[assign_np])**2).sum(1)
            epoch_loss += float(np.sqrt(np.maximum(d2, 0)).mean())

        # Reinit dead clusters
        dead = np.where(epoch_usage == 0)[0]
        if len(dead):
            key, subkey = jax.random.split(key)
            ridx      = jax.random.randint(subkey, (len(dead),), 0, len(X_tr))
            centroids = centroids.at[dead].set(X_tr[np.array(ridx)])
            counts    = counts.at[dead].set(1.0)

        epoch_loss /= num_batches
        n_active    = int((epoch_usage > 0).sum())
        history["match_loss"].append(epoch_loss)
        history["cluster_usage"].append(epoch_usage.tolist())

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_d = np.array(sq_dists(X_te, centroids))
            ea     = np.argmin(eval_d, axis=1)
            el     = float(np.sqrt(np.maximum(eval_d[np.arange(len(X_te)), ea], 0)).mean())
            hard_acc, soft_acc, _ = label_accuracy(centroids, X_te, Y_te)
            history["match_loss_eval"].append((epoch, el))
            history["label_acc_hard"].append((epoch, hard_acc))
            history["label_acc_soft"].append((epoch, soft_acc))
            eval_str = (f"  eval_loss={el:.2f}"
                        f"  lbl_hard={hard_acc:.1%}  lbl_bayes={soft_acc:.1%}")

        logging.info(f"epoch {epoch:3d}  train_loss={epoch_loss:.2f}"
                     f"  active={n_active}/{N_CODES}  dead_reinit={len(dead)}{eval_str}"
                     f"  {time.perf_counter()-t0:.1f}s")

    return centroids, history, time.perf_counter() - t0


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(jnp.float32)
    Y_test  = np.array(data.y_test).astype(np.int32)

    key = jax.random.PRNGKey(SEED)
    logging.info("K-means++ initialisation …")
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
                         subtitle=f"Small mini-batch K-means  N_CODES={N_CODES}  BS={BATCH_SIZE}")
    save_centroid_grid(params["centroids"], EXP_NAME, tag="final")

    final_d = np.array(sq_dists(X_test, centroids))
    final_a = np.argmin(final_d, axis=1)
    final_l = float(np.sqrt(np.maximum(final_d[np.arange(len(X_test)), final_a], 0)).mean())
    hard_acc, soft_acc, label_probs = label_accuracy(centroids, X_test, Y_test)

    np.save(Path(__file__).parent / f"label_probs_{EXP_NAME}.npy", label_probs)

    append_result({
        "experiment": EXP_NAME,
        "name": f"small-minibatch-kmeans N_CODES={N_CODES} BS={BATCH_SIZE}",
        "n_params": int(centroids.size), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_match_l2": round(final_l, 4),
        "final_label_acc_hard": round(hard_acc, 6),
        "final_label_acc_soft": round(soft_acc, 6),
        "n_codes": N_CODES, "batch_size": BATCH_SIZE,
    })
    logging.info(f"Done. loss={final_l:.2f}  lbl_hard={hard_acc:.1%}"
                 f"  lbl_bayes={soft_acc:.1%}  {elapsed:.1f}s")
