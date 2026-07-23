"""
exp4 — Full-batch K-means (Lloyd's), N_CODES=512.

Same as exp1 but with 4× more clusters to see how label purity scales.

Baseline: classic Lloyd's algorithm.  Each epoch:
  1. Compute all N assignments to nearest centroid (single GPU matrix multiply).
  2. Recompute each centroid as the mean of its assigned samples.
  3. Reinit any dead clusters from a random training sample.

K-means++ initialisation (GPU-accelerated via JAX).

Everything stays on GPU as JAX arrays; host only receives numpy copies at
logging/eval checkpoints.

Usage:
    uv run python projects/online-kmeans/experiments1.py
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

EXP_NAME = "exp4"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_CODES    = 512
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
    """Squared L2 distances: (n, d) × (m, d) → (n, m)."""
    return (A**2).sum(1, keepdims=True) + (B**2).sum(1) - 2.0 * (A @ B.T)


@jax.jit
def lloyd_step(centroids, X):
    """One Lloyd's epoch: assign + recompute means.  Returns (new_centroids, assign, counts)."""
    dists  = sq_dists(X, centroids)                       # (N, K)
    assign = jnp.argmin(dists, axis=1)                    # (N,)
    sums   = jnp.zeros_like(centroids).at[assign].add(X)  # (K, D)
    counts = jnp.zeros(N_CODES).at[assign].add(1.0)       # (K,)
    alive  = counts > 0
    new_c  = jnp.where(alive[:, None],
                       sums / jnp.where(alive[:, None], counts[:, None], 1.0),
                       centroids)
    return new_c, assign, counts


# ── K-means++ initialisation (GPU) ────────────────────────────────────────────

def kmeans_plus_plus(X, n_clusters, key):
    """Incremental K-means++: O(K·N·D) total, fully on-device."""
    idx = jax.random.randint(key, (), 0, len(X))
    center = X[idx]
    min_d2 = ((X - center) ** 2).sum(1)                  # (N,)

    centers = [center]
    for i in range(1, n_clusters):
        key, subkey = jax.random.split(key)
        p   = jnp.clip(min_d2, 0.0, None)
        p   = p / p.sum()
        idx = jax.random.choice(subkey, len(X), p=p)
        center = X[idx]
        centers.append(center)
        new_d2 = ((X - center) ** 2).sum(1)
        min_d2 = jnp.minimum(min_d2, new_d2)

    return jnp.stack(centers)    # (K, D)


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
    soft_acc = float((( p_c @ label_probs).argmax(1) == Y_te).mean())

    return hard_acc, soft_acc, label_probs


# ── Training ───────────────────────────────────────────────────────────────────

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

        centroids, assign, counts = lloyd_step(centroids, Xs)
        assign_np = np.array(assign)
        counts_np = np.array(counts)

        # Dead-cluster reinit
        dead = np.where(counts_np == 0)[0]
        if len(dead):
            key, subkey = jax.random.split(key)
            ridx = jax.random.randint(subkey, (len(dead),), 0, len(X_tr))
            centroids = centroids.at[dead].set(X_tr[np.array(ridx)])

        # Metrics
        dists_tr  = np.array(sq_dists(Xs, centroids))
        usage     = np.bincount(assign_np, minlength=N_CODES)
        n_active  = int((usage > 0).sum())
        train_loss = float(np.sqrt(np.maximum(dists_tr[np.arange(len(Xs)), assign_np], 0)).mean())
        history["match_loss"].append(train_loss)
        history["cluster_usage"].append(usage.tolist())

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_d  = np.array(sq_dists(X_te, centroids))
            ea      = np.argmin(eval_d, axis=1)
            el      = float(np.sqrt(np.maximum(eval_d[np.arange(len(X_te)), ea], 0)).mean())
            hard_acc, soft_acc, _ = label_accuracy(centroids, X_te, Y_te)
            history["match_loss_eval"].append((epoch, el))
            history["label_acc_hard"].append((epoch, hard_acc))
            history["label_acc_soft"].append((epoch, soft_acc))
            eval_str = (f"  eval_loss={el:.2f}"
                        f"  lbl_hard={hard_acc:.1%}  lbl_bayes={soft_acc:.1%}")

        logging.info(f"epoch {epoch:3d}  train_loss={train_loss:.2f}"
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
    logging.info(f"History saved → {hist_path}")

    params = {"centroids": np.array(centroids)}
    params_path = Path(__file__).parent / f"params_{EXP_NAME}.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    logging.info(f"Params saved → {params_path}")

    save_learning_curves(history, EXP_NAME,
                         subtitle=f"Lloyd's K-means  N_CODES={N_CODES}")
    save_centroid_grid(params["centroids"], EXP_NAME, tag="final")

    final_d  = np.array(sq_dists(X_test, centroids))
    final_a  = np.argmin(final_d, axis=1)
    final_l  = float(np.sqrt(np.maximum(final_d[np.arange(len(X_test)), final_a], 0)).mean())
    hard_acc, soft_acc, label_probs = label_accuracy(centroids, X_test, Y_test)

    np.save(Path(__file__).parent / f"label_probs_{EXP_NAME}.npy", label_probs)

    append_result({
        "experiment": EXP_NAME,
        "name": f"lloyd-kmeans N_CODES={N_CODES}",
        "n_params": int(centroids.size), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_match_l2": round(final_l, 4),
        "final_label_acc_hard": round(hard_acc, 6),
        "final_label_acc_soft": round(soft_acc, 6),
        "n_codes": N_CODES, "batch_size": "full",
    })
    logging.info(f"Done. loss={final_l:.2f}  lbl_hard={hard_acc:.1%}"
                 f"  lbl_bayes={soft_acc:.1%}  {elapsed:.1f}s")
