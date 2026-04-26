"""
exp36 — Linear Probe + k-NN Classification on Frozen Variations Encoder

Post-hoc evaluation: freeze encoder from exp33/34/35, train a linear head
(W: 2048→10, b: 10) on CIFAR-10 labels, report test accuracy.
Also run k-NN (k=1, k=5) using sklearn.

No new model training — just probe quality of learned representations.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/variations/experiments36.py
"""

import json, time, sys, pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

sys.path.insert(0, str(Path(__file__).parent))
from lib.logging_utils import setup_logging
from experiments35 import encode, LATENT_DIM, IMG_FLAT, N_CAT

EXP_NAME = "exp36"

# ── Probe hyperparameters ──────────────────────────────────────────────────────

PROBE_EPOCHS = 100
PROBE_BATCH  = 1024
PROBE_LR     = 1e-3

EXPS_TO_PROBE = ["exp33", "exp35", "exp34"]

JSONL = Path(__file__).parent / "results.jsonl"

# ── Utilities ──────────────────────────────────────────────────────────────────

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

def load_params(exp_name):
    pkl_path = Path(__file__).parent / f"params_{exp_name}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

# ── Encoding ───────────────────────────────────────────────────────────────────

def encode_all(params, X_raw):
    """Encode full dataset in chunks. X_raw: (N, 3072) float32, 0-255."""
    parts = []
    for i in range(0, len(X_raw), 512):
        batch = jnp.array(X_raw[i:i+512]) / 255.0
        parts.append(np.array(encode(params, batch)))
    return np.concatenate(parts, 0)  # (N, LATENT_DIM)

# ── Linear probe ───────────────────────────────────────────────────────────────

def linear_probe(E_tr, Y_tr, E_te, Y_te):
    """Train a linear probe on frozen embeddings. Returns (train_acc, test_acc)."""
    n_tr = (len(E_tr) // PROBE_BATCH) * PROBE_BATCH
    E_b  = jnp.array(E_tr[:n_tr].reshape(-1, PROBE_BATCH, LATENT_DIM))
    Y_b  = jnp.array(Y_tr[:n_tr].reshape(-1, PROBE_BATCH))

    # cosine decay schedule: 1e-3 → 1e-4 over PROBE_EPOCHS * batches_per_epoch
    num_steps   = PROBE_EPOCHS * E_b.shape[0]
    schedule    = optax.cosine_decay_schedule(PROBE_LR, num_steps, alpha=0.1)
    probe_opt   = optax.adam(schedule)
    W           = jnp.zeros((LATENT_DIM, 10))
    b           = jnp.zeros(10)
    probe_state = probe_opt.init((W, b))

    def probe_step(carry, inputs):
        (W, b), state = carry
        e, y = inputs
        def ce(wb):
            return optax.softmax_cross_entropy_with_integer_labels(e @ wb[0] + wb[1], y).mean()
        loss, grads = jax.value_and_grad(ce)((W, b))
        updates, new_state = probe_opt.update(grads, state, (W, b))
        return (optax.apply_updates((W, b), updates), new_state), loss

    @jax.jit
    def probe_epoch(W, b, state):
        (W, b), state = jax.lax.scan(probe_step, ((W, b), state), (E_b, Y_b))[0]
        return W, b, state

    for ep in range(PROBE_EPOCHS):
        W, b, probe_state = probe_epoch(W, b, probe_state)
        if (ep + 1) % 20 == 0:
            logits_tr = jnp.array(E_tr[:n_tr]) @ W + b
            tr_acc = float((jnp.argmax(logits_tr, 1) == jnp.array(Y_tr[:n_tr])).mean())
            logging.info(f"  probe ep {ep+1}/{PROBE_EPOCHS}  train_acc={tr_acc:.4f}")

    logits_tr = jnp.array(E_tr) @ W + b
    logits_te = jnp.array(E_te) @ W + b
    train_acc = float((jnp.argmax(logits_tr, 1) == jnp.array(Y_tr)).mean())
    test_acc  = float((jnp.argmax(logits_te, 1) == jnp.array(Y_te)).mean())
    return train_acc, test_acc

# ── k-NN classification ────────────────────────────────────────────────────────

def knn_accuracy(E_tr, Y_tr, E_te, Y_te, k):
    """k-NN on CPU using sklearn."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(E_tr, Y_tr)
    return float(knn.score(E_te, Y_te))

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_logging(EXP_NAME)
    logging.info(f"=== {EXP_NAME}: linear probe + k-NN on frozen encoder ===")

    # Load CIFAR-10
    logging.info("Loading CIFAR-10...")
    ds = load_supervised_image("cifar10")
    X_tr = ds.X.reshape(ds.n_samples,           IMG_FLAT).astype(np.float32)  # (50000, 3072)
    Y_tr = np.array(ds.y,      dtype=np.int32)                                # (50000,)
    X_te = ds.X_test.reshape(ds.n_test_samples, IMG_FLAT).astype(np.float32)  # (10000, 3072)
    Y_te = np.array(ds.y_test, dtype=np.int32)                                # (10000,)
    logging.info(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

    for exp_src in EXPS_TO_PROBE:
        logging.info(f"\n--- Probing {exp_src} ---")

        # 1. Load params
        params = load_params(exp_src)
        logging.info(f"  Loaded params_{exp_src}.pkl")

        # 2. Encode all images
        t0 = time.perf_counter()
        logging.info("  Encoding train set...")
        E_tr = encode_all(params, X_tr)  # (50000, 2048)
        logging.info("  Encoding test set...")
        E_te = encode_all(params, X_te)  # (10000, 2048)
        t_enc = time.perf_counter() - t0
        logging.info(f"  Encoding done in {t_enc:.1f}s  E_tr={E_tr.shape}")

        # 3. Linear probe
        logging.info(f"  Training linear probe ({PROBE_EPOCHS} epochs, cosine LR, batch={PROBE_BATCH})...")
        t1 = time.perf_counter()
        probe_tr_acc, probe_te_acc = linear_probe(E_tr, Y_tr, E_te, Y_te)
        t_probe = time.perf_counter() - t1
        logging.info(f"  Linear probe: train={probe_tr_acc:.4f}  test={probe_te_acc:.4f}  ({t_probe:.1f}s)")

        # 4. k-NN
        logging.info("  Running k-NN (k=1)...")
        knn1 = knn_accuracy(E_tr, Y_tr, E_te, Y_te, k=1)
        logging.info(f"  k=1: {knn1:.4f}")

        logging.info("  Running k-NN (k=5)...")
        knn5 = knn_accuracy(E_tr, Y_tr, E_te, Y_te, k=5)
        logging.info(f"  k=5: {knn5:.4f}")

        # 5. Save result
        row = {
            "experiment":        EXP_NAME,
            "source_exp":        exp_src,
            "probe_acc_train":   probe_tr_acc,
            "probe_acc_test":    probe_te_acc,
            "knn1_acc":          knn1,
            "knn5_acc":          knn5,
            "probe_epochs":      PROBE_EPOCHS,
            "probe_lr":          PROBE_LR,
            "probe_batch":       PROBE_BATCH,
            "latent_dim":        LATENT_DIM,
            "time_enc_s":        t_enc,
            "time_probe_s":      t_probe,
        }
        append_result(row)
        logging.info(f"  Result appended to results.jsonl")

    logging.info(f"\n=== {EXP_NAME} complete ===")

if __name__ == "__main__":
    main()
