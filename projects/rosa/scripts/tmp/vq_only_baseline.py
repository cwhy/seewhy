"""
Phase 0.3 baseline — DiVeQ-MLP MNIST classifier (no ROSA).

Validates that the DiVeQ pipeline + recurrent input MLP + classifier head can
hit the standard MNIST accuracy ceiling. Provides the "neural without ROSA"
upper bound exp1 must beat.

Architecture:
    image (28×28) -> proj_image -> h_0
    for k in 1..16:  h_k = block(h_{k-1});  z_k = head_emit(h_k)
    DiVeQ each z_k against codebook (V=1024, D=64)
    mean-pool the 16 quantized embeddings -> classifier MLP -> 10-way logits

No streams, no indicators, no NO_OUTPUT class — those come in 0.4 and exp1.

Usage:
    uv run python projects/rosa/scripts/tmp/vq_only_baseline.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.random_utils import infinite_safe_keys_from_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Hyperparameters ───────────────────────────────────────────────────────────

DATASET      = "fashion_mnist"   # "mnist" or "fashion_mnist"
VOCAB_SIZE   = 1024
TOKEN_DIM    = 64
HIDDEN       = 256
K_SAMPLE     = 16
SIGMA        = 0.01
LAMBDA_CB    = 1.0
BATCH_SIZE   = 128
N_EPOCHS     = 50
LR           = 3e-4
SEED         = 42

# Reserve IDs 0 and 1 for indicator slots (unused in this baseline) so the
# codebook init is consistent with later experiments.
N_RESERVED   = 2
N_CONTENT    = VOCAB_SIZE - N_RESERVED


# ── Params ────────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    p = {
        # Recurrent input MLP
        "proj_image_W": n((784, HIDDEN), 784 ** -0.5),
        "proj_image_b": jnp.zeros(HIDDEN),
        "rms_pre_gamma": jnp.ones(HIDDEN),               # RMSNorm before each block step
        "block_W1":     n((HIDDEN, HIDDEN), HIDDEN ** -0.5),
        "block_b1":     jnp.zeros(HIDDEN),
        "block_W2":     n((HIDDEN, HIDDEN), HIDDEN ** -0.5),
        "block_b2":     jnp.zeros(HIDDEN),
        "emit_rms_gamma": jnp.ones(HIDDEN),              # RMSNorm before emit head
        "emit_W":       n((HIDDEN, TOKEN_DIM), HIDDEN ** -0.5),
        "emit_b":       jnp.zeros(TOKEN_DIM),
        # Codebook init: N(0, 1/√D) so |c| ≈ 1, matching RMSNorm-scaled z
        "codebook":     n((VOCAB_SIZE, TOKEN_DIM), TOKEN_DIM ** -0.5),
        # Classifier
        "cls_W1":       n((TOKEN_DIM, HIDDEN), TOKEN_DIM ** -0.5),
        "cls_b1":       jnp.zeros(HIDDEN),
        "cls_W2":       n((HIDDEN, 10), HIDDEN ** -0.5),
        "cls_b2":       jnp.zeros(10),
    }
    return p


def n_params(p) -> int:
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


# ── Forward ───────────────────────────────────────────────────────────────────

def rmsnorm(x, gamma, eps=1e-6):
    rms = jnp.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def run_input_mlp(p, x):
    """x: (B, 784) → z: (B, K_SAMPLE, TOKEN_DIM)."""
    h = jax.nn.gelu(x @ p["proj_image_W"] + p["proj_image_b"])
    zs = []
    for _ in range(K_SAMPLE):
        h_norm = rmsnorm(h, p["rms_pre_gamma"])
        h2 = jax.nn.gelu(h_norm @ p["block_W1"] + p["block_b1"])
        h  = h2 @ p["block_W2"] + p["block_b2"] + h           # post-norm residual
        z  = rmsnorm(h, p["emit_rms_gamma"]) @ p["emit_W"] + p["emit_b"]
        zs.append(z)
    return jnp.stack(zs, axis=1)                              # (B, K, D)


def diveq_quantize(z, codebook, key):
    """DiVeQ reparameterization. z: (..., D), codebook: (V, D).
    Returns (z_q: (..., D), ids: (..., int32))."""
    # Mask reserved indicator slots from argmin so this baseline only ever
    # picks content tokens.
    z_flat = z.reshape(-1, TOKEN_DIM)                         # (N, D)
    dists = ((z_flat[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)  # (N, V)
    mask = jnp.zeros(VOCAB_SIZE).at[:N_RESERVED].set(jnp.inf)
    dists_masked = dists + mask[None, :]
    ids_flat = jnp.argmin(dists_masked, axis=-1)              # (N,)
    c_star = codebook[ids_flat]                               # (N, D)

    d_tilde = jax.lax.stop_gradient(c_star - z_flat)
    v       = jax.random.normal(key, z_flat.shape) * SIGMA
    vd      = v + d_tilde
    direction = jax.lax.stop_gradient(
        vd / (jnp.linalg.norm(vd, axis=-1, keepdims=True) + 1e-8)
    )
    magnitude = jnp.linalg.norm(c_star - z_flat, axis=-1, keepdims=True)
    z_q_flat = z_flat + magnitude * direction

    z_q = z_q_flat.reshape(z.shape)
    ids = ids_flat.reshape(z.shape[:-1])
    return z_q, ids


def classify(p, z_q):
    """z_q: (B, K, D) → logits: (B, 10)."""
    pooled = z_q.mean(axis=1)                                 # (B, D)
    h = jax.nn.gelu(pooled @ p["cls_W1"] + p["cls_b1"])
    return h @ p["cls_W2"] + p["cls_b2"]


# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(p, x_batch, y_batch, key):
    z = run_input_mlp(p, x_batch)                             # (B, K, D)
    z_q, ids = diveq_quantize(z, p["codebook"], key)          # (B, K, D), (B, K)
    logits = classify(p, z_q)                                 # (B, 10)
    cls = optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()

    # Codebook loss: pulls codebook entries toward their assigned z's.
    c_star = p["codebook"][ids]                               # (B, K, D)
    cb_loss = ((jax.lax.stop_gradient(z) - c_star) ** 2).mean()

    return cls + LAMBDA_CB * cb_loss, (cls, cb_loss, ids)


# ── Training step ─────────────────────────────────────────────────────────────

def make_step_fn(optimizer):
    @jax.jit
    def step(p, opt_state, x_batch, y_batch, key):
        (loss, (cls, cb, ids)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, x_batch, y_batch, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss, cls, cb, ids
    return step


# ── Eval ──────────────────────────────────────────────────────────────────────

@jax.jit
def _eval_batch(p, x_batch):
    z = run_input_mlp(p, x_batch)
    z_q, ids = diveq_quantize(z, p["codebook"], jnp.zeros(2, dtype=jnp.uint32))
    logits = classify(p, z_q)
    return jnp.argmax(logits, axis=-1), ids


def eval_metrics(p, X, y):
    preds_all, ids_all = [], []
    for i in range(0, len(X), 256):
        bx = jnp.array(X[i:i + 256])
        preds, ids = _eval_batch(p, bx)
        preds_all.append(np.array(preds))
        ids_all.append(np.array(ids))
    preds = np.concatenate(preds_all)
    ids   = np.concatenate(ids_all)
    acc = float((preds == np.array(y)).mean())
    cb_used = int(len(np.unique(ids)))
    return acc, cb_used


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.info(f"JAX devices: {jax.devices()}")
    data = load_supervised_image(DATASET)
    X_train = np.array(data.X.reshape(data.n_samples,           784)) / 255.0
    X_test  = np.array(data.X_test.reshape(data.n_test_samples, 784)) / 255.0
    y_train = np.array(data.y)
    y_test  = np.array(data.y_test)
    logging.info(f"MNIST: train={X_train.shape}  test={X_test.shape}")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = init_params(key_gen)
    logging.info(f"params: {n_params(p):,}  V={VOCAB_SIZE}  D={TOKEN_DIM}  K={K_SAMPLE}")

    num_batches = len(X_train) // BATCH_SIZE
    schedule = optax.cosine_decay_schedule(LR, N_EPOCHS * num_batches, alpha=0.05)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(p)
    step_fn = make_step_fn(optimizer)

    rng = np.random.default_rng(SEED + 1)
    t0 = time.perf_counter()
    history = {"loss": [], "cls": [], "cb": [], "test_acc": [], "cb_used": []}

    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs = X_train[perm]
        ys = y_train[perm]
        ep_loss, ep_cls, ep_cb = 0.0, 0.0, 0.0
        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx = jnp.array(Xs[s:e])
            by = jnp.array(ys[s:e])
            key = next(key_gen).get()
            p, opt_state, loss, cls, cb, _ = step_fn(p, opt_state, bx, by, key)
            ep_loss += float(loss); ep_cls += float(cls); ep_cb += float(cb)

        ep_loss /= num_batches; ep_cls /= num_batches; ep_cb /= num_batches
        test_acc, cb_used = eval_metrics(p, X_test, y_test)
        history["loss"].append(ep_loss)
        history["cls"].append(ep_cls)
        history["cb"].append(ep_cb)
        history["test_acc"].append(test_acc)
        history["cb_used"].append(cb_used)

        logging.info(
            f"epoch {epoch:2d}  loss={ep_loss:.4f}  cls={ep_cls:.4f}  "
            f"cb={ep_cb:.4f}  test={test_acc:.4f}  cb_used={cb_used}/{N_CONTENT}  "
            f"{time.perf_counter() - t0:.1f}s"
        )

    elapsed = time.perf_counter() - t0
    final_acc, cb_used = eval_metrics(p, X_test, y_test)
    train_acc, _ = eval_metrics(p, X_train[:10_000], y_train[:10_000])

    logging.info(f"\nFINAL  test={final_acc:.4f}  train(10k)={train_acc:.4f}  "
                 f"cb_used={cb_used}/{N_CONTENT}  time={elapsed:.0f}s")

    out = Path(__file__).parent / f"vq_only_baseline_result.{DATASET}.json"
    with open(out, "w") as f:
        json.dump({
            "final_test_acc": final_acc,
            "final_train_acc": train_acc,
            "cb_used": cb_used,
            "n_params": n_params(p),
            "epochs": N_EPOCHS,
            "time_s": elapsed,
            "history": history,
            "hp": {
                "VOCAB_SIZE": VOCAB_SIZE, "TOKEN_DIM": TOKEN_DIM,
                "HIDDEN": HIDDEN, "K_SAMPLE": K_SAMPLE,
                "SIGMA": SIGMA, "LAMBDA_CB": LAMBDA_CB,
                "BATCH_SIZE": BATCH_SIZE, "LR": LR, "N_EPOCHS": N_EPOCHS,
            },
        }, f, indent=2)
    logging.info(f"Result saved to {out}")


if __name__ == "__main__":
    main()
