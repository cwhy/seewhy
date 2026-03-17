"""
Linear RNN for MNIST pixel-by-pixel classification.

Architecture
============
1. Learned embeddings  (one pixel at a time)
   - pos_emb  : (784, d)  — position lookup table
   - val_emb  : (K,   d)  — value lookup table (K=32 bins of 0-255)
   - emb_t = pos_emb[t] + val_emb[bin(pixel_t)]

2. Outer-product linear RNN  (processes tokens one at a time)
   k_t = W_k @ emb_t         (d_k,)
   v_t = W_v @ emb_t         (d_v,)
   M_t = M_{t-1} + k_t ⊗ v_t    ← LINEAR state update, no nonlinearity

   M_T = Σ_t k_t ⊗ v_t  is the final (d_k × d_v) state matrix.
   This captures ALL pairwise co-occurrences of key- and value-features
   across all positions — far more expressive than a vector-state RNN while
   remaining strictly linear in the recurrence.

   Equivalent view: M_T[a,b] = Σ_t k_t[a] · v_t[b]
                              = (K^T V)[a,b]    where K,V ∈ R^{784×d}
   So the whole thing can be computed via two matrix multiplications (no scan),
   which makes training fast on CPU.

3. Layer-norm + 2-layer MLP classifier
   logits = W2 · relu(W1 · LayerNorm(vec(M_T)) + b1) + b2
"""

import time
import jax
import optax
import logging
from functools import partial
from jax import Array
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Iterator

from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_VAL_BINS = 32   # quantise pixel 0-255 into 32 bins


def layer_norm(x: Array, gamma: Array, beta: Array, eps: float = 1e-5) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def init() -> Tuple[Dict[str, Any], Dict[str, Array], Iterator]:
    config = {
        "dataset_name": "mnist",
        "num_epochs":   30,
        "embed_dim":    64,    # d
        "d_k":          64,    # key  dimension
        "d_v":          64,    # value dimension → state is (d_k × d_v) = 4096-d
        "mlp_dim":      1024,
        "n_val_bins":   N_VAL_BINS,
        "batch_size":   128,
        "learning_rate": 1e-3,
        "random_seed":  42,
        "target_accuracy": 0.95,
    }

    key     = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)

    d   = config["embed_dim"]
    dk  = config["d_k"]
    dv  = config["d_v"]
    K   = config["n_val_bins"]
    ml  = config["mlp_dim"]
    state_dim = dk * dv   # 4096

    params = {
        # ── embeddings ───────────────────────────────────────────────────────
        "pos_emb": jax.random.normal(next(key_gen).get(), (784, d)) * 0.02,
        "val_emb": jax.random.normal(next(key_gen).get(), (K,   d)) * 0.02,
        # ── key / value projections ──────────────────────────────────────────
        "W_k": jax.random.normal(next(key_gen).get(), (dk, d)) / d**0.5,
        "W_v": jax.random.normal(next(key_gen).get(), (dv, d)) / d**0.5,
        # ── layer norm over vec(M_T) ─────────────────────────────────────────
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        # ── MLP readout ───────────────────────────────────────────────────────
        "W1": jax.random.normal(next(key_gen).get(), (ml, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(ml),
        "W2": jax.random.normal(next(key_gen).get(), (10, ml)) * 0.01,
        "b2": jnp.zeros(10),
    }

    return config, params, key_gen


def rnn_forward(params: Dict[str, Array], x: Array) -> Array:
    """
    x: (batch, 784) float, values 0-255.

    Conceptually processes one pixel at a time:
      M_t = M_{t-1} + k_t ⊗ v_t   (linear RNN with matrix state)

    Equivalent closed-form (since A=I, B=identity):
      M_T = K^T V   where  K = emb @ W_k.T,  V = emb @ W_v.T
    """
    batch_size  = x.shape[0]
    pixels_norm = x / 255.0   # (batch, 784) in [0,1]

    # ── build per-step embeddings ────────────────────────────────────────────
    pixel_bins = jnp.floor(pixels_norm * (N_VAL_BINS - 1)).astype(jnp.int32)
    val_part   = params["val_emb"][pixel_bins]                # (batch, 784, d)
    emb        = params["pos_emb"][None, :, :] + val_part     # (batch, 784, d)

    # ── keys and values ──────────────────────────────────────────────────────
    k = emb @ params["W_k"].T     # (batch, 784, d_k)
    v = emb @ params["W_v"].T     # (batch, 784, d_v)

    # ── linear RNN: accumulate outer products  M_T = Σ_t k_t ⊗ v_t ─────────
    # Written as batched matrix multiply: M_T[b] = k[b].T @ v[b]
    M = jnp.einsum("bti,btj->bij", k, v)    # (batch, d_k, d_v)

    # ── normalise + classify ─────────────────────────────────────────────────
    M_flat = M.reshape(batch_size, -1)       # (batch, d_k*d_v)
    M_norm = layer_norm(M_flat, params["ln_gamma"], params["ln_beta"])
    h1     = jax.nn.relu(M_norm @ params["W1"].T + params["b1"])
    return h1 @ params["W2"].T + params["b2"]


def loss_fn(params: Dict[str, Array], x: Array, y: Array) -> Array:
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(rnn_forward(params, x), y)
    )


def accuracy_fn(params: Dict[str, Array], x: Array, y: Array) -> float:
    return float(jnp.mean(jnp.argmax(rnn_forward(params, x), axis=1) == y))


@partial(jax.jit, static_argnums=(0,))
def train_step(
    optimizer: optax.GradientTransformation,
    params: Dict[str, Array],
    opt_state: Any,
    batch_x: Array,
    batch_y: Array,
) -> Tuple[Dict[str, Array], Any, Array]:
    loss_value, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt_state, loss_value


if __name__ == "__main__":
    config, params, key_gen = init()

    logging.info("Loading MNIST dataset...")
    data = load_supervised_image(config["dataset_name"])

    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    logging.info(f"Train: {X_train.shape}  Test: {X_test.shape}")
    logging.info(
        f"Config: d_emb={config['embed_dim']}  d_k={config['d_k']}  d_v={config['d_v']}  "
        f"state_dim={config['d_k']*config['d_v']}  "
        f"val_bins={N_VAL_BINS}  mlp={config['mlp_dim']}  "
        f"lr={config['learning_rate']}  batch={config['batch_size']}"
    )

    num_batches  = data.n_samples // config["batch_size"]
    total_steps  = config["num_epochs"] * num_batches

    schedule  = optax.cosine_decay_schedule(config["learning_rate"], total_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(params)

    logging.info("Training: outer-product linear RNN (pixel-by-pixel MNIST)...")
    start_time = time.perf_counter()

    for epoch in range(config["num_epochs"]):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, data.n_samples)
        Xs, ys = X_train[perm], y_train[perm]

        epoch_loss = 0.0
        for batch in range(num_batches):
            s = batch * config["batch_size"]
            e = s + config["batch_size"]
            params, opt_state, loss_value = train_step(
                optimizer, params, opt_state, Xs[s:e], ys[s:e]
            )
            epoch_loss += float(loss_value)
            if batch % 100 == 0:
                logging.info(
                    f"Epoch {epoch}  batch {batch:3d}/{num_batches}  loss {loss_value:.4f}"
                )

        avg_loss  = epoch_loss / num_batches
        train_acc = accuracy_fn(params, X_train[:4096], y_train[:4096])
        test_acc  = accuracy_fn(params, X_test, y_test)
        elapsed   = time.perf_counter() - start_time

        logging.info(
            f"=== Epoch {epoch:2d}  avg_loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  elapsed={elapsed:.1f}s ==="
        )

        if test_acc >= config["target_accuracy"]:
            logging.info(f"Target {config['target_accuracy']:.0%} reached — done.")
            break

    final_test_acc = accuracy_fn(params, X_test, y_test)
    logging.info(
        f"Final test accuracy: {final_test_acc:.4f}  "
        f"total: {time.perf_counter() - start_time:.1f}s"
    )
