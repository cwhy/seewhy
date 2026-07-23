"""
Simplification experiments for the outer-product linear RNN on MNIST.

Runs each variant in sequence and prints a summary table.

Variants (in order of the plan):
  baseline  — W_k, W_v projections + MLP readout  (d=64, state=4096)
  S1        — symmetric k=v (one shared projection W)
  S2        — no projection   (k=v=emb directly)
  S3        — linear readout  (S2 + drop MLP → single W_out)
  S4        — d=32 embedding  (S3 with smaller embedding)
  delta     — delta rule on top of S1 (if S1 passes)

Each variant trains for up to MAX_EPOCHS epochs, stopping early at TARGET_ACC.
"""

import time
import jax
import optax
import logging
import jax.numpy as jnp
from functools import partial
from jax import Array
from typing import Dict, Any, Tuple

from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── shared hyperparameters ───────────────────────────────────────────────────
N_VAL_BINS  = 32
BATCH_SIZE  = 128
LR          = 1e-3
MAX_EPOCHS  = 10
TARGET_ACC  = 0.95
SEED        = 42


# ── shared utilities ─────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def make_optimizer(num_batches):
    total_steps = MAX_EPOCHS * num_batches
    schedule = optax.cosine_decay_schedule(LR, total_steps)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))


def loss_fn(forward_fn, params, x, y):
    logits = forward_fn(params, x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


def accuracy(forward_fn, params, x, y):
    return float(jnp.mean(jnp.argmax(forward_fn(params, x), axis=1) == y))


def train(name, forward_fn, params, X_train, y_train, X_test, y_test):
    """Train one variant. Returns (test_acc, elapsed, epochs_run)."""
    num_batches = len(X_train) // BATCH_SIZE
    optimizer   = make_optimizer(num_batches)
    opt_state   = optimizer.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    @partial(jax.jit, static_argnums=(0,))
    def step(opt, p, s, bx, by):
        loss, grads = jax.value_and_grad(partial(loss_fn, forward_fn))(p, bx, by)
        updates, s2 = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s2, loss

    t0 = time.perf_counter()
    test_acc, epoch = 0.0, 0
    for epoch in range(MAX_EPOCHS):
        perm = jax.random.permutation(next(key_gen).get(), len(X_train))
        Xs, ys = X_train[perm], y_train[perm]
        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            params, opt_state, _ = step(optimizer, params, opt_state, Xs[s:e], ys[s:e])
        test_acc = accuracy(forward_fn, params, X_test, y_test)
        elapsed  = time.perf_counter() - t0
        logging.info(f"[{name}] epoch {epoch}  test_acc={test_acc:.4f}  {elapsed:.1f}s")
        if test_acc >= TARGET_ACC:
            break

    return test_acc, time.perf_counter() - t0, epoch + 1


# ── variant definitions ──────────────────────────────────────────────────────

def baseline_init(d=64, mlp=1024):
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    K = N_VAL_BINS
    state_dim = d * d
    return {
        "pos_emb":  jax.random.normal(next(key_gen).get(), (784, d)) * 0.02,
        "val_emb":  jax.random.normal(next(key_gen).get(), (K,   d)) * 0.02,
        "W_k":      jax.random.normal(next(key_gen).get(), (d,   d)) / d**0.5,
        "W_v":      jax.random.normal(next(key_gen).get(), (d,   d)) / d**0.5,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1":       jax.random.normal(next(key_gen).get(), (mlp, state_dim)) / state_dim**0.5,
        "b1":       jnp.zeros(mlp),
        "W2":       jax.random.normal(next(key_gen).get(), (10, mlp)) * 0.01,
        "b2":       jnp.zeros(10),
    }

def baseline_fwd(params, x):
    batch = x.shape[0]
    bins  = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    emb   = params["pos_emb"][None] + params["val_emb"][bins]   # (B,784,d)
    k     = emb @ params["W_k"].T
    v     = emb @ params["W_v"].T
    M     = jnp.einsum("bti,btj->bij", k, v).reshape(batch, -1)
    M     = layer_norm(M, params["ln_gamma"], params["ln_beta"])
    return jax.nn.relu(M @ params["W1"].T + params["b1"]) @ params["W2"].T + params["b2"]


# S1: one shared projection W (symmetric Gram matrix M = E^T E where E = emb @ W^T)
def s1_init(d=64, mlp=1024):
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    K = N_VAL_BINS; state_dim = d * d
    return {
        "pos_emb":  jax.random.normal(next(key_gen).get(), (784, d)) * 0.02,
        "val_emb":  jax.random.normal(next(key_gen).get(), (K,   d)) * 0.02,
        "W":        jax.random.normal(next(key_gen).get(), (d,   d)) / d**0.5,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1":       jax.random.normal(next(key_gen).get(), (mlp, state_dim)) / state_dim**0.5,
        "b1":       jnp.zeros(mlp),
        "W2":       jax.random.normal(next(key_gen).get(), (10, mlp)) * 0.01,
        "b2":       jnp.zeros(10),
    }

def s1_fwd(params, x):
    batch = x.shape[0]
    bins  = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    emb   = params["pos_emb"][None] + params["val_emb"][bins]
    e     = emb @ params["W"].T                                 # single projection
    M     = jnp.einsum("bti,btj->bij", e, e).reshape(batch, -1)  # symmetric
    M     = layer_norm(M, params["ln_gamma"], params["ln_beta"])
    return jax.nn.relu(M @ params["W1"].T + params["b1"]) @ params["W2"].T + params["b2"]


# S2: no projection at all — M = emb^T emb
def s2_init(d=64, mlp=1024):
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    K = N_VAL_BINS; state_dim = d * d
    return {
        "pos_emb":  jax.random.normal(next(key_gen).get(), (784, d)) * 0.02,
        "val_emb":  jax.random.normal(next(key_gen).get(), (K,   d)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1":       jax.random.normal(next(key_gen).get(), (mlp, state_dim)) / state_dim**0.5,
        "b1":       jnp.zeros(mlp),
        "W2":       jax.random.normal(next(key_gen).get(), (10, mlp)) * 0.01,
        "b2":       jnp.zeros(10),
    }

def s2_fwd(params, x):
    batch = x.shape[0]
    bins  = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    emb   = params["pos_emb"][None] + params["val_emb"][bins]   # (B,784,d)
    M     = jnp.einsum("bti,btj->bij", emb, emb).reshape(batch, -1)
    M     = layer_norm(M, params["ln_gamma"], params["ln_beta"])
    return jax.nn.relu(M @ params["W1"].T + params["b1"]) @ params["W2"].T + params["b2"]


# S3: no projection + linear readout (drop MLP entirely)
def s3_init(d=64):
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    K = N_VAL_BINS; state_dim = d * d
    return {
        "pos_emb":  jax.random.normal(next(key_gen).get(), (784, d)) * 0.02,
        "val_emb":  jax.random.normal(next(key_gen).get(), (K,   d)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W_out":    jax.random.normal(next(key_gen).get(), (10, state_dim)) * 0.01,
        "b_out":    jnp.zeros(10),
    }

def s3_fwd(params, x):
    batch = x.shape[0]
    bins  = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    emb   = params["pos_emb"][None] + params["val_emb"][bins]
    M     = jnp.einsum("bti,btj->bij", emb, emb).reshape(batch, -1)
    M     = layer_norm(M, params["ln_gamma"], params["ln_beta"])
    return M @ params["W_out"].T + params["b_out"]


# S4: d=32 + no projection + linear readout
def s4_init(d=32):
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    K = N_VAL_BINS; state_dim = d * d
    return {
        "pos_emb":  jax.random.normal(next(key_gen).get(), (784, d)) * 0.02,
        "val_emb":  jax.random.normal(next(key_gen).get(), (K,   d)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W_out":    jax.random.normal(next(key_gen).get(), (10, state_dim)) * 0.01,
        "b_out":    jnp.zeros(10),
    }

def s4_fwd(params, x):
    batch = x.shape[0]
    bins  = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    emb   = params["pos_emb"][None] + params["val_emb"][bins]
    M     = jnp.einsum("bti,btj->bij", emb, emb).reshape(batch, -1)
    M     = layer_norm(M, params["ln_gamma"], params["ln_beta"])
    return M @ params["W_out"].T + params["b_out"]


# Delta: delta rule on top of S1 (asymmetric W_k,W_v, MLP, + error correction)
def delta_init(d=64, mlp=1024):
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    K = N_VAL_BINS; state_dim = d * d
    return {
        "pos_emb":  jax.random.normal(next(key_gen).get(), (784, d)) * 0.02,
        "val_emb":  jax.random.normal(next(key_gen).get(), (K,   d)) * 0.02,
        "W_k":      jax.random.normal(next(key_gen).get(), (d,   d)) / d**0.5,
        "W_v":      jax.random.normal(next(key_gen).get(), (d,   d)) / d**0.5,
        "log_beta": jnp.zeros(()),          # scalar beta = sigmoid(log_beta), init=0.5
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1":       jax.random.normal(next(key_gen).get(), (mlp, state_dim)) / state_dim**0.5,
        "b1":       jnp.zeros(mlp),
        "W2":       jax.random.normal(next(key_gen).get(), (10, mlp)) * 0.01,
        "b2":       jnp.zeros(10),
    }

def delta_fwd(params, x):
    """
    Delta rule: M_t = M_{t-1} + beta * (v_t - M_{t-1} k_t) ⊗ k_t
    = M_{t-1} + beta * v_t k_t^T - beta * (M_{t-1} k_t) k_t^T

    In closed form over the full sequence this becomes:
      M_T = beta * V^T K - beta * (cumulative correction terms)
    This is NOT a simple einsum — we need the scan.
    We use jax.lax.scan for correctness.
    """
    batch = x.shape[0]
    bins  = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    emb   = params["pos_emb"][None] + params["val_emb"][bins]   # (B,784,d)
    k_all = emb @ params["W_k"].T     # (B, 784, d)
    v_all = emb @ params["W_v"].T     # (B, 784, d)

    beta  = jax.nn.sigmoid(params["log_beta"])

    # Transpose for scan: (784, B, d)
    k_seq = jnp.transpose(k_all, (1, 0, 2))
    v_seq = jnp.transpose(v_all, (1, 0, 2))

    def step(M, kv):
        k_t, v_t = kv                           # each (B, d)
        pred      = jnp.einsum("bij,bj->bi", M, k_t)   # M_{t-1} k_t
        error     = v_t - pred
        M_new     = M + beta * jnp.einsum("bi,bj->bij", error, k_t)
        return M_new, None

    M0 = jnp.zeros((batch, params["W_k"].shape[0], params["W_k"].shape[0]))
    M_final, _ = jax.lax.scan(step, M0, (k_seq, v_seq))

    M_flat = M_final.reshape(batch, -1)
    M_norm = layer_norm(M_flat, params["ln_gamma"], params["ln_beta"])
    return jax.nn.relu(M_norm @ params["W1"].T + params["b1"]) @ params["W2"].T + params["b2"]


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    def param_count(p):
        return sum(v.size for v in jax.tree_util.tree_leaves(p))

    experiments = [
        ("baseline", baseline_init(), baseline_fwd),
        ("S1 sym k=v",     s1_init(),  s1_fwd),
        ("S2 no-proj",     s2_init(),  s2_fwd),
        ("S3 linear-out",  s3_init(),  s3_fwd),
        ("S4 d=32",        s4_init(),  s4_fwd),
        ("delta rule",     delta_init(), delta_fwd),
    ]

    results = []
    for name, params, fwd in experiments:
        n_params = param_count(params)
        logging.info(f"\n{'='*60}\n{name}  ({n_params:,} params)\n{'='*60}")
        acc, elapsed, epochs = train(name, fwd, params, X_train, y_train, X_test, y_test)
        results.append((name, n_params, acc, epochs, elapsed))

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Variant':<18} {'Params':>10}  {'Test acc':>9}  {'Epochs':>7}  {'Time':>7}")
    print("-"*70)
    for name, n_params, acc, epochs, elapsed in results:
        flag = " ✓" if acc >= TARGET_ACC else "  "
        print(f"{name:<18} {n_params:>10,}  {acc:>9.4f}{flag}  {epochs:>7}  {elapsed:>6.1f}s")
    print("="*70)
