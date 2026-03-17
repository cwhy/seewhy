"""
Push toward 96%+ accuracy. All variants report train + test accuracy each epoch.

Baseline: outer(8,8) mlp=128 → 95.2%, 138K params (best from experiments4)

Axes explored:
  1. MLP capacity     — mlp ∈ {256, 512, 1024} with outer(8,8)
  2. Weight decay     — AdamW λ=1e-4 with outer(8,8) mlp=128 and mlp=256
  3. Longer training  — 30 epochs, warmup 2 epochs, outer(8,8) mlp=128
  4. Dropout          — p=0.1, 0.2 in MLP hidden layer
  5. Asymmetric proj  — outer emb → W_k, W_v projections → outer-product state
                        (equivalent to original model but with outer embeddings)
  6. Deeper MLP       — 3 layers (d→256→64→10)
"""

import time
import jax
import jax.numpy as jnp
import optax
import logging
from functools import partial

from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_VAL_BINS = 32
BATCH_SIZE = 128
LR         = 1e-3
MAX_EPOCHS = 30
TARGET_ACC = 0.96
SEED       = 42

# ── utils ─────────────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def outer_emb(params, x, d_p, d_v):
    """outer-product embedding: (B,784,d_p*d_v)"""
    bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    pe   = params["pos_emb"][None]            # (1, 784, d_p)
    ve   = params["val_emb"][bins]            # (B, 784, d_v)
    return (pe[..., :, None] * ve[..., None, :]).reshape(x.shape[0], 784, d_p * d_v)


def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


def accuracy_batched(fwd, params, x, y):
    preds = jnp.concatenate([
        jnp.argmax(fwd(params, x[i:i+BATCH_SIZE]), 1)
        for i in range(0, len(x), BATCH_SIZE)
    ])
    return float(jnp.mean(preds == y))


# ── model builders ────────────────────────────────────────────────────────────

def make_outer_mlp(d_p, d_v, mlp_dim, dropout_p=0.0, weight_decay=0.0):
    """outer(d_p,d_v) → Gram matrix → MLP. Configurable dropout + weight decay."""
    emb_dim   = d_p * d_v
    state_dim = emb_dim * emb_dim

    def fwd(params, x, *, train=False, rng=None):
        B   = x.shape[0]
        emb = outer_emb(params, x, d_p, d_v)
        M   = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M   = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h   = jax.nn.relu(M @ params["W1"].T + params["b1"])
        if train and dropout_p > 0.0 and rng is not None:
            mask = jax.random.bernoulli(rng, 1.0 - dropout_p, h.shape)
            h    = h * mask / (1.0 - dropout_p)
        return h @ params["W2"].T + params["b2"]

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "pos_emb":  jax.random.normal(next(kg).get(), (784, d_p)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    return fwd, p, weight_decay


def make_outer_mlp3(d_p, d_v, h1, h2, weight_decay=0.0):
    """3-layer MLP: state → h1 → h2 → 10."""
    emb_dim   = d_p * d_v
    state_dim = emb_dim * emb_dim

    def fwd(params, x, *, train=False, rng=None):
        B   = x.shape[0]
        emb = outer_emb(params, x, d_p, d_v)
        M   = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M   = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        a   = jax.nn.relu(M @ params["W1"].T + params["b1"])
        b_  = jax.nn.relu(a @ params["W2"].T + params["b2"])
        return b_ @ params["W3"].T + params["b3"]

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "pos_emb":  jax.random.normal(next(kg).get(), (784, d_p)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (h1, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(h1),
        "W2": jax.random.normal(next(kg).get(), (h2, h1)) / h1**0.5,
        "b2": jnp.zeros(h2),
        "W3": jax.random.normal(next(kg).get(), (10, h2)) * 0.01,
        "b3": jnp.zeros(10),
    }
    return fwd, p, weight_decay


def make_outer_asym(d_p, d_v, d_k, d_v2, mlp_dim, weight_decay=0.0):
    """outer emb → W_k, W_v projections → outer-product state → MLP.
    Equivalent to the original model but uses outer embeddings as input."""
    emb_dim   = d_p * d_v
    state_dim = d_k * d_v2

    def fwd(params, x, *, train=False, rng=None):
        B   = x.shape[0]
        emb = outer_emb(params, x, d_p, d_v)       # (B, 784, emb_dim)
        k   = emb @ params["W_k"].T                  # (B, 784, d_k)
        v   = emb @ params["W_v"].T                  # (B, 784, d_v2)
        M   = jnp.einsum("bti,btj->bij", k, v).reshape(B, -1)
        M   = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h   = jax.nn.relu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "pos_emb":  jax.random.normal(next(kg).get(), (784, d_p)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
        "W_k": jax.random.normal(next(kg).get(), (d_k,   emb_dim)) / emb_dim**0.5,
        "W_v": jax.random.normal(next(kg).get(), (d_v2,  emb_dim)) / emb_dim**0.5,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    return fwd, p, weight_decay


# ── training harness ─────────────────────────────────────────────────────────

def train(name, fwd, params, wd, X_tr, y_tr, X_te, y_te, max_epochs=MAX_EPOCHS,
          warmup_epochs=0):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    warmup_steps = warmup_epochs * num_batches

    if warmup_steps > 0:
        sched = optax.join_schedules(
            [optax.linear_schedule(0.0, LR, warmup_steps),
             optax.cosine_decay_schedule(LR, total_steps - warmup_steps)],
            [warmup_steps],
        )
    else:
        sched = optax.cosine_decay_schedule(LR, total_steps)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(sched, weight_decay=wd) if wd > 0 else optax.adam(sched),
    )
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    has_dropout = any("drop" in name.lower() for _ in [name])

    def loss_fn(p, bx, by, rng=None):
        logits = fwd(p, bx, train=True, rng=rng)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, by))

    @partial(jax.jit, static_argnums=(0,))
    def step(tx_, p, s, bx, by, rng):
        loss, g = jax.value_and_grad(loss_fn)(p, bx, by, rng)
        u, s2   = tx_.update(g, s, p)
        return optax.apply_updates(p, u), s2, loss

    def acc(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i+BATCH_SIZE], train=False), 1)
            for i in range(0, len(x), BATCH_SIZE)
        ])
        return float(jnp.mean(preds == y))

    t0 = time.perf_counter()
    test_acc = 0.0
    for epoch in range(max_epochs):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        rng  = key
        for b in range(num_batches):
            s, e  = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            rng, = jax.random.split(rng, 1)
            params, opt_state, _ = step(tx, params, opt_state, Xs[s:e], ys[s:e], rng)
        train_acc = acc(params, X_tr, y_tr)
        test_acc  = acc(params, X_te, y_te)
        logging.info(
            f"[{name}] epoch {epoch:2d}  "
            f"train={train_acc:.4f}  test={test_acc:.4f}  "
            f"{time.perf_counter()-t0:.1f}s"
        )
        if test_acc >= TARGET_ACC:
            break
    return train_acc, test_acc, time.perf_counter() - t0, epoch + 1


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    experiments = [
        # name,                        fwd, params, wd,   max_epochs, warmup
        ("outer(8,8) mlp=128 [ref]",   *make_outer_mlp(8, 8, 128),   10, 0),
        ("outer(8,8) mlp=256",         *make_outer_mlp(8, 8, 256),   15, 0),
        ("outer(8,8) mlp=512",         *make_outer_mlp(8, 8, 512),   15, 0),
        ("outer(8,8) mlp=1024",        *make_outer_mlp(8, 8, 1024),  15, 0),
        ("outer(8,8) mlp=256 wd=1e-4", *make_outer_mlp(8, 8, 256, weight_decay=1e-4), 15, 0),
        ("outer(8,8) mlp=128 wd=1e-4", *make_outer_mlp(8, 8, 128, weight_decay=1e-4), 20, 0),
        ("outer(8,8) mlp=128 warmup",  *make_outer_mlp(8, 8, 128),   20, 2),
        ("outer(8,8) mlp=128 drop=0.1",*make_outer_mlp(8, 8, 128, dropout_p=0.1), 20, 0),
        ("outer(8,8) mlp=128 drop=0.2",*make_outer_mlp(8, 8, 128, dropout_p=0.2), 20, 0),
        ("outer(8,8) 3-layer 256→64",  *make_outer_mlp3(8, 8, 256, 64), 15, 0),
        ("outer(8,8) 3-layer 512→128", *make_outer_mlp3(8, 8, 512, 128), 15, 0),
        ("outer+asym(64,64) mlp=128",  *make_outer_asym(8, 8, 64, 64, 128), 15, 0),
        ("outer+asym(64,64) mlp=256",  *make_outer_asym(8, 8, 64, 64, 256), 15, 0),
    ]

    results = []
    for row in experiments:
        name, fwd, params, wd, max_ep, warmup = row
        np_ = n_params(params)
        logging.info(f"\n{'='*60}\n{name}  ({np_:,} params)\n{'='*60}")
        tr_acc, te_acc, elapsed, epochs = train(
            name, fwd, params, wd, X_train, y_train, X_test, y_test,
            max_epochs=max_ep, warmup_epochs=warmup,
        )
        results.append((name, np_, tr_acc, te_acc, epochs, elapsed))

    print(f"\n{'='*82}")
    print(f"{'Variant':<32} {'Params':>10} {'Train':>8} {'Test':>8} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*82}")
    for name, np_, tr, te, ep, t in results:
        flag = "✓" if te >= TARGET_ACC else " "
        print(f"{name:<32} {np_:>10,} {tr:>7.4f} {te:>7.4f}{flag} {ep:>4} {t:>7.1f}s")
    print(f"{'='*82}")
