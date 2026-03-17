"""
Outer-product embeddings: pos_emb[t] ⊗ val_emb[bin(pixel_t)]

Current (additive):  emb_t = pos_emb[t] + val_emb[bin]   — (d,) from 784d + 32d params
New    (outer):      emb_t = vec(pos_emb[t] ⊗ val_emb[bin]) — (d_p·d_v,) from 784·d_p + 32·d_v params

Why it might help
─────────────────
Additive embeddings: emb_t[i] = pos[t,i] + val[bin,i]
  → M_T[a,b] = Σ_t (pos[t,a]+val[t,a]) · (pos[t,b]+val[t,b])
  → 2nd-order cross terms between position and value features

Outer embeddings: emb_t[i,j] = pos[t,i] · val[bin,j]
  → M_T[(i,j),(k,l)] = Σ_t pos[t,i]·val[t,j]·pos[t,k]·val[t,l]
  → 4th-order statistics: (pos⊗pos) × (val⊗val) products
  → same state size but richer feature interactions

Why it's cheaper
────────────────
pos_emb: 784 × d_p  (instead of 784 × d)
val_emb: 32  × d_v  (instead of 32  × d)
emb dim = d_p·d_v   (same as before if d_p·d_v = d²... but much smaller d_p,d_v possible)

Example: additive d=64 → 52K embed params
         outer d_p=d_v=8 → 6.5K embed params, same 64-dim embedding

Experiments
───────────
For each (d_p, d_v, mlp_dim) we compare:
  - additive (baseline from experiments3)
  - outer product (this file)
at the same state_dim = (d_p*d_v)².
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
MAX_EPOCHS = 10
TARGET_ACC = 0.95
SEED       = 42


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


# ── additive embedding (baseline) ────────────────────────────────────────────

def make_additive(d, mlp_dim):
    state_dim = d * d

    def fwd(params, x):
        B    = x.shape[0]
        bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
        emb  = params["pos_emb"][None] + params["val_emb"][bins]    # (B, 784, d)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        if mlp_dim > 0:
            M = jax.nn.relu(M @ params["W1"].T + params["b1"])
            return M @ params["W2"].T + params["b2"]
        else:
            return M @ params["W_out"].T + params["b_out"]

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "pos_emb":  jax.random.normal(next(kg).get(), (784, d)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (N_VAL_BINS, d)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
    }
    if mlp_dim > 0:
        p["W1"] = jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5
        p["b1"] = jnp.zeros(mlp_dim)
        p["W2"] = jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01
        p["b2"] = jnp.zeros(10)
    else:
        p["W_out"] = jax.random.normal(next(kg).get(), (10, state_dim)) * 0.01
        p["b_out"] = jnp.zeros(10)
    return fwd, p


# ── outer-product embedding ───────────────────────────────────────────────────

def make_outer(d_p, d_v, mlp_dim):
    """
    emb_t = vec(pos_emb[t] ⊗ val_emb[bin])   shape: (d_p * d_v,)
    M_T   = Σ_t emb_t ⊗ emb_t                 shape: (d_p*d_v, d_p*d_v)
    state_dim = (d_p * d_v)²
    """
    emb_dim   = d_p * d_v
    state_dim = emb_dim * emb_dim

    def fwd(params, x):
        B    = x.shape[0]
        bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
        pe   = params["pos_emb"][None]          # (1, 784, d_p)
        ve   = params["val_emb"][bins]          # (B, 784, d_v)
        # outer product per token: (B, 784, d_p, d_v) → (B, 784, d_p*d_v)
        emb  = (pe[..., :, None] * ve[..., None, :]).reshape(B, 784, emb_dim)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        if mlp_dim > 0:
            M = jax.nn.relu(M @ params["W1"].T + params["b1"])
            return M @ params["W2"].T + params["b2"]
        else:
            return M @ params["W_out"].T + params["b_out"]

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "pos_emb":  jax.random.normal(next(kg).get(), (784, d_p)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
    }
    if mlp_dim > 0:
        p["W1"] = jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5
        p["b1"] = jnp.zeros(mlp_dim)
        p["W2"] = jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01
        p["b2"] = jnp.zeros(10)
    else:
        p["W_out"] = jax.random.normal(next(kg).get(), (10, state_dim)) * 0.01
        p["b_out"] = jnp.zeros(10)
    return fwd, p


# ── training harness ─────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


def train(name, fwd, params, X_tr, y_tr, X_te, y_te):
    num_batches = len(X_tr) // BATCH_SIZE
    schedule    = optax.cosine_decay_schedule(LR, MAX_EPOCHS * num_batches)
    optimizer   = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state   = optimizer.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    loss_fn = lambda p, bx, by: jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by)
    )

    def acc_fn(p, bx, by):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, bx[i:i+BATCH_SIZE]), 1)
            for i in range(0, len(bx), BATCH_SIZE)
        ])
        return float(jnp.mean(preds == by))

    @partial(jax.jit, static_argnums=(0,))
    def step(opt, p, s, bx, by):
        loss, g = jax.value_and_grad(loss_fn)(p, bx, by)
        u, s2   = opt.update(g, s, p)
        return optax.apply_updates(p, u), s2, loss

    t0 = time.perf_counter()
    test_acc = 0.0
    for epoch in range(MAX_EPOCHS):
        perm = jax.random.permutation(next(key_gen).get(), len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            params, opt_state, _ = step(optimizer, params, opt_state, Xs[s:e], ys[s:e])
        test_acc = acc_fn(params, X_te, y_te)
        logging.info(f"[{name}] epoch {epoch}  test={test_acc:.4f}  {time.perf_counter()-t0:.1f}s")
        if test_acc >= TARGET_ACC:
            break
    return test_acc, time.perf_counter() - t0, epoch + 1


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # Each entry: (label, fwd, params)
    # Grouped in pairs: additive vs outer at same (state_dim, mlp_dim)
    experiments = []

    # ── group 1: emb=64 (state=4096), mlp=128 ────────────────────────────────
    # additive d=64: 52K embed + 525K mlp = 586K
    # outer d_p=d_v=8: 6.5K embed + 525K mlp = 540K
    fwd, p = make_additive(64, 128)
    experiments.append(("add  d=64   mlp=128", fwd, p))
    fwd, p = make_outer(8, 8, 128)
    experiments.append(("outer(8,8)  mlp=128", fwd, p))

    # ── group 2: emb=64 (state=4096), mlp=64 ─────────────────────────────────
    fwd, p = make_additive(64, 64)
    experiments.append(("add  d=64   mlp=64 ", fwd, p))
    fwd, p = make_outer(8, 8, 64)
    experiments.append(("outer(8,8)  mlp=64 ", fwd, p))

    # ── group 3: emb=32 (state=1024), mlp=128 ────────────────────────────────
    # additive d=32: 27K embed + 131K mlp = 161K
    # outer d_p=4,d_v=8: 3.4K embed + 131K mlp = 138K
    fwd, p = make_additive(32, 128)
    experiments.append(("add  d=32   mlp=128", fwd, p))
    fwd, p = make_outer(4, 8, 128)
    experiments.append(("outer(4,8)  mlp=128", fwd, p))

    # ── group 4: emb=32 (state=1024), mlp=64 ─────────────────────────────────
    fwd, p = make_additive(32, 64)
    experiments.append(("add  d=32   mlp=64 ", fwd, p))
    fwd, p = make_outer(4, 8, 64)
    experiments.append(("outer(4,8)  mlp=64 ", fwd, p))

    # ── group 5: smaller outer — where additive can't go ─────────────────────
    # outer(4,4): emb=16, state=256; outer(8,4): emb=32, state=1024
    # outer(6,8): emb=48, state=2304
    fwd, p = make_outer(6, 8, 128)
    experiments.append(("outer(6,8)  mlp=128", fwd, p))  # ~230K, state=2304
    fwd, p = make_outer(4, 4, 128)
    experiments.append(("outer(4,4)  mlp=128", fwd, p))  # ~34K, state=256
    fwd, p = make_outer(4, 4, 64)
    experiments.append(("outer(4,4)  mlp=64 ", fwd, p))  # ~18K, state=256

    results = []
    for name, fwd, params in experiments:
        np_ = n_params(params)
        logging.info(f"\n{'='*60}\n{name}  ({np_:,} params)\n{'='*60}")
        acc, elapsed, epochs = train(name, fwd, params, X_train, y_train, X_test, y_test)
        results.append((name, np_, acc, epochs, elapsed))

    print(f"\n{'='*72}")
    print(f"{'Variant':<22} {'Params':>10} {'Test acc':>10} {'Epochs':>7} {'Time':>8}")
    print(f"{'-'*72}")
    for name, np_, acc, epochs, elapsed in results:
        flag = "✓" if acc >= TARGET_ACC else " "
        print(f"{name:<22} {np_:>10,} {acc:>9.4f}{flag} {epochs:>7} {elapsed:>7.1f}s")
    print(f"{'='*72}")
