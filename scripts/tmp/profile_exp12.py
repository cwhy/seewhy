"""
Simple wall-clock profiling of exp12 training loop.
Runs 3 warm-up epochs then times 5 measured epochs,
breaking down: data prep / epoch_fn (full scan) / eval.
Also micro-benchmarks individual loss components.

Usage:
    uv run python scripts/tmp/profile_exp12.py
"""

import sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "variations"))

import jax
import jax.numpy as jnp
import optax
import numpy as np

from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image
from experiments12 import (
    init_params, encode, decode, cats_to_cond,
    loss_fn, make_epoch_fn,
    BATCH_SIZE, N_VAR_SAMPLES, K_NN, MAX_EPOCHS,
    LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM, IN_DIM,
    ADAM_LR, SEED, ROWS, COLS,
    D, D_RC, D_R, D_C, DEC_RANK, MLP_HIDDEN,
)

def T(label, fn, reps=1):
    """Run fn() reps times, return mean wall time in ms."""
    jax.block_until_ready(fn())  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / reps * 1000

# ── Setup ──────────────────────────────────────────────────────────────────────

print("Loading data...")
data    = load_supervised_image("mnist")
X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)

KNN_CACHE = ROOT / "projects/variations/knn_mnist_k64.npy"
knn_idx   = np.load(KNN_CACHE)
print(f"kNN loaded: {knn_idx.shape}")

key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
params  = init_params(key_gen)

num_batches = len(X_train) // BATCH_SIZE
num_steps   = MAX_EPOCHS * num_batches
optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
opt_state   = optimizer.init(params)
epoch_fn    = make_epoch_fn(optimizer)
X_train_jax = jnp.array(X_train)

def make_batch(seed=0):
    rng  = np.random.RandomState(seed)
    perm = rng.permutation(len(X_train))
    n    = num_batches * BATCH_SIZE
    Xs   = jnp.array(X_train[perm[:n]].reshape(num_batches, BATCH_SIZE, 784))
    KNN  = jnp.array(knn_idx[perm[:n]].reshape(num_batches, BATCH_SIZE, K_NN))
    return Xs, KNN

print("\nPreparing batches and warming up JIT (3 epochs)...")
Xs_b, KNN_b = make_batch()
epoch_key   = jax.random.PRNGKey(99)

for i in range(3):
    params, opt_state, *_ = epoch_fn(params, opt_state, Xs_b, KNN_b, epoch_key, X_train_jax)
    jax.block_until_ready(params)
    print(f"  warm-up epoch {i} done")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Data prep vs epoch_fn split
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("1. DATA PREP vs EPOCH_FN (5 measured epochs)")
print("="*60)

N_MEASURE = 5
t_prep, t_epoch = [], []

for i in range(N_MEASURE):
    t0 = time.perf_counter()
    Xs_b, KNN_b = make_batch(seed=i + 10)
    t_prep.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    params, opt_state, *_ = epoch_fn(params, opt_state, Xs_b, KNN_b, epoch_key, X_train_jax)
    jax.block_until_ready(params)
    t_epoch.append(time.perf_counter() - t0)

    print(f"  epoch {i}: prep={t_prep[-1]*1000:.0f}ms  epoch_fn={t_epoch[-1]*1000:.0f}ms")

print(f"\n  mean data prep : {np.mean(t_prep)*1000:.0f} ms")
print(f"  mean epoch_fn  : {np.mean(t_epoch)*1000:.0f} ms")
print(f"  prep share     : {np.mean(t_prep)/(np.mean(t_prep)+np.mean(t_epoch))*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Single-batch loss component breakdown
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("2. SINGLE-BATCH LOSS COMPONENT BREAKDOWN")
print("="*60)

x_batch   = jnp.array(X_train[:BATCH_SIZE])
knn_batch = jnp.array(knn_idx[:BATCH_SIZE])
key_b     = jax.random.PRNGKey(0)
x_norm    = x_batch / 255.0

# 2a. encode
ms = T("encode", lambda: encode(params, x_norm), reps=50)
print(f"  encode (B={BATCH_SIZE})                   : {ms:.2f} ms")

# 2b. full decode — recon path (B=64)
z       = encode(params, x_norm)
cats_r  = jnp.zeros((1, N_CAT), dtype=jnp.int32)
rc      = jnp.broadcast_to(params["recon_emb"] + cats_to_cond(params, cats_r), (BATCH_SIZE, COND_DIM))
zc_r    = jnp.concatenate([z, rc], axis=-1)
ms = T("decode recon (B=64)", lambda: decode(params, zc_r), reps=50)
print(f"  decode recon (B={BATCH_SIZE})                : {ms:.2f} ms")

# 2c. decode — variation path (B*N_VAR = 2048)
cats_v  = jax.random.randint(key_b, (N_VAR_SAMPLES, N_CAT), 0, VOCAB_SIZE)
cond_v  = params["recon_emb"][None] + cats_to_cond(params, cats_v)
z_rep   = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(BATCH_SIZE * N_VAR_SAMPLES, LATENT_DIM)
c_rep   = jnp.broadcast_to(cond_v[None], (BATCH_SIZE, N_VAR_SAMPLES, COND_DIM)).reshape(BATCH_SIZE * N_VAR_SAMPLES, COND_DIM)
zc_v    = jnp.concatenate([z_rep, c_rep], axis=-1)
ms = T(f"decode variation (B*N={BATCH_SIZE*N_VAR_SAMPLES})", lambda: decode(params, zc_v), reps=20)
print(f"  decode variation (B*N={BATCH_SIZE*N_VAR_SAMPLES})           : {ms:.2f} ms")

# 2d. kNN gather + distance computation
Y_all    = decode(params, zc_v).reshape(BATCH_SIZE, N_VAR_SAMPLES, 784)
knn_imgs = (X_train_jax / 255.0)[knn_batch]  # (B, K_NN, 784)

def dist_step():
    sq_knn = (knn_imgs ** 2).sum(-1)
    sq_Y   = (Y_all ** 2).sum(-1)
    dot    = jnp.einsum("bkd,bvd->bkv", knn_imgs, Y_all)
    dists  = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot
    return jnp.argmin(dists, axis=-1)

ms = T("kNN distance + argmin", dist_step, reps=50)
print(f"  kNN dist + argmin (B={BATCH_SIZE},K={K_NN},V={N_VAR_SAMPLES})   : {ms:.2f} ms")

# 2e. full loss_fn (fwd only)
@jax.jit
def fwd_loss():
    return loss_fn(params, x_batch, knn_batch, key_b, X_train_jax)

ms = T("full loss_fn (fwd)", fwd_loss, reps=20)
print(f"  full loss_fn fwd                      : {ms:.2f} ms")

# 2f. full loss_fn + grad
@jax.jit
def fwd_bwd_loss():
    return jax.value_and_grad(loss_fn, has_aux=True)(params, x_batch, knn_batch, key_b, X_train_jax)

ms = T("full loss_fn (fwd+bwd)", fwd_bwd_loss, reps=20)
print(f"  full loss_fn fwd+bwd                  : {ms:.2f} ms")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Data prep breakdown
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("3. DATA PREP BREAKDOWN")
print("="*60)

rng  = np.random.RandomState(42)

t0 = time.perf_counter()
for _ in range(20):
    perm = rng.permutation(len(X_train))
print(f"  np.permutation (x20 mean)   : {(time.perf_counter()-t0)/20*1000:.1f} ms")

t0 = time.perf_counter()
n  = num_batches * BATCH_SIZE
for _ in range(20):
    perm = rng.permutation(len(X_train))
    _ = X_train[perm[:n]].reshape(num_batches, BATCH_SIZE, 784)
print(f"  X shuffle+reshape (x20)     : {(time.perf_counter()-t0)/20*1000:.1f} ms")

t0 = time.perf_counter()
for _ in range(20):
    perm = rng.permutation(len(X_train))
    _ = knn_idx[perm[:n]].reshape(num_batches, BATCH_SIZE, K_NN)
print(f"  knn_idx shuffle+reshape (x20): {(time.perf_counter()-t0)/20*1000:.1f} ms")

t0 = time.perf_counter()
for _ in range(20):
    perm = rng.permutation(len(X_train))
    Xs  = X_train[perm[:n]].reshape(num_batches, BATCH_SIZE, 784)
    KNN = knn_idx[perm[:n]].reshape(num_batches, BATCH_SIZE, K_NN)
    jax.block_until_ready(jnp.array(Xs))
    jax.block_until_ready(jnp.array(KNN))
print(f"  full prep incl jnp.array (x20): {(time.perf_counter()-t0)/20*1000:.1f} ms")

print("\nDone.")
