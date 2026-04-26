"""
Targeted profiling of the kNN matching steps inside loss_fn:
  1. Gather knn_imgs from full X_train (the index op)
  2. einsum distance (the matmul)
  3. argmin (the matching)
  4. All three together vs full loss fwd vs full loss fwd+bwd

Usage:
    uv run python scripts/tmp/profile_exp12_knn.py
"""

import sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "variations"))

import jax
import jax.numpy as jnp
import numpy as np

from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image
from experiments12 import (
    init_params, encode, decode, cats_to_cond, loss_fn,
    BATCH_SIZE, N_VAR_SAMPLES, K_NN, LATENT_DIM,
    N_CAT, VOCAB_SIZE, COND_DIM, SEED,
)

def T(label, fn, reps=100):
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {label:<45}: {ms:.3f} ms")
    return ms

# ── Setup ──────────────────────────────────────────────────────────────────────

print("Loading data...")
data      = load_supervised_image("mnist")
X_train   = data.X.reshape(data.n_samples, 784).astype(np.float32)
knn_idx   = np.load(ROOT / "projects/variations/knn_mnist_k64.npy")

key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
params    = init_params(key_gen)

X_train_jax = jnp.array(X_train)          # (60000, 784) on GPU
x_batch     = jnp.array(X_train[:BATCH_SIZE])
knn_batch   = jnp.array(knn_idx[:BATCH_SIZE])   # (64, 64) indices
key_b       = jax.random.PRNGKey(0)

# Build Y_all (B, N_VAR, 784) — variation outputs
x_norm    = x_batch / 255.0
z         = encode(params, x_norm)
cats_v    = jax.random.randint(key_b, (N_VAR_SAMPLES, N_CAT), 0, VOCAB_SIZE)
cond_v    = params["recon_emb"][None] + cats_to_cond(params, cats_v)
z_rep     = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(BATCH_SIZE * N_VAR_SAMPLES, LATENT_DIM)
c_rep     = jnp.broadcast_to(cond_v[None], (BATCH_SIZE, N_VAR_SAMPLES, COND_DIM)).reshape(BATCH_SIZE * N_VAR_SAMPLES, COND_DIM)
zc_v      = jnp.concatenate([z_rep, c_rep], axis=-1)
Y_all     = decode(params, zc_v).reshape(BATCH_SIZE, N_VAR_SAMPLES, 784)

# Pre-gathered knn_imgs (bypass gather cost)
knn_imgs_pre = (X_train_jax / 255.0)[knn_batch]   # (64, 64, 784)

print(f"\nShapes: x_batch={x_batch.shape}  knn_batch={knn_batch.shape}  "
      f"X_train={X_train_jax.shape}  Y_all={Y_all.shape}  knn_imgs={knn_imgs_pre.shape}")

print("\n" + "="*60)
print("KNN MATCHING BREAKDOWN")
print("="*60)

# 1. Gather only
@jax.jit
def gather_only():
    return (X_train_jax / 255.0)[knn_batch]

t_gather = T("gather knn_imgs from X_train (64,64,784)", gather_only)

# 2. einsum distance only (pre-gathered)
@jax.jit
def einsum_only():
    sq_knn = (knn_imgs_pre ** 2).sum(-1)
    sq_Y   = (Y_all ** 2).sum(-1)
    return jnp.einsum("bkd,bvd->bkv", knn_imgs_pre, Y_all)

t_einsum = T("einsum dist (B=64,K=64,V=32,D=784)", einsum_only)

# 3. argmin only (distance already computed)
sq_knn = (knn_imgs_pre ** 2).sum(-1)
sq_Y   = (Y_all ** 2).sum(-1)
dot    = jnp.einsum("bkd,bvd->bkv", knn_imgs_pre, Y_all)
dists  = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot

@jax.jit
def argmin_only():
    return jnp.argmin(dists, axis=-1)

t_argmin = T("argmin (B=64,K=64,V=32)", argmin_only)

# 4. full knn matching (gather + dist + argmin)
@jax.jit
def full_knn_matching():
    knn_imgs = (X_train_jax / 255.0)[knn_batch]
    sq_knn   = (knn_imgs ** 2).sum(-1)
    sq_Y     = (Y_all ** 2).sum(-1)
    dot      = jnp.einsum("bkd,bvd->bkv", knn_imgs, Y_all)
    dists    = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot
    return jnp.argmin(dists, axis=-1)

t_full_knn = T("full kNN match (gather+dist+argmin)", full_knn_matching)

# 5. decode variation (for comparison)
@jax.jit
def decode_var():
    return decode(params, zc_v)

t_decode = T("decode variation (B*N=2048)", decode_var)

# 6. full loss fwd
@jax.jit
def fwd():
    return loss_fn(params, x_batch, knn_batch, key_b, X_train_jax)

t_fwd = T("full loss_fn fwd", fwd, reps=50)

# 7. full loss fwd+bwd
@jax.jit
def fwd_bwd():
    return jax.value_and_grad(loss_fn, has_aux=True)(
        params, x_batch, knn_batch, key_b, X_train_jax)

t_fwdbwd = T("full loss_fn fwd+bwd", fwd_bwd, reps=50)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  gather alone          : {t_gather:.2f} ms")
print(f"  einsum (dist)         : {t_einsum:.2f} ms")
print(f"  argmin                : {t_argmin:.2f} ms")
print(f"  full kNN match        : {t_full_knn:.2f} ms")
print(f"  decode variation      : {t_decode:.2f} ms")
print(f"  ---")
print(f"  loss fwd              : {t_fwd:.2f} ms")
print(f"  loss fwd+bwd          : {t_fwdbwd:.2f} ms")
print(f"  implied backward      : {t_fwdbwd - t_fwd:.2f} ms")
print(f"  937 batches × fwd+bwd : {937 * t_fwdbwd / 1000:.1f} s  (predicted epoch time)")
print()
print(f"  kNN fraction of fwd   : {t_full_knn/t_fwd*100:.1f}%")
print(f"  kNN fraction of fwd+bwd: {t_full_knn/t_fwdbwd*100:.1f}%")
