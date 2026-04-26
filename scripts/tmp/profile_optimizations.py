"""
Benchmark three candidate optimizations against the baseline:

  A. jax.checkpoint on decode  — avoids storing large intermediates (f_all, h1)
     during backward; recomputes them instead
  B. B=128, N_VAR=16           — same B*N=2048, but 469 scan steps vs 937
  C. Both together

For each, reports:
  - fwd+bwd per batch (ms)
  - predicted epoch time (s)
  - predicted 100-epoch time (min)

Usage:
    uv run python scripts/tmp/profile_optimizations.py
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
    BATCH_SIZE, N_VAR_SAMPLES, K_NN, LATENT_DIM,
    N_CAT, VOCAB_SIZE, COND_DIM, SEED, ALPHA,
    ADAM_LR, MAX_EPOCHS,
)

REPS = 30

def T(fn, reps=REPS):
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / reps * 1000

# ── Setup ──────────────────────────────────────────────────────────────────────

print("Loading data...")
data    = load_supervised_image("mnist")
X_train = data.X.reshape(data.n_samples, 784).astype(np.float32)
knn_idx = np.load(ROOT / "projects/variations/knn_mnist_k64.npy")

key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
params      = init_params(key_gen)
X_train_jax = jnp.array(X_train)

x64  = jnp.array(X_train[:64])
x128 = jnp.array(X_train[:128])
knn64  = jnp.array(knn_idx[:64])
knn128 = jnp.array(knn_idx[:128])
key_b  = jax.random.PRNGKey(0)

decode_ckpt = jax.checkpoint(decode)

# ── Shared loss body ───────────────────────────────────────────────────────────

def make_loss(batch_size, n_var, dec_fn):
    def loss_fn(params, x_batch, knn_idx_batch, key, X_train):
        B      = x_batch.shape[0]
        x_norm = x_batch / 255.0
        z      = encode(params, x_norm)

        cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
        recon_cond = jnp.broadcast_to(
            params["recon_emb"] + cats_to_cond(params, cats_recon), (B, COND_DIM))
        x_recon = dec_fn(params, jnp.concatenate([z, recon_cond], axis=-1))
        L_recon = ((x_recon - x_norm) ** 2).mean()

        cats    = jax.random.randint(key, (n_var, N_CAT), 0, VOCAB_SIZE)
        cond_v  = params["recon_emb"][None] + cats_to_cond(params, cats)
        z_rep   = jnp.repeat(z[:, None, :], n_var, axis=1).reshape(B * n_var, LATENT_DIM)
        c_rep   = jnp.broadcast_to(cond_v[None], (B, n_var, COND_DIM)).reshape(B * n_var, COND_DIM)
        Y_all   = dec_fn(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, n_var, 784)

        knn_imgs = (X_train / 255.0)[knn_idx_batch]
        sq_knn   = (knn_imgs ** 2).sum(-1)
        sq_Y     = (Y_all ** 2).sum(-1)
        dot      = jnp.einsum("bkd,bvd->bkv", knn_imgs, Y_all)
        dists    = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot
        best     = jnp.argmin(dists, axis=-1)
        matched  = Y_all[jnp.arange(B)[:, None], best]

        L_var = ((matched - jax.lax.stop_gradient(knn_imgs)) ** 2).mean()
        return L_recon + ALPHA * L_var, (L_recon, L_var)
    return loss_fn

# ── Benchmark helper ───────────────────────────────────────────────────────────

def bench(label, batch_size, n_var, dec_fn, x_batch, knn_batch):
    loss_fn = make_loss(batch_size, n_var, dec_fn)
    batches_per_epoch = 60000 // batch_size

    @jax.jit
    def fwd_bwd():
        return jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_batch, knn_batch, key_b, X_train_jax)

    ms = T(fwd_bwd)
    epoch_s  = batches_per_epoch * ms / 1000
    total_m  = epoch_s * MAX_EPOCHS / 60
    print(f"  {label:<40}  B={batch_size:3d}  N={n_var:2d}  "
          f"steps={batches_per_epoch:4d}  "
          f"fwd+bwd={ms:6.1f}ms  "
          f"epoch={epoch_s:5.1f}s  "
          f"100ep={total_m:5.1f}min")
    return ms, epoch_s

print("\n" + "="*85)
print("OPTIMIZATION COMPARISON  (B*N=2048 decode/step in all cases)")
print("="*85)
print(f"  {'config':<40}  {'B':>4}  {'N':>3}  {'steps':>5}  {'fwd+bwd':>10}  {'epoch':>8}  {'100ep':>8}")
print("-"*85)

# Baseline (exp12)
bench("baseline  (exp12, plain decode)",  64, 32, decode,      x64,  knn64)

# A: checkpoint only
bench("A: checkpoint(decode)",            64, 32, decode_ckpt, x64,  knn64)

# B: larger batch, fewer steps
bench("B: B=128, N=16 (plain decode)",   128, 16, decode,      x128, knn128)

# C: both
bench("C: B=128, N=16 + checkpoint",     128, 16, decode_ckpt, x128, knn128)

print("="*85)
