"""
Benchmark deconvnet decoder vs current Gram decoder.

Deconvnet: MLP stem → reshape (B, C, 7, 7) → ConvTranspose ×2 → (B, 1, 28, 28)
Tries a few channel widths to find the sweet spot of speed vs capacity.

Usage:
    uv run python scripts/tmp/profile_deconvnet.py
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
    N_CAT, VOCAB_SIZE, COND_DIM, IN_DIM, SEED, ALPHA,
    ADAM_LR, MAX_EPOCHS,
)

def T(fn, reps=30):
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / reps * 1000

# ── Data / params ──────────────────────────────────────────────────────────────

print("Loading...")
data        = load_supervised_image("mnist")
X_train     = data.X.reshape(data.n_samples, 784).astype(np.float32)
knn_idx     = np.load(ROOT / "projects/variations/knn_mnist_k64.npy")
X_train_jax = jnp.array(X_train)
key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
params_gram = init_params(key_gen)

x_batch  = jnp.array(X_train[:BATCH_SIZE])
knn_batch = jnp.array(knn_idx[:BATCH_SIZE])
key_b    = jax.random.PRNGKey(0)

# ── Deconvnet helpers ──────────────────────────────────────────────────────────

def conv_transpose(x, w, stride):
    """x: (B,C,H,W)  w: (C_in, C_out, kH, kW) → (B, C_out, H*stride, W*stride)"""
    return jax.lax.conv_transpose(
        x, w, strides=(stride, stride), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )

def init_deconv_params(key_gen, C1, C2, stem_hidden):
    """
    Architecture:
      (B, IN_DIM) → Linear(stem_hidden) → GELU
                  → Linear(C1*7*7) → reshape (B, C1, 7, 7)
                  → ConvTranspose(C1→C2, 4×4, stride=2) → GELU → (B, C2, 14, 14)
                  → ConvTranspose(C2→1,  4×4, stride=2)         → (B, 1,  28, 28)
                  → sigmoid → reshape (B, 784)
    """
    def n(shape, scale):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    return {
        "cat_emb":   n((N_CAT, VOCAB_SIZE, 4), scale=0.1),
        "recon_emb": n((COND_DIM,),            scale=0.1),
        "enc_W1": n((784, 512),        scale=784**-0.5),
        "enc_b1": jnp.zeros(512),
        "enc_W2": n((512, LATENT_DIM), scale=512**-0.5),
        "enc_b2": jnp.zeros(LATENT_DIM),
        # Decoder stem
        "dc_W1": n((IN_DIM, stem_hidden),  scale=IN_DIM**-0.5),
        "dc_b1": jnp.zeros(stem_hidden),
        "dc_W2": n((stem_hidden, C1*7*7),  scale=stem_hidden**-0.5),
        "dc_b2": jnp.zeros(C1*7*7),
        # ConvTranspose weights: (C_in, C_out, kH, kW)
        "dc_ct1": n((C1, C2, 4, 4), scale=(C1*4*4)**-0.5),
        "dc_ct2": n((C2,  1, 4, 4), scale=(C2*4*4)**-0.5),
    }

def make_decode_deconv(C1, C2):
    def decode_deconv(params, zc):
        B = zc.shape[0]
        h = jax.nn.gelu(zc @ params["dc_W1"] + params["dc_b1"])   # (B, stem_hidden)
        h = jax.nn.gelu(h  @ params["dc_W2"] + params["dc_b2"])   # (B, C1*49)
        h = h.reshape(B, C1, 7, 7)                                 # (B, C1, 7, 7)
        h = jax.nn.gelu(conv_transpose(h, params["dc_ct1"], 2))    # (B, C2, 14, 14)
        h = conv_transpose(h, params["dc_ct2"], 2)                 # (B, 1, 28, 28)
        return jax.nn.sigmoid(h.reshape(B, 784))
    return decode_deconv

def make_loss_deconv(decode_fn, batch_size, n_var):
    def _loss(params, x_batch, knn_idx_batch, key, X_train):
        B      = x_batch.shape[0]
        x_norm = x_batch / 255.0
        z      = encode(params, x_norm)

        cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
        recon_cond = jnp.broadcast_to(
            params["recon_emb"] + cats_to_cond(params, cats_recon), (B, COND_DIM))
        x_recon = decode_fn(params, jnp.concatenate([z, recon_cond], axis=-1))
        L_recon = ((x_recon - x_norm) ** 2).mean()

        cats   = jax.random.randint(key, (n_var, N_CAT), 0, VOCAB_SIZE)
        cond_v = params["recon_emb"][None] + cats_to_cond(params, cats)
        z_rep  = jnp.repeat(z[:, None, :], n_var, axis=1).reshape(B*n_var, LATENT_DIM)
        c_rep  = jnp.broadcast_to(cond_v[None], (B, n_var, COND_DIM)).reshape(B*n_var, COND_DIM)
        Y_all  = decode_fn(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, n_var, 784)

        knn_imgs = (X_train / 255.0)[knn_idx_batch]
        sq_knn   = (knn_imgs**2).sum(-1)
        sq_Y     = (Y_all**2).sum(-1)
        dot      = jnp.einsum("bkd,bvd->bkv", knn_imgs, Y_all)
        dists    = sq_knn[:,:,None] + sq_Y[:,None,:] - 2.0*dot
        best     = jnp.argmin(dists, axis=-1)
        matched  = Y_all[jnp.arange(B)[:,None], best]
        L_var    = ((matched - jax.lax.stop_gradient(knn_imgs))**2).mean()
        return L_recon + ALPHA * L_var, (L_recon, L_var)
    return _loss

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

# ── Benchmark ──────────────────────────────────────────────────────────────────

print("\n" + "="*90)
print("DECODE SPEED: Gram (baseline) vs Deconvnet variants  [B*N=2048, B=128, N=16]")
print("="*90)
print(f"  {'config':<35}  {'n_params':>10}  {'decode_fwd':>12}  {'fwd+bwd':>10}  {'epoch':>8}  {'100ep':>8}")
print("-"*90)

B_EXP, N_EXP = 128, 16
steps = 60000 // B_EXP  # 468

x128  = jnp.array(X_train[:B_EXP])
knn128 = jnp.array(knn_idx[:B_EXP])
x_var = jnp.array(X_train[:B_EXP*N_EXP])  # for standalone decode bench

# -- Baseline: Gram decoder (exp12 config, B=128 N=16) -------------------------
gram_loss = make_loss_deconv(decode, B_EXP, N_EXP)  # reuse gram decode

@jax.jit
def gram_decode_fwd():
    z     = encode(params_gram, x128/255.0)
    cats  = jax.random.randint(key_b, (N_EXP, N_CAT), 0, VOCAB_SIZE)
    cond  = params_gram["recon_emb"][None] + cats_to_cond(params_gram, cats)
    z_rep = jnp.repeat(z[:,None,:], N_EXP, axis=1).reshape(B_EXP*N_EXP, LATENT_DIM)
    c_rep = jnp.broadcast_to(cond[None], (B_EXP, N_EXP, COND_DIM)).reshape(B_EXP*N_EXP, COND_DIM)
    return decode(params_gram, jnp.concatenate([z_rep, c_rep], axis=-1))

@jax.jit
def gram_fwdbwd():
    return jax.value_and_grad(gram_loss, has_aux=True)(
        params_gram, x128, knn128, key_b, X_train_jax)

t_dec  = T(gram_decode_fwd)
t_step = T(gram_fwdbwd)
epoch_s = steps * t_step / 1000
print(f"  {'Gram (exp12/13 baseline)':<35}  {n_params(params_gram):>10,}  "
      f"{t_dec:>10.1f}ms  {t_step:>8.1f}ms  {epoch_s:>6.1f}s  {epoch_s*100/60:>6.1f}min")

# -- Deconvnet variants --------------------------------------------------------
for C1, C2, stem in [(64, 32, 256), (128, 64, 256), (128, 64, 512), (256, 128, 512)]:
    label = f"DeconvNet C1={C1} C2={C2} stem={stem}"
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 10))
    p  = init_deconv_params(kg, C1, C2, stem)
    dec_fn = make_decode_deconv(C1, C2)
    dc_loss = make_loss_deconv(dec_fn, B_EXP, N_EXP)

    @jax.jit
    def dc_decode_fwd(p=p, dec_fn=dec_fn):
        z     = encode(p, x128/255.0)
        cats  = jax.random.randint(key_b, (N_EXP, N_CAT), 0, VOCAB_SIZE)
        cond  = p["recon_emb"][None] + cats_to_cond(p, cats)
        z_rep = jnp.repeat(z[:,None,:], N_EXP, axis=1).reshape(B_EXP*N_EXP, LATENT_DIM)
        c_rep = jnp.broadcast_to(cond[None], (B_EXP, N_EXP, COND_DIM)).reshape(B_EXP*N_EXP, COND_DIM)
        return dec_fn(p, jnp.concatenate([z_rep, c_rep], axis=-1))

    @jax.jit
    def dc_fwdbwd(p=p, dc_loss=dc_loss):
        return jax.value_and_grad(dc_loss, has_aux=True)(
            p, x128, knn128, key_b, X_train_jax)

    t_dec  = T(dc_decode_fwd)
    t_step = T(dc_fwdbwd)
    epoch_s = steps * t_step / 1000
    print(f"  {label:<35}  {n_params(p):>10,}  "
          f"{t_dec:>10.1f}ms  {t_step:>8.1f}ms  {epoch_s:>6.1f}s  {epoch_s*100/60:>6.1f}min")

print("="*90)
