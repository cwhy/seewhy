"""
exp26 — DeltaNet Autoregressive Decoder.

exp23–25 broke the 0.0046 plateau (LATENT=512 → 0.0022 MSE), but the deconvnet
decoder generates all pixels in one parallel pass. An autoregressive decoder models
p(x|z, style) as a product of conditionals, capturing texture/fine detail.

DeltaNet (delta-rule linear RNN) over 64 patch tokens (4×4 RGB = 48-d each).
Matrix-valued state W ∈ R^{DM×DK}, O(T) training via lax.scan.

Usage:
    uv run python projects/variations/experiments26.py
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

EXP_NAME = "exp26"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM     = 512
N_CAT          = 4
VOCAB_SIZE     = 8
E_DIM          = 4
COND_DIM       = N_CAT * E_DIM   # 16
IN_DIM         = LATENT_DIM + COND_DIM  # 528
ALPHA          = 1.0
BATCH_SIZE     = 128
N_VAR_SAMPLES  = 64
MAX_EPOCHS     = 100
ADAM_LR        = 3e-4
SEED           = 42
K_NN           = 16

# CNN encoder (unchanged from exp23)
ENC_C1         = 64
ENC_C2         = 128
ENC_C3         = 256
ENC_FC1        = 512

# DeltaNet decoder
DM             = 256    # model dimension (= DV for residual connections)
DK             = 64     # key dimension for delta rule
N_LAYERS       = 2      # stacked DeltaNet + FF layers
PATCH_SIZE     = 4      # 4×4 patches
PATCH_DIM      = 48     # = 4*4*3
T_SEQ          = 64     # sequence length = (32//4)^2 = 64

IMG_FLAT       = 3072
EVAL_EVERY     = 5

JSONL = Path(__file__).parent / "results.jsonl"

_rng_eval = np.random.RandomState(0)
EVAL_COND_CATS = jnp.array(
    _rng_eval.randint(0, VOCAB_SIZE, (256, N_CAT)), dtype=jnp.int32
)

# ── Utilities ──────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Patch helpers ──────────────────────────────────────────────────────────────

def extract_patches(x_flat):
    """(B, 3072) → (B, 64, 48): raster-order 4×4 patches."""
    B = x_flat.shape[0]
    x = x_flat.reshape(B, 3, 32, 32).transpose(0, 2, 3, 1)          # (B,32,32,3)
    x = x.reshape(B, 8, 4, 8, 4, 3).transpose(0, 1, 3, 2, 4, 5)    # (B,8,8,4,4,3)
    return x.reshape(B, T_SEQ, PATCH_DIM)

def merge_patches(patches):
    """(B, 64, 48) → (B, 3072)."""
    B = patches.shape[0]
    x = patches.reshape(B, 8, 8, 4, 4, 3).transpose(0, 1, 3, 2, 4, 5)  # (B,8,4,8,4,3)
    return x.reshape(B, 32, 32, 3).transpose(0, 3, 1, 2).reshape(B, IMG_FLAT)

def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    std  = jnp.sqrt(((x - mean) ** 2).mean(-1, keepdims=True) + eps)
    return g * (x - mean) / std + b

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    p = {
        "cat_emb":   n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        "recon_emb": n((COND_DIM,),                scale=0.1),
        # Encoder (identical to exp23)
        "enc_c1": n((3, 3,     3, ENC_C1), scale=(3*3*3    )**-0.5),
        "enc_b1": jnp.zeros(ENC_C1),
        "enc_c2": n((3, 3, ENC_C1, ENC_C2), scale=(3*3*ENC_C1)**-0.5),
        "enc_b2": jnp.zeros(ENC_C2),
        "enc_c3": n((3, 3, ENC_C2, ENC_C3), scale=(3*3*ENC_C2)**-0.5),
        "enc_b3": jnp.zeros(ENC_C3),
        "enc_W1": n((ENC_C3 * 4 * 4, ENC_FC1), scale=(ENC_C3*16)**-0.5),
        "enc_b4": jnp.zeros(ENC_FC1),
        "enc_W2": n((ENC_FC1, LATENT_DIM), scale=ENC_FC1**-0.5),
        "enc_b5": jnp.zeros(LATENT_DIM),
        # Decoder input/output projections
        "dc_pin":    n((PATCH_DIM, DM), scale=PATCH_DIM**-0.5),
        "dc_pbin":   jnp.zeros(DM),
        "dc_stok":   n((DM,), scale=0.02),
        "dc_posemb": n((T_SEQ, DM), scale=0.02),
        "dc_outW":   n((DM, PATCH_DIM), scale=DM**-0.5),
        "dc_outb":   jnp.zeros(PATCH_DIM),
        "dc_lnog":   jnp.ones(DM),
        "dc_lnob":   jnp.zeros(DM),
    }
    for i in range(N_LAYERS):
        p.update({
            f"dn{i}_Wq":   n((DM, DK),     scale=DM**-0.5),
            f"dn{i}_bq":   jnp.zeros(DK),
            f"dn{i}_cWq":  n((IN_DIM, DK), scale=IN_DIM**-0.5),
            f"dn{i}_Wk":   n((DM, DK),     scale=DM**-0.5),
            f"dn{i}_bk":   jnp.zeros(DK),
            f"dn{i}_cWk":  n((IN_DIM, DK), scale=IN_DIM**-0.5),
            f"dn{i}_Wv":   n((DM, DM),     scale=DM**-0.5),
            f"dn{i}_bv":   jnp.zeros(DM),
            f"dn{i}_cWv":  n((IN_DIM, DM), scale=IN_DIM**-0.5),
            f"dn{i}_Wb":   n((DM, 1),      scale=DM**-0.5),
            f"dn{i}_bb":   jnp.zeros(1),
            f"dn{i}_cWb":  n((IN_DIM, 1),  scale=IN_DIM**-0.5),
            f"dn{i}_ff1":  n((DM, 4*DM),   scale=DM**-0.5),
            f"dn{i}_bf1":  jnp.zeros(4*DM),
            f"dn{i}_ff2":  n((4*DM, DM),   scale=(4*DM)**-0.5),
            f"dn{i}_bf2":  jnp.zeros(DM),
            f"dn{i}_ln1g": jnp.ones(DM),
            f"dn{i}_ln1b": jnp.zeros(DM),
            f"dn{i}_ln2g": jnp.ones(DM),
            f"dn{i}_ln2b": jnp.zeros(DM),
        })
    return p

def cats_to_cond(params, cats):
    embs = params["cat_emb"][jnp.arange(N_CAT)[None, :], cats]
    return embs.reshape(cats.shape[0], COND_DIM)

_ENC_DIMS = ("NHWC", "HWIO", "NHWC")

def _conv_stride2(x, W, b):
    return jax.lax.conv_general_dilated(
        x, W, (2, 2), "SAME", dimension_numbers=_ENC_DIMS
    ) + b

def encode(params, x_flat):
    B = x_flat.shape[0]
    h = x_flat.reshape(B, 32, 32, 3)
    h = jax.nn.gelu(_conv_stride2(h, params["enc_c1"], params["enc_b1"]))
    h = jax.nn.gelu(_conv_stride2(h, params["enc_c2"], params["enc_b2"]))
    h = jax.nn.gelu(_conv_stride2(h, params["enc_c3"], params["enc_b3"]))
    h = h.reshape(B, -1)
    h = jax.nn.gelu(h @ params["enc_W1"] + params["enc_b4"])
    return h @ params["enc_W2"] + params["enc_b5"]

# ── DeltaNet decoder ───────────────────────────────────────────────────────────

def _dn_step(W0, W1, x_emb, params, cond_q0, cond_k0, cond_v0, cond_b0,
             cond_q1, cond_k1, cond_v1, cond_b1):
    """Apply 2 DeltaNet+FF layers. Returns (W0_new, W1_new, h)."""
    h = x_emb

    # ── Layer 0 ──────────────────────────────────────────────────────────────
    x_ln = layer_norm(h, params["dn0_ln1g"], params["dn0_ln1b"])
    q    = x_ln @ params["dn0_Wq"] + params["dn0_bq"] + cond_q0          # (B, DK)
    k    = x_ln @ params["dn0_Wk"] + params["dn0_bk"] + cond_k0          # (B, DK)
    k    = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    v    = x_ln @ params["dn0_Wv"] + params["dn0_bv"] + cond_v0          # (B, DM)
    beta = jax.nn.sigmoid(x_ln @ params["dn0_Wb"] + params["dn0_bb"] + cond_b0)  # (B,1)
    Wk   = jnp.einsum("bde,be->bd", W0, k)
    W0   = W0 + beta[:, :, None] * jnp.einsum("bd,be->bde", v - Wk, k)
    h    = h + jnp.einsum("bde,be->bd", W0, q)

    h_ln = layer_norm(h, params["dn0_ln2g"], params["dn0_ln2b"])
    h    = h + jax.nn.gelu(h_ln @ params["dn0_ff1"] + params["dn0_bf1"]) @ params["dn0_ff2"] + params["dn0_bf2"]

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    x_ln = layer_norm(h, params["dn1_ln1g"], params["dn1_ln1b"])
    q    = x_ln @ params["dn1_Wq"] + params["dn1_bq"] + cond_q1
    k    = x_ln @ params["dn1_Wk"] + params["dn1_bk"] + cond_k1
    k    = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    v    = x_ln @ params["dn1_Wv"] + params["dn1_bv"] + cond_v1
    beta = jax.nn.sigmoid(x_ln @ params["dn1_Wb"] + params["dn1_bb"] + cond_b1)
    Wk   = jnp.einsum("bde,be->bd", W1, k)
    W1   = W1 + beta[:, :, None] * jnp.einsum("bd,be->bde", v - Wk, k)
    h    = h + jnp.einsum("bde,be->bd", W1, q)

    h_ln = layer_norm(h, params["dn1_ln2g"], params["dn1_ln2b"])
    h    = h + jax.nn.gelu(h_ln @ params["dn1_ff1"] + params["dn1_bf1"]) @ params["dn1_ff2"] + params["dn1_bf2"]

    return W0, W1, h


def decode_tf(params, zc, x_flat):
    """Teacher-forced decode. (B, IN_DIM), (B, 3072) → (B, 3072)."""
    B = zc.shape[0]

    # Precompute per-layer condition biases (constant across T)
    cq0 = zc @ params["dn0_cWq"]   # (B, DK)
    ck0 = zc @ params["dn0_cWk"]
    cv0 = zc @ params["dn0_cWv"]   # (B, DM)
    cb0 = zc @ params["dn0_cWb"]   # (B, 1)
    cq1 = zc @ params["dn1_cWq"]
    ck1 = zc @ params["dn1_cWk"]
    cv1 = zc @ params["dn1_cWv"]
    cb1 = zc @ params["dn1_cWb"]

    pos_embs   = params["dc_posemb"]                                      # (T_SEQ, DM)
    x_patches  = extract_patches(x_flat)                                  # (B, T_SEQ, 48)

    # t=0: start token + pos_emb[0]; t=1..T-1: patch_embed(patches[:, t-1]) + pos_emb[t]
    start      = jnp.broadcast_to(
        (params["dc_stok"] + pos_embs[0])[None, None, :], (B, 1, DM))
    patch_emb  = x_patches[:, :T_SEQ-1] @ params["dc_pin"] + params["dc_pbin"]  # (B,63,DM)
    patch_emb  = patch_emb + pos_embs[None, 1:T_SEQ, :]
    inputs_t   = jnp.concatenate([start, patch_emb], axis=1).transpose(1, 0, 2)  # (T,B,DM)

    init_carry = (jnp.zeros((B, DM, DK)), jnp.zeros((B, DM, DK)))

    def tf_step(carry, x_emb):
        W0, W1 = carry
        W0, W1, h = _dn_step(W0, W1, x_emb, params, cq0, ck0, cv0, cb0, cq1, ck1, cv1, cb1)
        out_ln = layer_norm(h, params["dc_lnog"], params["dc_lnob"])
        patch  = jax.nn.sigmoid(out_ln @ params["dc_outW"] + params["dc_outb"])   # (B,48)
        return (W0, W1), patch

    (_, _), patches = jax.lax.scan(jax.remat(tf_step), init_carry, inputs_t)  # (T,B,48)
    return merge_patches(patches.transpose(1, 0, 2))                      # (B,3072)


def decode_ar(params, zc):
    """Autoregressive decode (greedy). (B, IN_DIM) → (B, 3072)."""
    B = zc.shape[0]

    cq0 = zc @ params["dn0_cWq"]
    ck0 = zc @ params["dn0_cWk"]
    cv0 = zc @ params["dn0_cWv"]
    cb0 = zc @ params["dn0_cWb"]
    cq1 = zc @ params["dn1_cWq"]
    ck1 = zc @ params["dn1_cWk"]
    cv1 = zc @ params["dn1_cWv"]
    cb1 = zc @ params["dn1_cWb"]

    pos_embs   = params["dc_posemb"]                                      # (T_SEQ, DM)
    start_emb  = jnp.broadcast_to(
        (params["dc_stok"] + pos_embs[0])[None, :], (B, DM))

    # xs[t] = pos_emb for the NEXT step (used to build the next carry)
    next_pos = jnp.concatenate([pos_embs[1:], pos_embs[-1:]], axis=0)    # (T_SEQ, DM)

    init_carry = (jnp.zeros((B, DM, DK)), jnp.zeros((B, DM, DK)), start_emb)

    def ar_step(carry, nxt_pos):
        W0, W1, x_emb = carry
        W0, W1, h = _dn_step(W0, W1, x_emb, params, cq0, ck0, cv0, cb0, cq1, ck1, cv1, cb1)
        out_ln    = layer_norm(h, params["dc_lnog"], params["dc_lnob"])
        patch     = jax.nn.sigmoid(out_ln @ params["dc_outW"] + params["dc_outb"])  # (B,48)
        next_emb  = patch @ params["dc_pin"] + params["dc_pbin"] + nxt_pos[None, :]
        return (W0, W1, next_emb), patch

    (_, _, _), patches = jax.lax.scan(ar_step, init_carry, next_pos)     # (T,B,48)
    return merge_patches(patches.transpose(1, 0, 2))                      # (B,3072)

# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(params, x_batch, knn_idx_batch, key, X_train):
    B      = x_batch.shape[0]
    x_norm = x_batch / 255.0
    z      = encode(params, x_norm)

    cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond = jnp.broadcast_to(
        params["recon_emb"] + cats_to_cond(params, cats_recon), (B, COND_DIM))
    L_recon = ((decode_tf(params, jnp.concatenate([z, recon_cond], axis=-1), x_norm) - x_norm) ** 2).mean()

    # IMLE assignment: sample N_VAR styles, decode AR (no gradient), find best per kNN
    cats     = jax.random.randint(key, (N_VAR_SAMPLES, N_CAT), 0, VOCAB_SIZE)
    cond_var = params["recon_emb"][None] + cats_to_cond(params, cats)    # (N_VAR, COND_DIM)
    z_rep    = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(B * N_VAR_SAMPLES, LATENT_DIM)
    c_rep    = jnp.broadcast_to(cond_var[None], (B, N_VAR_SAMPLES, COND_DIM)).reshape(B * N_VAR_SAMPLES, COND_DIM)
    Y_all    = jax.lax.stop_gradient(
        decode_ar(params, jnp.concatenate([z_rep, c_rep], axis=-1))
    ).reshape(B, N_VAR_SAMPLES, IMG_FLAT)

    knn_imgs  = (X_train / 255.0)[knn_idx_batch]                         # (B, K_NN, 3072)
    sq_knn    = (knn_imgs ** 2).sum(-1)
    sq_Y      = (Y_all  ** 2).sum(-1)
    dot       = jnp.einsum("bkd,bvd->bkv", knn_imgs, Y_all)
    dists     = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot
    best_slot = jnp.argmin(dists, axis=-1)                               # (B, K_NN)

    # Teacher-forced MSE for the selected (style, kNN-neighbor) pairs.
    # Process one kNN slot at a time via lax.map to avoid storing T=64 W-carries
    # for B*K_NN=2048 examples simultaneously (~8 GB). Each map iteration uses B=128.
    best_cats  = cats[best_slot]                                          # (B, K_NN, N_CAT)
    best_c_all = params["recon_emb"][None] + cats_to_cond(
        params, best_cats.reshape(B * K_NN, N_CAT))                      # (B*K_NN, COND_DIM)

    z_by_knn   = jnp.repeat(z[:, None, :], K_NN, axis=1).transpose(1, 0, 2)          # (K_NN,B,LATENT)
    c_by_knn   = best_c_all.reshape(B, K_NN, COND_DIM).transpose(1, 0, 2)            # (K_NN,B,COND)
    img_by_knn = knn_imgs.transpose(1, 0, 2)                                          # (K_NN,B,3072)

    def loss_one_knn(tup):
        z_k, c_k, img_k = tup
        pred = decode_tf(params, jnp.concatenate([z_k, c_k], axis=-1), img_k)
        return ((pred - jax.lax.stop_gradient(img_k)) ** 2).mean()

    # jax.remat so backward recomputes one kNN slot at a time: only T*B W-carries
    # in memory (~536 MB) rather than K_NN*T*B (~8 GB).
    L_var = jax.lax.map(jax.remat(loss_one_knn), (z_by_knn, c_by_knn, img_by_knn)).mean()

    return L_recon + ALPHA * L_var, (L_recon, L_var)

# ── Training loop ─────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    @jax.jit
    def epoch_fn(params, opt_state, Xs_batched, KNN_batched, epoch_key, X_train):
        def scan_body(carry, inputs):
            params, opt_state = carry
            x_batch, knn_idx_batch, key = inputs
            (loss, (L_recon, L_var)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, x_batch, knn_idx_batch, key, X_train
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), (loss, L_recon, L_var)

        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (params, opt_state), (losses, l_recon, l_var) = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_batched, KNN_batched, keys)
        )
        return params, opt_state, losses.mean(), l_recon.mean(), l_var.mean()

    return epoch_fn

@jax.jit
def _eval_batch(params, x_batch):
    x_norm     = x_batch / 255.0
    B          = x_batch.shape[0]
    z          = encode(params, x_norm)
    cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond = jnp.broadcast_to(
        params["recon_emb"] + cats_to_cond(params, cats_recon), (B, COND_DIM))
    x_recon    = decode_ar(params, jnp.concatenate([z, recon_cond], axis=-1))
    return ((x_recon - x_norm) ** 2).mean()

def eval_recon_mse(params, X_te):
    mses = [float(_eval_batch(params, jnp.array(X_te[i:i+256])))
            for i in range(0, len(X_te), 256)]
    return float(np.mean(mses))

EVAL_RECALL_B = 8

@jax.jit
def _recall_batch_jit(params, x_norm, knn_norm, all_conds):
    B       = x_norm.shape[0]
    N_SLOTS = all_conds.shape[0]
    z     = encode(params, x_norm)
    z_rep = jnp.repeat(z[:, None, :], N_SLOTS, axis=1).reshape(B * N_SLOTS, LATENT_DIM)
    c_rep = jnp.broadcast_to(all_conds[None], (B, N_SLOTS, COND_DIM)).reshape(B * N_SLOTS, COND_DIM)
    Y_all = decode_ar(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_SLOTS, IMG_FLAT)

    sq_knn = (knn_norm ** 2).sum(-1)
    sq_Y   = (Y_all ** 2).sum(-1)
    dot    = jnp.einsum("bkd,bvd->bkv", knn_norm, Y_all)
    dists  = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot
    best   = jnp.argmin(dists, axis=-1)
    matched = Y_all[jnp.arange(B)[:, None], best]

    sq_x       = (x_norm ** 2).sum(-1)
    dot_xk     = jnp.einsum("bd,bkd->bk", x_norm, knn_norm)
    dist_x_knn = sq_x[:, None] + sq_knn - 2.0 * dot_xk
    rank_order = jnp.argsort(dist_x_knn, axis=-1)

    sorted_matched = matched[jnp.arange(B)[:, None], rank_order]
    sorted_knn     = knn_norm[jnp.arange(B)[:, None], rank_order]

    per_rank = ((sorted_matched - sorted_knn) ** 2).mean(axis=-1).mean(axis=0)
    return per_rank.mean(), per_rank

def eval_recall_mse(params, X_tr, knn_idx, n_images=512):
    all_conds = params["recon_emb"] + cats_to_cond(params, EVAL_COND_CATS)
    X_norm    = (X_tr / 255.0).astype(np.float32)
    n         = (n_images // EVAL_RECALL_B) * EVAL_RECALL_B
    means, per_ranks = [], []
    for start in range(0, n, EVAL_RECALL_B):
        sl       = slice(start, start + EVAL_RECALL_B)
        x_norm   = jnp.array(X_norm[sl])
        knn_norm = jnp.array(X_norm[knn_idx[sl]])
        mean_r, rank_r = _recall_batch_jit(params, x_norm, knn_norm, all_conds)
        means.append(float(mean_r))
        per_ranks.append(np.array(rank_r))
    return float(np.mean(means)), np.mean(per_ranks, axis=0).tolist()

def train(params, X_tr, X_te, knn_idx):
    num_batches = len(X_tr) // BATCH_SIZE
    num_steps   = MAX_EPOCHS * num_batches
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    X_train_jax = jnp.array(X_tr)

    history = {
        "loss": [], "L_recon": [], "L_var": [],
        "eval_mse": [], "eval_recall_mse": [], "eval_recall_by_rank": [],
    }
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        n    = num_batches * BATCH_SIZE
        Xs_batched  = jnp.array(X_tr[perm[:n]].reshape(num_batches, BATCH_SIZE, IMG_FLAT))
        KNN_batched = jnp.array(knn_idx[perm[:n]].reshape(num_batches, BATCH_SIZE, K_NN))

        params, opt_state, loss, l_recon, l_var = epoch_fn(
            params, opt_state, Xs_batched, KNN_batched, key, X_train_jax
        )
        history["loss"].append(float(loss))
        history["L_recon"].append(float(l_recon))
        history["L_var"].append(float(l_var))

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            mse                        = eval_recon_mse(params, X_te)
            recall_mse, recall_by_rank = eval_recall_mse(params, X_tr, knn_idx)
            history["eval_mse"].append((epoch, mse))
            history["eval_recall_mse"].append((epoch, recall_mse))
            history["eval_recall_by_rank"].append((epoch, recall_by_rank))
            eval_str = f"  eval_mse={mse:.5f}  recall_mse={recall_mse:.5f}"

        logging.info(
            f"epoch {epoch:3d}  loss={float(loss):.5f}"
            f"  L_recon={float(l_recon):.5f}  L_var={float(l_var):.5f}"
            f"{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

    return params, history, time.perf_counter() - t0

if __name__ == "__main__":
    setup_logging(EXP_NAME)
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().strip().splitlines():
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping"); exit(0)

    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("cifar10")
    X_train = data.X.reshape(data.n_samples,           IMG_FLAT).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, IMG_FLAT).astype(np.float32)

    KNN64_CACHE = Path(__file__).parent / "knn_cifar_k64.npy"
    if not KNN64_CACHE.exists():
        raise FileNotFoundError(f"Need {KNN64_CACHE} — run exp16 first to build it")
    knn_idx = np.load(KNN64_CACHE)[:, :K_NN]
    logging.info(f"kNN ready (sliced from k64): {knn_idx.shape}")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(
        f"Parameters: {n_params(params):,}  "
        f"LATENT={LATENT_DIM}  N_VAR={N_VAR_SAMPLES}  K_NN={K_NN}  B={BATCH_SIZE}"
        f"  coverage_ratio={N_VAR_SAMPLES/K_NN:.1f}x"
        f"  encoder=CNN  decoder=deltanet-patch64"
        f"  DM={DM}  DK={DK}  N_LAYERS={N_LAYERS}  T_SEQ={T_SEQ}"
        f"  dataset=cifar10"
    )

    params, history, elapsed = train(params, X_train, X_test, knn_idx)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_recon_mse(params, X_test)
    append_result({
        "experiment": EXP_NAME,
        "name": f"LATENT=512 deltanet-patch64 N_VAR=64 K_NN=16 CIFAR-10",
        "n_params": n_params(params), "epochs": MAX_EPOCHS,
        "time_s": round(elapsed, 1),
        "final_recon_mse": round(float(final_mse), 6),
        "loss_curve":    [round(v, 6) for v in history["loss"]],
        "L_recon_curve": [round(v, 6) for v in history["L_recon"]],
        "L_var_curve":   [round(v, 6) for v in history["L_var"]],
        "eval_mse_curve":            history["eval_mse"],
        "eval_recall_mse_curve":     history["eval_recall_mse"],
        "eval_recall_by_rank_curve": history["eval_recall_by_rank"],
        "latent_dim": LATENT_DIM, "n_cat": N_CAT, "vocab_size": VOCAB_SIZE,
        "e_dim": E_DIM, "cond_dim": COND_DIM, "n_var_samples": N_VAR_SAMPLES,
        "k_nn": K_NN, "enc_c1": ENC_C1, "enc_c2": ENC_C2, "enc_c3": ENC_C3,
        "enc_fc1": ENC_FC1, "dm": DM, "dk": DK, "n_layers": N_LAYERS,
        "patch_size": PATCH_SIZE, "t_seq": T_SEQ, "alpha": ALPHA, "adam_lr": ADAM_LR,
        "batch_size": BATCH_SIZE, "encoder": "cnn", "decoder": "deltanet-patch64",
        "dataset": "cifar10", "img_flat": IMG_FLAT,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s")
