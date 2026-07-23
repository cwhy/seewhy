"""
exp18 — Extra decoder stage + ResNet encoder on CIFAR-10.

Changes from exp17:
1. Decoder starts at 2×2 instead of 4×4 (one extra ConvTranspose stage).
   DC_C0=512 (2×2→4×4), then DC_C1=256→128→64→3 as usual.
2. ResNet encoder: 2 residual blocks per downsampling stage (no BatchNorm, GELU).
   Stages: 3→64 (stride-2), 64→128 (stride-2), 128→256 (stride-2)
   FC head: 4096→512→128 (same as exp16/17)

Hypothesis: more gradual upsampling (2×2 start) + richer content representation
from residual connections should reduce blur further.

Usage:
    uv run python projects/variations/experiments18.py
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

EXP_NAME = "exp18"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM     = 128
N_CAT          = 4
VOCAB_SIZE     = 8
E_DIM          = 4
COND_DIM       = N_CAT * E_DIM   # 16
IN_DIM         = LATENT_DIM + COND_DIM  # 144
ALPHA          = 1.0
BATCH_SIZE     = 128
N_VAR_SAMPLES  = 16
MAX_EPOCHS     = 100
ADAM_LR        = 3e-4
SEED           = 42
K_NN           = 64

# ResNet encoder channels
ENC_C1         = 64    # 32×32 → 16×16
ENC_C2         = 128   # 16×16 → 8×8
ENC_C3         = 256   # 8×8   → 4×4
ENC_FC1        = 512   # flatten 4096 → 512

# Decoder: extra stage (starts at 2×2), wide channels
DC_STEM        = 1024   # MLP stem width
DC_C0          = 512    # 2×2 stage (new)
DC_C1          = 256    # 4×4 → 8×8
DC_C2          = 128    # 8×8 → 16×16
DC_C3          = 64     # 16×16 → 32×32
# → 4-stage ConvTranspose → (B, 3, 32, 32)

IMG_FLAT       = 3072  # 3 × 32 × 32
EVAL_EVERY     = 5

JSONL = Path(__file__).parent / "results.jsonl"

_rng_eval = np.random.RandomState(0)
EVAL_COND_CATS = jnp.array(
    _rng_eval.randint(0, VOCAB_SIZE, (256, N_CAT)), dtype=jnp.int32
)

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── kNN pre-computation ────────────────────────────────────────────────────────

def compute_knn(X, k=K_NN, chunk=2000):
    X  = np.array(X)
    N  = len(X)
    knn_idx = np.zeros((N, k), dtype=np.int32)
    sq = (X ** 2).sum(1)
    XT = X.T
    logging.info(f"Computing {k}-NN for {N} images (chunk={chunk})…")
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        dists = sq[start:end, None] + sq[None, :] - 2.0 * (X[start:end] @ XT)
        for j in range(end - start):
            dists[j, start + j] = np.inf
        knn_idx[start:end] = np.argpartition(dists, k, axis=1)[:, :k].astype(np.int32)
        if (start // chunk) % 5 == 0:
            logging.info(f"  kNN {end}/{N}")
    return knn_idx

# ── Encoder helpers (NHWC / HWIO) ─────────────────────────────────────────────

_ENC_DIMS = ("NHWC", "HWIO", "NHWC")

def _conv_s2(x, W, b):
    """Stride-2 conv 3×3, SAME padding, NHWC."""
    return jax.lax.conv_general_dilated(
        x, W, (2, 2), "SAME", dimension_numbers=_ENC_DIMS
    ) + b

def _conv_s2_1x1(x, W, b):
    """Stride-2 conv 1×1 (projection shortcut), NHWC."""
    return jax.lax.conv_general_dilated(
        x, W, (2, 2), "SAME", dimension_numbers=_ENC_DIMS
    ) + b

def _conv_s1(x, W, b):
    """Stride-1 conv 3×3, SAME padding, NHWC."""
    return jax.lax.conv_general_dilated(
        x, W, (1, 1), "SAME", dimension_numbers=_ENC_DIMS
    ) + b

def _res_down(params, x, pfx):
    """Downsampling residual block: (B,H,W,Cin) → (B,H/2,W/2,Cout).
    Main path: stride-2 conv3×3 → GELU → conv3×3
    Skip path: stride-2 conv1×1 (projection)
    """
    h = jax.nn.gelu(_conv_s2(x, params[pfx + "_c1"], params[pfx + "_b1"]))
    h = _conv_s1(h, params[pfx + "_c2"], params[pfx + "_b2"])
    skip = _conv_s2_1x1(x, params[pfx + "_proj"], params[pfx + "_pb"])
    return jax.nn.gelu(h + skip)

def _res_block(params, x, pfx):
    """Regular residual block: same spatial size and channels.
    Main path: conv3×3 → GELU → conv3×3
    Skip path: identity
    """
    h = jax.nn.gelu(_conv_s1(x, params[pfx + "_c1"], params[pfx + "_b1"]))
    h = _conv_s1(h, params[pfx + "_c2"], params[pfx + "_b2"])
    return jax.nn.gelu(h + x)

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    p = {
        # Categorical embeddings
        "cat_emb":   n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        "recon_emb": n((COND_DIM,),                scale=0.1),

        # ── ResNet Encoder ──
        # Stage 1: 3→64, stride-2 down block + 1 regular resblock
        "s1d_c1":   n((3, 3,     3, ENC_C1), scale=(3*3*3)**-0.5),
        "s1d_b1":   jnp.zeros(ENC_C1),
        "s1d_c2":   n((3, 3, ENC_C1, ENC_C1), scale=(3*3*ENC_C1)**-0.5),
        "s1d_b2":   jnp.zeros(ENC_C1),
        "s1d_proj": n((1, 1,     3, ENC_C1), scale=(3)**-0.5),
        "s1d_pb":   jnp.zeros(ENC_C1),
        "s1r_c1":   n((3, 3, ENC_C1, ENC_C1), scale=(3*3*ENC_C1)**-0.5),
        "s1r_b1":   jnp.zeros(ENC_C1),
        "s1r_c2":   n((3, 3, ENC_C1, ENC_C1), scale=(3*3*ENC_C1)**-0.5),
        "s1r_b2":   jnp.zeros(ENC_C1),

        # Stage 2: 64→128, stride-2 down block + 1 regular resblock
        "s2d_c1":   n((3, 3, ENC_C1, ENC_C2), scale=(3*3*ENC_C1)**-0.5),
        "s2d_b1":   jnp.zeros(ENC_C2),
        "s2d_c2":   n((3, 3, ENC_C2, ENC_C2), scale=(3*3*ENC_C2)**-0.5),
        "s2d_b2":   jnp.zeros(ENC_C2),
        "s2d_proj": n((1, 1, ENC_C1, ENC_C2), scale=(ENC_C1)**-0.5),
        "s2d_pb":   jnp.zeros(ENC_C2),
        "s2r_c1":   n((3, 3, ENC_C2, ENC_C2), scale=(3*3*ENC_C2)**-0.5),
        "s2r_b1":   jnp.zeros(ENC_C2),
        "s2r_c2":   n((3, 3, ENC_C2, ENC_C2), scale=(3*3*ENC_C2)**-0.5),
        "s2r_b2":   jnp.zeros(ENC_C2),

        # Stage 3: 128→256, stride-2 down block + 1 regular resblock
        "s3d_c1":   n((3, 3, ENC_C2, ENC_C3), scale=(3*3*ENC_C2)**-0.5),
        "s3d_b1":   jnp.zeros(ENC_C3),
        "s3d_c2":   n((3, 3, ENC_C3, ENC_C3), scale=(3*3*ENC_C3)**-0.5),
        "s3d_b2":   jnp.zeros(ENC_C3),
        "s3d_proj": n((1, 1, ENC_C2, ENC_C3), scale=(ENC_C2)**-0.5),
        "s3d_pb":   jnp.zeros(ENC_C3),
        "s3r_c1":   n((3, 3, ENC_C3, ENC_C3), scale=(3*3*ENC_C3)**-0.5),
        "s3r_b1":   jnp.zeros(ENC_C3),
        "s3r_c2":   n((3, 3, ENC_C3, ENC_C3), scale=(3*3*ENC_C3)**-0.5),
        "s3r_b2":   jnp.zeros(ENC_C3),

        # FC head: 4×4×256 → 512 → LATENT_DIM
        "enc_W1": n((ENC_C3 * 4 * 4, ENC_FC1), scale=(ENC_C3*16)**-0.5),
        "enc_b4": jnp.zeros(ENC_FC1),
        "enc_W2": n((ENC_FC1, LATENT_DIM),     scale=ENC_FC1**-0.5),
        "enc_b5": jnp.zeros(LATENT_DIM),

        # ── Decoder MLP stem: IN_DIM → DC_STEM → DC_C0×2×2 ──
        "dc_W1": n((IN_DIM,  DC_STEM),       scale=IN_DIM**-0.5),
        "dc_b1": jnp.zeros(DC_STEM),
        "dc_W2": n((DC_STEM, DC_C0 * 2 * 2), scale=DC_STEM**-0.5),
        "dc_b2": jnp.zeros(DC_C0 * 2 * 2),
        # ConvTranspose weights: OIHW (C_out, C_in, kH, kW)
        "dc_ct0": n((DC_C0, DC_C1, 4, 4), scale=(DC_C0 * 4 * 4)**-0.5),  # 2×2 → 4×4
        "dc_ct1": n((DC_C1, DC_C2, 4, 4), scale=(DC_C1 * 4 * 4)**-0.5),  # 4×4 → 8×8
        "dc_ct2": n((DC_C2, DC_C3, 4, 4), scale=(DC_C2 * 4 * 4)**-0.5),  # 8×8 → 16×16
        "dc_ct3": n((DC_C3, 3,    4, 4),  scale=(DC_C3 * 4 * 4)**-0.5),  # 16×16 → 32×32
    }
    return p

# ── Conditioning lookup ────────────────────────────────────────────────────────

def cats_to_cond(params, cats):
    embs = params["cat_emb"][jnp.arange(N_CAT)[None, :], cats]
    return embs.reshape(cats.shape[0], COND_DIM)

# ── Encoder (ResNet) ───────────────────────────────────────────────────────────

def encode(params, x_flat):
    """x_flat (B, 3072) in [0,1] → (B, LATENT_DIM)"""
    B = x_flat.shape[0]
    h = x_flat.reshape(B, 32, 32, 3)           # NHWC

    # Stage 1: 32×32×3 → 16×16×64
    h = _res_down(params, h, "s1d")            # (B, 16, 16, 64)
    h = _res_block(params, h, "s1r")           # (B, 16, 16, 64)

    # Stage 2: 16×16×64 → 8×8×128
    h = _res_down(params, h, "s2d")            # (B, 8, 8, 128)
    h = _res_block(params, h, "s2r")           # (B, 8, 8, 128)

    # Stage 3: 8×8×128 → 4×4×256
    h = _res_down(params, h, "s3d")            # (B, 4, 4, 256)
    h = _res_block(params, h, "s3r")           # (B, 4, 4, 256)

    h = h.reshape(B, -1)                        # (B, 4096)
    h = jax.nn.gelu(h @ params["enc_W1"] + params["enc_b4"])  # (B, 512)
    return h @ params["enc_W2"] + params["enc_b5"]             # (B, 128)

# ── Decoder (4-stage deconvnet, starts at 2×2) ────────────────────────────────

def decode(params, zc):
    """zc (B, IN_DIM=144) → recon (B, 3072) in (0, 1)."""
    B = zc.shape[0]
    h = jax.nn.gelu(zc @ params["dc_W1"] + params["dc_b1"])   # (B, DC_STEM)
    h = jax.nn.gelu(h  @ params["dc_W2"] + params["dc_b2"])   # (B, DC_C0*4)
    h = h.reshape(B, DC_C0, 2, 2)                              # (B, DC_C0, 2, 2)
    h = jax.lax.conv_transpose(
        h, params["dc_ct0"],
        strides=(2, 2), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )                                                           # (B, DC_C1, 4, 4)
    h = jax.nn.gelu(h)
    h = jax.lax.conv_transpose(
        h, params["dc_ct1"],
        strides=(2, 2), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )                                                           # (B, DC_C2, 8, 8)
    h = jax.nn.gelu(h)
    h = jax.lax.conv_transpose(
        h, params["dc_ct2"],
        strides=(2, 2), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )                                                           # (B, DC_C3, 16, 16)
    h = jax.nn.gelu(h)
    h = jax.lax.conv_transpose(
        h, params["dc_ct3"],
        strides=(2, 2), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )                                                           # (B, 3, 32, 32)
    return jax.nn.sigmoid(h.reshape(B, IMG_FLAT))

# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(params, x_batch, knn_idx_batch, key, X_train):
    B      = x_batch.shape[0]
    x_norm = x_batch / 255.0
    z      = encode(params, x_norm)

    cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond = jnp.broadcast_to(
        params["recon_emb"] + cats_to_cond(params, cats_recon), (B, COND_DIM))
    x_recon = decode(params, jnp.concatenate([z, recon_cond], axis=-1))
    L_recon = ((x_recon - x_norm) ** 2).mean()

    cats     = jax.random.randint(key, (N_VAR_SAMPLES, N_CAT), 0, VOCAB_SIZE)
    cond_var = params["recon_emb"][None] + cats_to_cond(params, cats)
    z_rep    = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(B * N_VAR_SAMPLES, LATENT_DIM)
    c_rep    = jnp.broadcast_to(cond_var[None], (B, N_VAR_SAMPLES, COND_DIM)).reshape(B * N_VAR_SAMPLES, COND_DIM)
    Y_all    = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_VAR_SAMPLES, IMG_FLAT)

    knn_imgs   = (X_train / 255.0)[knn_idx_batch]
    sq_knn     = (knn_imgs ** 2).sum(-1)
    sq_Y       = (Y_all ** 2).sum(-1)
    dot        = jnp.einsum("bkd,bvd->bkv", knn_imgs, Y_all)
    dists      = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot
    best_slot  = jnp.argmin(dists, axis=-1)
    matched    = Y_all[jnp.arange(B)[:, None], best_slot]
    L_var      = ((matched - jax.lax.stop_gradient(knn_imgs)) ** 2).mean()

    return L_recon + ALPHA * L_var, (L_recon, L_var)

# ── Training ──────────────────────────────────────────────────────────────────

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

# ── Evaluation ────────────────────────────────────────────────────────────────

@jax.jit
def _eval_batch(params, x_batch):
    x_norm     = x_batch / 255.0
    B          = x_batch.shape[0]
    z          = encode(params, x_norm)
    cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond = jnp.broadcast_to(
        params["recon_emb"] + cats_to_cond(params, cats_recon), (B, COND_DIM))
    x_recon    = decode(params, jnp.concatenate([z, recon_cond], axis=-1))
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
    Y_all = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_SLOTS, IMG_FLAT)

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

# ── Training loop ─────────────────────────────────────────────────────────────

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

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging(EXP_NAME)
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().strip().splitlines():
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping")
        exit(0)

    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("cifar10")
    X_train = data.X.reshape(data.n_samples,           IMG_FLAT).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, IMG_FLAT).astype(np.float32)

    KNN_CACHE = Path(__file__).parent / f"knn_cifar_k{K_NN}.npy"
    if KNN_CACHE.exists():
        knn_idx = np.load(KNN_CACHE)
        logging.info(f"kNN loaded from cache: {KNN_CACHE}")
    else:
        knn_idx = compute_knn(X_train, k=K_NN)
        np.save(KNN_CACHE, knn_idx)
        logging.info(f"kNN saved to {KNN_CACHE}")
    logging.info(f"kNN ready: {knn_idx.shape}")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(
        f"Parameters: {n_params(params):,}  "
        f"LATENT={LATENT_DIM}  N_CAT={N_CAT}  VOCAB={VOCAB_SIZE}  E_DIM={E_DIM}"
        f"  COND_DIM={COND_DIM}  IN_DIM={IN_DIM}"
        f"  N_VAR={N_VAR_SAMPLES}  K_NN={K_NN}"
        f"  B*N_VAR={BATCH_SIZE*N_VAR_SAMPLES} decode/step"
        f"  encoder=ResNet(C1={ENC_C1},C2={ENC_C2},C3={ENC_C3},FC={ENC_FC1}→{LATENT_DIM})"
        f"  decoder=4stage-deconvnet(C0={DC_C0},C1={DC_C1},C2={DC_C2},C3={DC_C3},stem={DC_STEM})"
        f"  dataset=cifar10  img_flat={IMG_FLAT}"
    )

    params, history, elapsed = train(params, X_train, X_test, knn_idx)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_recon_mse(params, X_test)
    append_result({
        "experiment":      EXP_NAME,
        "name":            f"ResNet encoder + 4-stage decoder CIFAR-10  LATENT={LATENT_DIM} N_CAT={N_CAT} VOCAB={VOCAB_SIZE} N_VAR={N_VAR_SAMPLES} K_NN={K_NN} B={BATCH_SIZE}",
        "n_params":        n_params(params),
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "final_recon_mse": round(float(final_mse), 6),
        "loss_curve":      [round(v, 6) for v in history["loss"]],
        "L_recon_curve":   [round(v, 6) for v in history["L_recon"]],
        "L_var_curve":     [round(v, 6) for v in history["L_var"]],
        "eval_mse_curve":              history["eval_mse"],
        "eval_recall_mse_curve":       history["eval_recall_mse"],
        "eval_recall_by_rank_curve":   history["eval_recall_by_rank"],
        "latent_dim":      LATENT_DIM,
        "n_cat":           N_CAT,
        "vocab_size":      VOCAB_SIZE,
        "e_dim":           E_DIM,
        "cond_dim":        COND_DIM,
        "n_var_samples":   N_VAR_SAMPLES,
        "k_nn":            K_NN,
        "enc_c1":          ENC_C1,
        "enc_c2":          ENC_C2,
        "enc_c3":          ENC_C3,
        "enc_fc1":         ENC_FC1,
        "dc_c0":           DC_C0,
        "dc_c1":           DC_C1,
        "dc_c2":           DC_C2,
        "dc_c3":           DC_C3,
        "dc_stem":         DC_STEM,
        "alpha":           ALPHA,
        "adam_lr":         ADAM_LR,
        "batch_size":      BATCH_SIZE,
        "encoder":         "resnet",
        "decoder":         "4stage-deconvnet",
        "dataset":         "cifar10",
        "img_flat":        IMG_FLAT,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s")
