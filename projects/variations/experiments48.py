"""
exp48 — exp45 + training K_NN 16→32, MAX_EPOCHS=300.

Hypothesis: training with 2× more neighbours per image gives richer signal per step
at near-zero extra cost (Phase 2 goes from 1024→2048 items; Phase 1 unchanged).
Eval metric stays at K=16 for apples-to-apples comparison with exp35/exp45.

Changes from exp45:
- K_NN = 32 (training)  vs 16 in exp45
- EVAL_K_NN = 16 (eval stays comparable)
- MAX_EPOCHS = 300
- No early stopping (run full schedule)

Expected epoch time: ~47s (+10% from 2× Phase 2 items).
Expected: if K_NN scaling helps, ep5 recall should beat exp45's 0.00929.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/variations/experiments48.py
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

EXP_NAME = "exp48"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM    = 2048
N_CAT         = 4
VOCAB_SIZE    = 8
E_DIM         = 16
COND_DIM      = N_CAT * E_DIM   # 64
IN_DIM        = LATENT_DIM + COND_DIM  # 2112
BATCH_SIZE    = 64
N_VAR_SAMPLES = 128
MAX_EPOCHS    = 300
ADAM_LR       = 3e-4
ALPHA         = 1.0
SEED          = 42

K_NN          = 32    # training neighbours (2× exp45)
EVAL_K_NN     = 16    # eval neighbours — fixed for comparison with exp35/exp45

# CNN encoder channels (same as exp35)
ENC_C1        = 64
ENC_C2        = 128
ENC_C3        = 256
ENC_FC1       = 2048

# Wide decoder channels (same as exp35)
DC_STEM       = 1024
DC_C1         = 256
DC_C2         = 128
DC_C3         = 64

IMG_FLAT      = 3072
EVAL_EVERY    = 5
EVAL_RECALL_B = 8

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

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    return {
        "cat_emb": n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        "enc_c1": n((3, 3,     3, ENC_C1), scale=(3*3*3    )**-0.5),
        "enc_b1": jnp.zeros(ENC_C1),
        "enc_c2": n((3, 3, ENC_C1, ENC_C2), scale=(3*3*ENC_C1)**-0.5),
        "enc_b2": jnp.zeros(ENC_C2),
        "enc_c3": n((3, 3, ENC_C2, ENC_C3), scale=(3*3*ENC_C2)**-0.5),
        "enc_b3": jnp.zeros(ENC_C3),
        "enc_W1": n((ENC_C3 * 4 * 4, ENC_FC1), scale=(ENC_C3*16)**-0.5),
        "enc_b4": jnp.zeros(ENC_FC1),
        "enc_W2": n((ENC_FC1, LATENT_DIM),     scale=ENC_FC1**-0.5),
        "enc_b5": jnp.zeros(LATENT_DIM),
        "dc_W1": n((IN_DIM,  DC_STEM),       scale=IN_DIM**-0.5),
        "dc_b1": jnp.zeros(DC_STEM),
        "dc_W2": n((DC_STEM, DC_C1 * 4*4),   scale=DC_STEM**-0.5),
        "dc_b2": jnp.zeros(DC_C1 * 4*4),
        "dc_ct1": n((DC_C1, DC_C2, 4, 4), scale=(DC_C1 * 4 * 4)**-0.5),
        "dc_ct2": n((DC_C2, DC_C3, 4, 4), scale=(DC_C2 * 4 * 4)**-0.5),
        "dc_ct3": n((DC_C3, 3, 4, 4),     scale=(DC_C3 * 4 * 4)**-0.5),
    }

# ── Model ─────────────────────────────────────────────────────────────────────

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

def decode(params, zc):
    B = zc.shape[0]
    h = jax.nn.gelu(zc @ params["dc_W1"] + params["dc_b1"])
    h = jax.nn.gelu(h  @ params["dc_W2"] + params["dc_b2"])
    h = h.reshape(B, DC_C1, 4, 4)
    h = jax.lax.conv_transpose(h, params["dc_ct1"], strides=(2,2), padding="SAME",
                                dimension_numbers=("NCHW","IOHW","NCHW"))
    h = jax.nn.gelu(h)
    h = jax.lax.conv_transpose(h, params["dc_ct2"], strides=(2,2), padding="SAME",
                                dimension_numbers=("NCHW","IOHW","NCHW"))
    h = jax.nn.gelu(h)
    h = jax.lax.conv_transpose(h, params["dc_ct3"], strides=(2,2), padding="SAME",
                                dimension_numbers=("NCHW","IOHW","NCHW"))
    return jax.nn.sigmoid(h.reshape(B, IMG_FLAT))

# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(params, x_batch, cats, X_train_norm):
    B    = x_batch.shape[0]
    x_nm = x_batch / 255.0
    z    = encode(params, x_nm)

    # K_NN=32 nearest neighbours via GEMM, skip self (rank 0)
    sq_x    = (x_nm ** 2).sum(-1, keepdims=True)
    sq_tr   = (X_train_norm ** 2).sum(-1)
    dot_xt  = x_nm @ X_train_norm.T
    dists_x = sq_x + sq_tr - 2.0 * dot_xt
    _, knn_idx = jax.lax.top_k(-dists_x, k=K_NN + 1)
    knn = jax.lax.stop_gradient(X_train_norm[knn_idx[:, 1:]])                  # (B, K_NN=32, D)

    # Phase 1 — chunked argmin over N_VAR (stop_gradient, no activation storage)
    CHUNK    = N_VAR_SAMPLES // 2
    cond_all = cats_to_cond(params, cats)
    sq_knn   = (knn ** 2).sum(-1)
    best_dist = jnp.full((B, K_NN), jnp.inf)
    best_var  = jnp.zeros((B, K_NN), dtype=jnp.int32)
    for c0 in range(0, N_VAR_SAMPLES, CHUNK):
        cond_c = cond_all[c0 : c0 + CHUNK]
        z_c    = jnp.repeat(z[:, None, :], CHUNK, axis=1).reshape(B * CHUNK, LATENT_DIM)
        cc_c   = jnp.broadcast_to(cond_c[None], (B, CHUNK, COND_DIM)).reshape(B * CHUNK, COND_DIM)
        Y_c    = jax.lax.stop_gradient(
            decode(params, jnp.concatenate([z_c, cc_c], axis=-1))
        ).reshape(B, CHUNK, IMG_FLAT)
        sq_Y_c  = (Y_c ** 2).sum(-1)
        dists_c = sq_knn[:, :, None] + sq_Y_c[:, None, :] - 2.0 * jnp.einsum("bkd,bvd->bkv", knn, Y_c)
        min_d_c = dists_c.min(axis=-1)
        min_i_c = jnp.argmin(dists_c, axis=-1) + c0
        upd      = min_d_c < best_dist
        best_dist = jnp.where(upd, min_d_c, best_dist)
        best_var  = jnp.where(upd, min_i_c, best_var)
    best_var = jax.lax.stop_gradient(best_var)

    # Phase 2 — re-decode K_NN=32 winners WITH gradient (B×32=2048 items)
    win_cats = cats[best_var]
    win_cond = cats_to_cond(params, win_cats.reshape(B * K_NN, N_CAT))
    win_z    = jnp.repeat(z, K_NN, axis=0)
    matched  = decode(params, jnp.concatenate([win_z, win_cond], axis=-1))
    L_imle   = ((matched - knn.reshape(B * K_NN, IMG_FLAT)) ** 2).mean()

    # Reconstruction loss
    cats_zero  = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond = jnp.broadcast_to(cats_to_cond(params, cats_zero), (B, COND_DIM))
    x_recon    = decode(params, jnp.concatenate([z, recon_cond], axis=-1))
    L_recon    = ((x_recon - x_nm) ** 2).mean()

    return L_imle + ALPHA * L_recon, (L_recon, L_imle)

def make_train_step(optimizer):
    @jax.jit
    def _train_step(params, opt_state, x_batch, cats, X_train_norm):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_batch, cats, X_train_norm)
        updates, new_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss, aux
    return _train_step

# ── Evaluation ────────────────────────────────────────────────────────────────

@jax.jit
def _eval_recon_batch(params, x_batch):
    x_nm      = x_batch / 255.0
    B         = x_batch.shape[0]
    z         = encode(params, x_nm)
    cats_zero = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    cond      = jnp.broadcast_to(cats_to_cond(params, cats_zero), (B, COND_DIM))
    x_recon   = decode(params, jnp.concatenate([z, cond], axis=-1))
    return ((x_recon - x_nm) ** 2).mean()

def eval_recon_mse(params, X_te):
    mses = [float(_eval_recon_batch(params, jnp.array(X_te[i:i+256])))
            for i in range(0, len(X_te), 256)]
    return float(np.mean(mses))

@jax.jit
def _recall_batch_jit(params, x_norm, X_train_norm, all_conds):
    """Eval at EVAL_K_NN=16 for comparison with exp35/exp45."""
    B       = x_norm.shape[0]
    N_SLOTS = all_conds.shape[0]

    sq_x    = (x_norm ** 2).sum(-1, keepdims=True)
    sq_tr   = (X_train_norm ** 2).sum(-1)
    dot_xt  = x_norm @ X_train_norm.T
    dists_x = sq_x + sq_tr - 2.0 * dot_xt
    knn_idx  = jnp.argsort(dists_x, axis=1)[:, :EVAL_K_NN]                    # K=16 for eval
    knn_norm = X_train_norm[knn_idx]

    z     = encode(params, x_norm)
    z_rep = jnp.repeat(z[:, None, :], N_SLOTS, axis=1).reshape(B * N_SLOTS, LATENT_DIM)
    c_rep = jnp.broadcast_to(all_conds[None], (B, N_SLOTS, COND_DIM)).reshape(B * N_SLOTS, COND_DIM)
    Y_all = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_SLOTS, IMG_FLAT)

    sq_knn   = (knn_norm ** 2).sum(-1)
    sq_Y     = (Y_all ** 2).sum(-1)
    dot_kv   = jnp.einsum("bkd,bvd->bkv", knn_norm, Y_all)
    dists_kv = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot_kv
    best     = jnp.argmin(dists_kv, axis=-1)
    matched  = Y_all[jnp.arange(B)[:, None], best]

    sq_xi      = (x_norm ** 2).sum(-1)
    dot_xk     = jnp.einsum("bd,bkd->bk", x_norm, knn_norm)
    dist_x_knn = sq_xi[:, None] + sq_knn - 2.0 * dot_xk
    rank_order = jnp.argsort(dist_x_knn, axis=-1)

    sorted_matched = matched[jnp.arange(B)[:, None], rank_order]
    sorted_knn     = knn_norm[jnp.arange(B)[:, None], rank_order]

    per_rank = ((sorted_matched - sorted_knn) ** 2).mean(axis=-1).mean(axis=0)
    return per_rank.mean(), per_rank

def eval_recall_mse(params, X_train_jax, n_images=512):
    all_conds    = cats_to_cond(params, EVAL_COND_CATS)
    X_train_norm = X_train_jax / 255.0
    n            = (n_images // EVAL_RECALL_B) * EVAL_RECALL_B
    means, per_ranks = [], []
    for start in range(0, n, EVAL_RECALL_B):
        x_norm = X_train_norm[start:start + EVAL_RECALL_B]
        mean_r, rank_r = _recall_batch_jit(params, x_norm, X_train_norm, all_conds)
        means.append(float(mean_r))
        per_ranks.append(np.array(rank_r))
    return float(np.mean(means)), np.mean(per_ranks, axis=0).tolist()

# ── Training ──────────────────────────────────────────────────────────────────

def train(params, X_train_jax, X_te, X_tr_np):
    num_batches  = X_tr_np.shape[0] // BATCH_SIZE
    num_steps    = MAX_EPOCHS * num_batches
    optimizer    = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state    = optimizer.init(params)
    key_gen      = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    _train_step  = make_train_step(optimizer)
    X_train_norm = X_train_jax / 255.0

    history = {
        "loss": [], "L_recon": [], "L_imle": [],
        "eval_recon_mse": [], "eval_recall_mse": [], "eval_recall_by_rank": [],
    }
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, X_tr_np.shape[0]))

        epoch_losses, epoch_lrecon, epoch_limle = [], [], []
        for i in range(0, num_batches * BATCH_SIZE, BATCH_SIZE):
            x_batch  = X_train_jax[perm[i:i+BATCH_SIZE]]
            step_idx = i // BATCH_SIZE
            ck   = jax.random.fold_in(key, step_idx)
            cats = jax.random.randint(ck, (N_VAR_SAMPLES, N_CAT), 0, VOCAB_SIZE)
            params, opt_state, loss, (l_recon, l_imle) = _train_step(params, opt_state, x_batch, cats, X_train_norm)
            epoch_losses.append(float(loss))
            epoch_lrecon.append(float(l_recon))
            epoch_limle.append(float(l_imle))

        avg_loss   = float(np.mean(epoch_losses))
        avg_lrecon = float(np.mean(epoch_lrecon))
        avg_limle  = float(np.mean(epoch_limle))
        history["loss"].append(avg_loss)
        history["L_recon"].append(avg_lrecon)
        history["L_imle"].append(avg_limle)

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            recon_mse          = eval_recon_mse(params, X_te)
            recall_mse, rank_r = eval_recall_mse(params, X_train_jax)
            history["eval_recon_mse"].append((epoch, recon_mse))
            history["eval_recall_mse"].append((epoch, recall_mse))
            history["eval_recall_by_rank"].append((epoch, rank_r))
            eval_str = f"  eval_recon={recon_mse:.5f}  recall_mse={recall_mse:.5f}"

        logging.info(
            f"epoch {epoch:3d}  loss={avg_loss:.5f}  L_recon={avg_lrecon:.5f}  L_imle={avg_limle:.5f}"
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
        logging.info(f"{EXP_NAME} already done — skipping"); exit(0)

    logging.info(f"JAX devices: {jax.devices()}")
    data        = load_supervised_image("cifar10")
    X_train_np  = np.array(data.X).reshape(data.n_samples,           IMG_FLAT).astype(np.float32)
    X_test_np   = np.array(data.X_test).reshape(data.n_test_samples, IMG_FLAT).astype(np.float32)

    logging.info(f"Loading X_train to GPU: {X_train_np.shape} ({X_train_np.nbytes/1e6:.0f}MB)…")
    X_train_jax = jnp.array(X_train_np)
    logging.info(f"X_train on GPU. Peak VRAM during kNN GEMM: ~{BATCH_SIZE*data.n_samples*4/1e6:.0f}MB")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(
        f"Parameters: {n_params(params):,}  "
        f"LATENT={LATENT_DIM}  N_CAT={N_CAT}  VOCAB={VOCAB_SIZE}  E_DIM={E_DIM}"
        f"  COND_DIM={COND_DIM}  IN_DIM={IN_DIM}"
        f"  N_VAR={N_VAR_SAMPLES}  B={BATCH_SIZE}  train_K={K_NN}  eval_K={EVAL_K_NN}"
        f"  encoder=CNN  decoder=wide-deconvnet  dataset=cifar10  MAX_EPOCHS={MAX_EPOCHS}"
    )

    params, history, elapsed = train(params, X_train_jax, X_test_np, X_train_np)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_recon_mse              = eval_recon_mse(params, X_test_np)
    final_recall_mse, final_ranks = eval_recall_mse(params, X_train_jax)
    append_result({
        "experiment":       EXP_NAME,
        "name":             f"Local IMLE on-the-fly kNN CIFAR-10 LATENT=2048 N_VAR=128 train_K={K_NN} eval_K={EVAL_K_NN} B=64 300ep",
        "n_params":         n_params(params),
        "epochs":           MAX_EPOCHS,
        "time_s":           round(elapsed, 1),
        "final_recon_mse":  round(float(final_recon_mse),  6),
        "final_recall_mse": round(float(final_recall_mse), 6),
        "final_recall_by_rank": [round(v, 6) for v in final_ranks],
        "loss_curve":              [round(v, 6) for v in history["loss"]],
        "L_recon_curve":           [round(v, 6) for v in history["L_recon"]],
        "L_imle_curve":            [round(v, 6) for v in history["L_imle"]],
        "eval_recon_mse_curve":    history["eval_recon_mse"],
        "eval_recall_mse_curve":   history["eval_recall_mse"],
        "eval_recall_by_rank_curve": history["eval_recall_by_rank"],
        "latent_dim":    LATENT_DIM,
        "n_cat":         N_CAT,
        "vocab_size":    VOCAB_SIZE,
        "e_dim":         E_DIM,
        "cond_dim":      COND_DIM,
        "n_var_samples": N_VAR_SAMPLES,
        "k_nn":          K_NN,
        "eval_k_nn":     EVAL_K_NN,
        "enc_c1": ENC_C1, "enc_c2": ENC_C2, "enc_c3": ENC_C3, "enc_fc1": ENC_FC1,
        "dc_c1":  DC_C1,  "dc_c2":  DC_C2,  "dc_c3":  DC_C3,  "dc_stem": DC_STEM,
        "adam_lr":    ADAM_LR,
        "alpha":      ALPHA,
        "batch_size": BATCH_SIZE,
        "encoder":    "cnn",
        "decoder":    "wide-deconvnet",
        "imle":       "local-onthefly-kNN-inverse+recon",
        "dataset":    "cifar10",
        "img_flat":   IMG_FLAT,
    })
    logging.info(f"Done. recon_mse={final_recon_mse:.5f}  recall_mse={final_recall_mse:.5f}  {elapsed:.1f}s")
