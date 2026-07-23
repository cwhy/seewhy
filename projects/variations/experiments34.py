"""
exp34 — N_VAR=256 + B=32, LATENT=2048, 200 epochs.

exp33 (LATENT=2048, N_VAR=128, B=64, 200ep): recon_mse=0.000983, recall_mse=0.007269.

Goal: match training coverage to eval coverage.
Eval uses 256 fixed conditions over K_NN=16 neighbors → 16× coverage ratio.
exp33 trained with N_VAR=128 → only 8× coverage.
Doubling N_VAR to 256 (historically −7% recall per doubling) should close this gap.

Changes from exp33:
- BATCH_SIZE:    64 → 32    (keep B×N_VAR=8,192 for memory safety)
- N_VAR_SAMPLES: 128 → 256  (match eval coverage ratio 256/K_NN=16×)
- num_batches = 50000//32 = 1562; n_full_chunks = 1562//390 = 4 dispatches/epoch

Memory safety: B×N_VAR = 32×256 = 8,192 per-step decode passes (identical to exp33 64×128).
Epoch time: ~280s (2× more batches, same per-step cost), total ~15.5h.
Expected recall_mse: ~0.0067–0.0071 (below exp33 0.00727).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/variations/experiments34.py
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

EXP_NAME = "exp34"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM     = 2048
N_CAT          = 4
VOCAB_SIZE     = 8
E_DIM          = 4
COND_DIM       = N_CAT * E_DIM   # 16
IN_DIM         = LATENT_DIM + COND_DIM  # 2064
ALPHA          = 1.0
BATCH_SIZE     = 32    # was 64 — keep B×N_VAR=8,192 to avoid scan OOM
N_VAR_SAMPLES  = 256   # was 128 — match eval coverage ratio (256/K_NN=16×)
MAX_EPOCHS     = 200
ADAM_LR        = 3e-4
SEED           = 42
K_NN           = 16
CHUNK_SIZE     = 390   # chunked scan: 1562//390=4 Python dispatches/epoch

# CNN encoder channels
ENC_C1         = 64
ENC_C2         = 128
ENC_C3         = 256
ENC_FC1        = 2048  # match LATENT_DIM

# Wide decoder channels
DC_STEM        = 1024
DC_C1          = 256
DC_C2          = 128
DC_C3          = 64

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

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    return {
        "cat_emb":   n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        "recon_emb": n((COND_DIM,),                scale=0.1),
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

def make_chunk_fn(optimizer):
    # Chunked lax.scan: CHUNK_SIZE=390 steps per dispatch.
    # 1562 batches/epoch → 4 Python dispatches (drops last 2 batches).
    # Same per-step memory as exp33 scan (B×N_VAR=8,192 decode passes).
    @jax.jit
    def chunk_fn(params, opt_state, Xs_chunk, KNN_chunk, chunk_key, X_train):
        # Xs_chunk:  (CHUNK_SIZE, BATCH_SIZE, IMG_FLAT)
        # KNN_chunk: (CHUNK_SIZE, BATCH_SIZE, K_NN)
        def scan_body(carry, inputs):
            params, opt_state = carry
            x_batch, knn_idx_batch, key = inputs
            (loss, (L_recon, L_var)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, x_batch, knn_idx_batch, key, X_train
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), (loss, L_recon, L_var)

        keys = jax.vmap(lambda b: jax.random.fold_in(chunk_key, b))(jnp.arange(CHUNK_SIZE))
        (params, opt_state), (losses, l_recon, l_var) = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_chunk, KNN_chunk, keys)
        )
        return params, opt_state, losses.mean(), l_recon.mean(), l_var.mean()

    return chunk_fn

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

def train(params, X_tr, X_te, knn_idx):
    num_batches  = len(X_tr) // BATCH_SIZE   # 1562
    num_steps    = MAX_EPOCHS * num_batches
    optimizer    = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state    = optimizer.init(params)
    chunk_fn     = make_chunk_fn(optimizer)
    key_gen      = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    X_train_jax  = jnp.array(X_tr)
    n_full_chunks = num_batches // CHUNK_SIZE  # 1562 // 390 = 4

    history = {
        "loss": [], "L_recon": [], "L_var": [],
        "eval_mse": [], "eval_recall_mse": [], "eval_recall_by_rank": [],
    }
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        n    = n_full_chunks * CHUNK_SIZE * BATCH_SIZE  # 4*390*32 = 49920
        Xs_batched  = jnp.array(X_tr[perm[:n]].reshape(n_full_chunks, CHUNK_SIZE, BATCH_SIZE, IMG_FLAT))
        KNN_batched = jnp.array(knn_idx[perm[:n]].reshape(n_full_chunks, CHUNK_SIZE, BATCH_SIZE, K_NN))

        all_losses, all_lrecon, all_lvar = [], [], []
        for c in range(n_full_chunks):  # 4 Python dispatches
            chunk_key = jax.random.fold_in(key, c)
            params, opt_state, loss_c, lr_c, lv_c = chunk_fn(
                params, opt_state, Xs_batched[c], KNN_batched[c], chunk_key, X_train_jax
            )
            all_losses.append(float(loss_c))
            all_lrecon.append(float(lr_c))
            all_lvar.append(float(lv_c))

        loss    = float(np.mean(all_losses))
        l_recon = float(np.mean(all_lrecon))
        l_var   = float(np.mean(all_lvar))
        history["loss"].append(loss)
        history["L_recon"].append(l_recon)
        history["L_var"].append(l_var)

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            mse                        = eval_recon_mse(params, X_te)
            recall_mse, recall_by_rank = eval_recall_mse(params, X_tr, knn_idx)
            history["eval_mse"].append((epoch, mse))
            history["eval_recall_mse"].append((epoch, recall_mse))
            history["eval_recall_by_rank"].append((epoch, recall_by_rank))
            eval_str = f"  eval_mse={mse:.5f}  recall_mse={recall_mse:.5f}"

        logging.info(
            f"epoch {epoch:3d}  loss={loss:.5f}"
            f"  L_recon={l_recon:.5f}  L_var={l_var:.5f}"
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
        f"LATENT={LATENT_DIM}  ENC_FC1={ENC_FC1}  N_VAR={N_VAR_SAMPLES}  K_NN={K_NN}  B={BATCH_SIZE}"
        f"  coverage_ratio={N_VAR_SAMPLES/K_NN:.1f}x"
        f"  CHUNK_SIZE={CHUNK_SIZE}  n_full_chunks={len(range(0, X_train.shape[0]//BATCH_SIZE))//CHUNK_SIZE}"
        f"  encoder=CNN  decoder=wide-deconvnet"
        f"  dataset=cifar10  MAX_EPOCHS={MAX_EPOCHS}"
    )

    params, history, elapsed = train(params, X_train, X_test, knn_idx)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_recon_mse(params, X_test)
    append_result({
        "experiment": EXP_NAME,
        "name": f"LATENT=2048 wide-deconvnet N_VAR=256 K_NN=16 CIFAR-10 200ep chunked-scan B=32",
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
        "enc_fc1": ENC_FC1, "dc_c1": DC_C1, "dc_c2": DC_C2, "dc_c3": DC_C3,
        "dc_stem": DC_STEM, "alpha": ALPHA, "adam_lr": ADAM_LR,
        "batch_size": BATCH_SIZE, "chunk_size": CHUNK_SIZE, "encoder": "cnn",
        "decoder": "wide-deconvnet", "dataset": "cifar10", "img_flat": IMG_FLAT,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s")
