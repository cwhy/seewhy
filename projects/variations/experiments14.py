"""
exp14 — deconvnet decoder, B=128, N_VAR=16.

Replaces the Gram-matrix decoder with a transposed-convolution decoder:
  (B, IN_DIM) → MLP stem → reshape (B, 64, 7, 7)
              → ConvTranspose(64→32, 4×4, stride=2) → GELU → (B, 32, 14, 14)
              → ConvTranspose(32→1,  4×4, stride=2)         → (B, 1,  28, 28)
              → sigmoid → (B, 784)

ConvTranspose backward is a regular conv, which cuDNN handles much more
efficiently than the Gram decoder's large intermediate tensors.

Benchmarked speedup vs exp13 (Gram, B=128, N=16): ~21 min vs ~31 min.

Usage:
    uv run python projects/variations/experiments14.py
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

EXP_NAME = "exp14"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM     = 64
N_CAT          = 4
VOCAB_SIZE     = 8
E_DIM          = 4
COND_DIM       = N_CAT * E_DIM   # 16
IN_DIM         = LATENT_DIM + COND_DIM  # 80
ALPHA          = 1.0
BATCH_SIZE     = 128
N_VAR_SAMPLES  = 16
MAX_EPOCHS     = 100
ADAM_LR        = 3e-4
SEED           = 42
K_NN           = 64

# Decoder architecture
DC_STEM        = 256   # MLP hidden before reshape
DC_C1          = 64    # channels after reshape  (7×7 spatial)
DC_C2          = 32    # channels after first ConvTranspose (14×14 spatial)
# → ConvTranspose → (B, 1, 28, 28)

EVAL_EVERY  = 5

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

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    return {
        # Categorical embeddings
        "cat_emb":   n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        "recon_emb": n((COND_DIM,),                scale=0.1),
        # Encoder: 784 → 512 → LATENT_DIM
        "enc_W1": n((784, 512),        scale=784**-0.5),
        "enc_b1": jnp.zeros(512),
        "enc_W2": n((512, LATENT_DIM), scale=512**-0.5),
        "enc_b2": jnp.zeros(LATENT_DIM),
        # Decoder MLP stem: IN_DIM → DC_STEM → DC_C1*7*7
        "dc_W1": n((IN_DIM,  DC_STEM),      scale=IN_DIM**-0.5),
        "dc_b1": jnp.zeros(DC_STEM),
        "dc_W2": n((DC_STEM, DC_C1 * 7*7),  scale=DC_STEM**-0.5),
        "dc_b2": jnp.zeros(DC_C1 * 7*7),
        # ConvTranspose weights: (C_in, C_out, kH, kW)
        # ct1: (DC_C1, 7, 7) → (DC_C2, 14, 14)  kernel 4×4
        "dc_ct1": n((DC_C1, DC_C2, 4, 4), scale=(DC_C1 * 4 * 4)**-0.5),
        # ct2: (DC_C2, 14, 14) → (1, 28, 28)    kernel 4×4
        "dc_ct2": n((DC_C2,    1, 4, 4), scale=(DC_C2 * 4 * 4)**-0.5),
    }

# ── Conditioning lookup ────────────────────────────────────────────────────────

def cats_to_cond(params, cats):
    embs = params["cat_emb"][jnp.arange(N_CAT)[None, :], cats]
    return embs.reshape(cats.shape[0], COND_DIM)

# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    return h @ params["enc_W2"] + params["enc_b2"]

# ── Decoder (deconvnet) ────────────────────────────────────────────────────────

def decode(params, zc):
    """zc (B, IN_DIM) → recon (B, 784) in (0, 1).

    MLP stem projects to (B, DC_C1, 7, 7), then two stride-2 ConvTranspose
    layers upsample: 7×7 → 14×14 → 28×28.
    ConvTranspose backward is a regular conv — cuDNN handles it efficiently.
    """
    B = zc.shape[0]
    # MLP stem
    h = jax.nn.gelu(zc @ params["dc_W1"] + params["dc_b1"])   # (B, DC_STEM)
    h = jax.nn.gelu(h  @ params["dc_W2"] + params["dc_b2"])   # (B, DC_C1*49)
    h = h.reshape(B, DC_C1, 7, 7)                              # (B, DC_C1, 7, 7)
    # ConvTranspose 7 → 14
    h = jax.lax.conv_transpose(
        h, params["dc_ct1"],
        strides=(2, 2), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )                                                           # (B, DC_C2, 14, 14)
    h = jax.nn.gelu(h)
    # ConvTranspose 14 → 28
    h = jax.lax.conv_transpose(
        h, params["dc_ct2"],
        strides=(2, 2), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )                                                           # (B, 1, 28, 28)
    return jax.nn.sigmoid(h.reshape(B, 784))

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
    Y_all    = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_VAR_SAMPLES, 784)

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
    Y_all = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_SLOTS, 784)

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
        Xs_batched  = jnp.array(X_tr[perm[:n]].reshape(num_batches, BATCH_SIZE, 784))
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
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)

    KNN_CACHE = Path(__file__).parent / f"knn_mnist_k{K_NN}.npy"
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
        f"  decoder=deconvnet(C1={DC_C1},C2={DC_C2},stem={DC_STEM})"
    )

    params, history, elapsed = train(params, X_train, X_test, knn_idx)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_recon_mse(params, X_test)
    append_result({
        "experiment":      EXP_NAME,
        "name":            f"deconvnet decoder  LATENT={LATENT_DIM} N_CAT={N_CAT} VOCAB={VOCAB_SIZE} N_VAR={N_VAR_SAMPLES} K_NN={K_NN} B={BATCH_SIZE} C1={DC_C1} C2={DC_C2} stem={DC_STEM}",
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
        "dc_c1":           DC_C1,
        "dc_c2":           DC_C2,
        "dc_stem":         DC_STEM,
        "alpha":           ALPHA,
        "adam_lr":         ADAM_LR,
        "batch_size":      BATCH_SIZE,
        "decoder":         "deconvnet",
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s")
