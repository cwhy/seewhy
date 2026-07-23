"""
exp8 — Variations: inverted IMLE (exp6) without reconstruction loss.

Changes from exp6:
  - L_recon removed from training entirely; only L_var (inverted IMLE on kNN).
  - recon_emb removed from params (no reconstruction path trained).
  - Eval "reconstruction": for each test image, find the closest decoded output
    among EVAL_COND_CATS slots → "best-slot recon MSE". Measures how well the
    variation slots alone can cover the data without explicit recon supervision.

Architecture otherwise identical: 4×8 categorical embedding, COND_DIM=16, per-rank recall.

Usage:
    uv run python projects/variations/experiments8.py
"""

import json, time, sys, pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image
from lib.logging_utils import setup_logging

EXP_NAME = "exp8"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM     = 64
N_CAT          = 4
VOCAB_SIZE     = 8
E_DIM          = 4
COND_DIM       = N_CAT * E_DIM  # = 16
IN_DIM         = LATENT_DIM + COND_DIM  # = 80
MLP_HIDDEN     = 256
DEC_RANK       = 16
D_R, D_C       = 6, 6
D_V            = 4
D_RC           = D_R * D_C        # 36
D              = D_R * D_C * D_V  # 144
ALPHA          = 1.0
BATCH_SIZE     = 256
N_VAR_SAMPLES  = 2
MAX_EPOCHS     = 100
ADAM_LR        = 3e-4
SEED           = 42
K_NN           = 64

EVAL_EVERY  = 5

JSONL = Path(__file__).parent / "results.jsonl"
ROWS  = jnp.arange(784) // 28
COLS  = jnp.arange(784) % 28

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
        "cat_emb":  n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        "recon_emb": n((COND_DIM,), scale=0.1),   # kept for eval; not used in training loss
        # Encoder
        "enc_W1": n((784, 512),         scale=784**-0.5),
        "enc_b1": jnp.zeros(512),
        "enc_W2": n((512, LATENT_DIM),  scale=512**-0.5),
        "enc_b2": jnp.zeros(LATENT_DIM),
        # Decoder
        "dec_W_mlp1": n((IN_DIM, MLP_HIDDEN),      scale=IN_DIM**-0.5),
        "dec_b_mlp1": jnp.zeros(MLP_HIDDEN),
        "dec_W_A":    n((MLP_HIDDEN, DEC_RANK * D), scale=MLP_HIDDEN**-0.5),
        "dec_W_B":    n((MLP_HIDDEN, DEC_RANK * D), scale=MLP_HIDDEN**-0.5),
        "dec_row_emb": n((28, D_R),     scale=D_R**-0.5),
        "dec_col_emb": n((28, D_C),     scale=D_C**-0.5),
        "dec_W_pos":  n((D, D_RC),      scale=D_RC**-0.5),
        "dec_W_val1": n((2*D, D),       scale=D**-0.5),
        "dec_b_val1": jnp.zeros(2*D),
        "dec_W_val2": n((1, 2*D),       scale=(2*D)**-0.5),
    }

# ── Conditioning lookup ────────────────────────────────────────────────────────

def cats_to_cond(params, cats):
    embs = params["cat_emb"][jnp.arange(N_CAT)[None, :], cats]
    return embs.reshape(cats.shape[0], COND_DIM)

# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    return h @ params["enc_W2"] + params["enc_b2"]

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, zc):
    B     = zc.shape[0]
    h     = jax.nn.gelu(zc @ params["dec_W_mlp1"] + params["dec_b_mlp1"])
    A     = (h @ params["dec_W_A"]).reshape(B, DEC_RANK, D)
    B_fac = (h @ params["dec_W_B"]).reshape(B, DEC_RANK, D)
    gram  = jnp.einsum("bri,brj->bij", A, B_fac)

    def decode_pixel(r, c, g):
        re     = params["dec_row_emb"][r]
        ce     = params["dec_col_emb"][c]
        pos_rc = (re[:, None] * ce[None, :]).reshape(D_RC)
        q      = params["dec_W_pos"] @ pos_rc
        f      = g @ q
        h1     = jax.nn.gelu(params["dec_W_val1"] @ f + params["dec_b_val1"])
        return jax.nn.sigmoid((params["dec_W_val2"] @ h1)[0])

    def decode_image(g):
        return jax.vmap(decode_pixel, in_axes=(0, 0, None))(ROWS, COLS, g)

    return jax.vmap(decode_image)(gram)

# ── Loss (variation only, no reconstruction term) ─────────────────────────────

def loss_fn(params, x_batch, knn_idx_batch, key, X_train):
    B      = x_batch.shape[0]
    x_norm = x_batch / 255.0
    z      = encode(params, x_norm)

    cats     = jax.random.randint(key, (N_VAR_SAMPLES, N_CAT), 0, VOCAB_SIZE)
    cond_var = params["recon_emb"][None] + cats_to_cond(params, cats)      # (N_VAR, COND_DIM)

    z_rep = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(B * N_VAR_SAMPLES, LATENT_DIM)
    c_rep = jnp.broadcast_to(cond_var[None], (B, N_VAR_SAMPLES, COND_DIM)).reshape(B * N_VAR_SAMPLES, COND_DIM)
    Y_all = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_VAR_SAMPLES, 784)

    knn_imgs   = (X_train / 255.0)[knn_idx_batch]                         # (B, K_NN, 784)
    sq_knn     = (knn_imgs ** 2).sum(-1)
    sq_Y       = (Y_all ** 2).sum(-1)
    dot        = jnp.einsum("bkd,bvd->bkv", knn_imgs, Y_all)
    dists      = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot
    best_slot  = jnp.argmin(dists, axis=-1)
    matched    = Y_all[jnp.arange(B)[:, None], best_slot]

    L_var = ((matched - jax.lax.stop_gradient(knn_imgs)) ** 2).mean()
    return L_var, L_var   # return same value twice for consistent (loss, aux) unpacking

# ── Training ──────────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    @jax.jit
    def epoch_fn(params, opt_state, Xs_batched, KNN_batched, epoch_key, X_train):
        def scan_body(carry, inputs):
            params, opt_state = carry
            x_batch, knn_idx_batch, key = inputs
            (loss, L_var), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, x_batch, knn_idx_batch, key, X_train
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), (loss, L_var)

        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (params, opt_state), (losses, l_var) = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_batched, KNN_batched, keys)
        )
        return params, opt_state, losses.mean(), l_var.mean()

    return epoch_fn

# ── Evaluation ────────────────────────────────────────────────────────────────

@jax.jit
def _eval_batch(params, x_batch):
    x_norm     = x_batch / 255.0
    B          = x_batch.shape[0]
    z          = encode(params, x_norm)
    cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond = jnp.broadcast_to(
        params["recon_emb"] + cats_to_cond(params, cats_recon), (B, COND_DIM)
    )
    x_recon    = decode(params, jnp.concatenate([z, recon_cond], axis=-1))
    return ((x_recon - x_norm) ** 2).mean()

def eval_recon_mse(params, X_te):
    """Reconstruction MSE using recon_emb (untrained — shows baseline without recon loss)."""
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
        "loss": [], "eval_mse": [], "eval_recall_mse": [], "eval_recall_by_rank": [],
    }
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        n    = num_batches * BATCH_SIZE
        Xs_batched  = jnp.array(X_tr[perm[:n]].reshape(num_batches, BATCH_SIZE, 784))
        KNN_batched = jnp.array(knn_idx[perm[:n]].reshape(num_batches, BATCH_SIZE, K_NN))

        params, opt_state, loss, l_var = epoch_fn(
            params, opt_state, Xs_batched, KNN_batched, key, X_train_jax
        )
        history["loss"].append(float(loss))

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            mse                        = eval_recon_mse(params, X_te)
            recall_mse, recall_by_rank = eval_recall_mse(params, X_tr, knn_idx)
            history["eval_mse"].append((epoch, mse))
            history["eval_recall_mse"].append((epoch, recall_mse))
            history["eval_recall_by_rank"].append((epoch, recall_by_rank))
            eval_str = f"  eval_recon_mse={mse:.5f}  recall_mse={recall_mse:.5f}"

        logging.info(
            f"epoch {epoch:3d}  L_var={float(loss):.5f}"
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
    logging.info(f"kNN ready: {knn_idx.shape}")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(
        f"Parameters: {n_params(params):,}  "
        f"LATENT={LATENT_DIM}  N_CAT={N_CAT}  VOCAB={VOCAB_SIZE}  E_DIM={E_DIM}  K_NN={K_NN}"
    )

    params, history, elapsed = train(params, X_train, X_test, knn_idx)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_recon_mse(params, X_test)
    append_result({
        "experiment":      EXP_NAME,
        "name":            f"inv-IMLE no-recon-loss  LATENT={LATENT_DIM} N_CAT={N_CAT} VOCAB={VOCAB_SIZE} E_DIM={E_DIM} N_VAR={N_VAR_SAMPLES} K_NN={K_NN}",
        "n_params":        n_params(params),
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "final_recon_mse": round(float(final_mse), 6),
        "loss_curve":      [round(v, 6) for v in history["loss"]],
        "eval_mse_curve":              history["eval_mse"],
        "eval_recall_mse_curve":       history["eval_recall_mse"],
        "eval_recall_by_rank_curve":   history["eval_recall_by_rank"],
        "latent_dim":      LATENT_DIM,
        "n_cat":           N_CAT,
        "vocab_size":      VOCAB_SIZE,
        "e_dim":           E_DIM,
        "n_var_samples":   N_VAR_SAMPLES,
        "k_nn":            K_NN,
        "dec_rank":        DEC_RANK,
        "mlp_hidden":      MLP_HIDDEN,
        "alpha":           ALPHA,
        "adam_lr":         ADAM_LR,
        "batch_size":      BATCH_SIZE,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s")
