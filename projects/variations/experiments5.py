"""
exp5 — Variations: inverted IMLE with 2×16 categorical embedding conditioning.

Conditioning change from exp3 (8 binary bits):
  Instead of: b ~ Bernoulli(0.5)^8
  Now:        2 categorical vars, each in {0,...,15}, with learned embeddings
              c = [embed1[i], embed2[j]], total dim = N_CAT * E_DIM = 2*4 = 8
  Reconstruction path: cat=(0, 0)  — canonical reconstruction slot
  Variation path: sample N_VAR_SAMPLES random (i, j) pairs per step

Total distinct slots: 16 × 16 = 256 (same as binary bits).
The embeddings are learned, so categories can spread to cover the data manifold.

N_VAR_SAMPLES=2: 2 (i,j) pairs per image → 512 total decode calls/step.
  "2x the samples as the images" — 2 per image in the batch.
  This is fast (no checkpointing needed) but sparse coverage per step.
  Embeddings compensate by accumulating stable slot assignments across steps.

B*N_VAR = 256*2 = 512 decode/step  →  no memory issue.

Usage:
    uv run python projects/variations/experiments5.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp5"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM     = 64
N_CAT          = 2               # number of categorical variables
VOCAB_SIZE     = 16              # categories per variable  (16×16 = 256 slots total)
E_DIM          = 4               # embedding dim per variable
COND_DIM       = N_CAT * E_DIM  # = 8, same total conditioning dim as N_BITS=8
MLP_HIDDEN     = 256
DEC_RANK       = 16
D_R, D_C       = 6, 6
D_V            = 4
D_RC           = D_R * D_C        # 36
D              = D_R * D_C * D_V  # 144
ALPHA          = 1.0
BATCH_SIZE     = 256
N_VAR_SAMPLES  = 2               # 2 (i,j) pairs per image per step → 512 decode/step
MAX_EPOCHS     = 100
ADAM_LR        = 3e-4
SEED           = 42
K_NN           = 64

EVAL_EVERY  = 5

JSONL = Path(__file__).parent / "results.jsonl"
ROWS  = jnp.arange(784) // 28
COLS  = jnp.arange(784) % 28

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

    IN_DIM = LATENT_DIM + COND_DIM
    return {
        # Categorical embeddings: (N_CAT, VOCAB_SIZE, E_DIM)
        "cat_emb": n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        # Encoder
        "enc_W1": n((784, 512),         scale=784**-0.5),
        "enc_b1": jnp.zeros(512),
        "enc_W2": n((512, LATENT_DIM),  scale=512**-0.5),
        "enc_b2": jnp.zeros(LATENT_DIM),
        # Decoder
        "dec_W_mlp1": n((IN_DIM, MLP_HIDDEN),     scale=IN_DIM**-0.5),
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
    """cats (V, N_CAT) int32 → conditioning vectors (V, COND_DIM)."""
    # cat_emb: (N_CAT, VOCAB_SIZE, E_DIM)
    # cats[v, k] = category index for variable k in sample v
    embs = params["cat_emb"][jnp.arange(N_CAT)[None, :], cats]  # (V, N_CAT, E_DIM)
    return embs.reshape(cats.shape[0], COND_DIM)

# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    return h @ params["enc_W2"] + params["enc_b2"]

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, zc):
    """zc (B, LATENT_DIM+COND_DIM) → recon (B, 784) in (0,1)."""
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

# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(params, x_batch, knn_idx_batch, key, X_train):
    """
    x_batch:       (B, 784) float32 in [0, 255]
    knn_idx_batch: (B, K_NN) int32
    X_train:       (N, 784) float32 in [0, 255]
    """
    B      = x_batch.shape[0]
    x_norm = x_batch / 255.0
    z      = encode(params, x_norm)                                        # (B, LATENT_DIM)

    # Reconstruction path: canonical slot (cat1=0, cat2=0)
    cat0     = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    cond0    = cats_to_cond(params, cat0)                                  # (1, COND_DIM)
    cond0_rep = jnp.broadcast_to(cond0, (B, COND_DIM))
    x_recon  = decode(params, jnp.concatenate([z, cond0_rep], axis=-1))
    L_recon  = ((x_recon - x_norm) ** 2).mean()

    # Variation path: sample N_VAR_SAMPLES random (cat1, cat2) pairs
    cats    = jax.random.randint(key, (N_VAR_SAMPLES, N_CAT), 0, VOCAB_SIZE)  # (N_VAR, N_CAT)
    c_vecs  = cats_to_cond(params, cats)                                   # (N_VAR, COND_DIM)

    z_rep = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(B * N_VAR_SAMPLES, LATENT_DIM)
    c_rep = jnp.broadcast_to(c_vecs[None], (B, N_VAR_SAMPLES, COND_DIM)).reshape(B * N_VAR_SAMPLES, COND_DIM)
    Y_all = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1))      # (B*N_VAR, 784)
    Y_all = Y_all.reshape(B, N_VAR_SAMPLES, 784)                          # (B, N_VAR, 784)

    # Gather kNN images
    knn_imgs = (X_train / 255.0)[knn_idx_batch]                           # (B, K_NN, 784)

    # For each kNN neighbor, find closest decoded output (stop-grad: target selection only)
    sq_knn = (knn_imgs ** 2).sum(-1)                                       # (B, K_NN)
    sq_Y   = (Y_all ** 2).sum(-1)                                          # (B, N_VAR)
    dot    = jnp.einsum("bkd,bvd->bkv", knn_imgs, Y_all)                  # (B, K_NN, N_VAR)
    dists  = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot           # (B, K_NN, N_VAR)
    best_slot  = jnp.argmin(dists, axis=-1)                               # (B, K_NN)
    matched    = Y_all[jnp.arange(B)[:, None], best_slot]                 # (B, K_NN, 784)

    L_var = ((matched - jax.lax.stop_gradient(knn_imgs)) ** 2).mean()

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
    x_norm   = x_batch / 255.0
    B        = x_batch.shape[0]
    z        = encode(params, x_norm)
    cat0     = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    cond0    = cats_to_cond(params, cat0)
    cond0_rep = jnp.broadcast_to(cond0, (B, COND_DIM))
    x_recon  = decode(params, jnp.concatenate([z, cond0_rep], axis=-1))
    return ((x_recon - x_norm) ** 2).mean()

def eval_recon_mse(params, X_te):
    mses = [float(_eval_batch(params, jnp.array(X_te[i:i+256])))
            for i in range(0, len(X_te), 256)]
    return float(np.mean(mses))

EVAL_RECALL_B = 32   # images per recall mini-batch (32 × 256 slots = 8192 decode calls)

@jax.jit
def _recall_batch_jit(params, x_norm, knn_norm, all_conds):
    """x_norm: (B,784), knn_norm: (B,K_NN,784), all_conds: (N_SLOTS,COND_DIM)"""
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
    return ((matched - knn_norm) ** 2).mean()

def eval_recall_mse(params, X_tr, knn_idx, n_images=512):
    """Recall MSE: among all 256 (cat1,cat2) slots, how closely can the model
    recover each kNN neighbor? Uses training images since knn_idx is for X_tr."""
    all_cats_jax = jnp.array([[i, j] for i in range(VOCAB_SIZE) for j in range(VOCAB_SIZE)],
                              dtype=jnp.int32)
    all_conds  = cats_to_cond(params, all_cats_jax)            # (256, COND_DIM)
    X_norm     = (X_tr / 255.0).astype(np.float32)
    n          = (n_images // EVAL_RECALL_B) * EVAL_RECALL_B  # drop partial last batch
    mses = []
    for start in range(0, n, EVAL_RECALL_B):
        sl       = slice(start, start + EVAL_RECALL_B)
        x_norm   = jnp.array(X_norm[sl])                      # (B, 784)
        knn_norm = jnp.array(X_norm[knn_idx[sl]])             # (B, K_NN, 784)
        mses.append(float(_recall_batch_jit(params, x_norm, knn_norm, all_conds)))
    return float(np.mean(mses))

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, X_te, knn_idx):
    num_batches = len(X_tr) // BATCH_SIZE
    num_steps   = MAX_EPOCHS * num_batches
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    X_train_jax = jnp.array(X_tr)

    history = {"loss": [], "L_recon": [], "L_var": [], "eval_mse": [], "eval_recall_mse": []}
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
            mse        = eval_recon_mse(params, X_te)
            recall_mse = eval_recall_mse(params, X_tr, knn_idx)
            history["eval_mse"].append((epoch, mse))
            history["eval_recall_mse"].append((epoch, recall_mse))
            eval_str = f"  eval_mse={mse:.5f}  recall_mse={recall_mse:.5f}"

        logging.info(
            f"epoch {epoch:3d}  loss={float(loss):.5f}"
            f"  L_recon={float(l_recon):.5f}  L_var={float(l_var):.5f}"
            f"{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
        f"  N_VAR={N_VAR_SAMPLES}  K_NN={K_NN}"
        f"  B*N_VAR={BATCH_SIZE*N_VAR_SAMPLES} decode/step"
    )

    params, history, elapsed = train(params, X_train, X_test, knn_idx)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_recon_mse(params, X_test)
    append_result({
        "experiment":      EXP_NAME,
        "name":            f"inv-IMLE cat-emb  LATENT={LATENT_DIM} N_CAT={N_CAT} VOCAB={VOCAB_SIZE} E_DIM={E_DIM} N_VAR={N_VAR_SAMPLES} K_NN={K_NN}",
        "n_params":        n_params(params),
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "final_recon_mse": round(float(final_mse), 6),
        "loss_curve":      [round(v, 6) for v in history["loss"]],
        "L_recon_curve":   [round(v, 6) for v in history["L_recon"]],
        "L_var_curve":     [round(v, 6) for v in history["L_var"]],
        "eval_mse_curve":        history["eval_mse"],
        "eval_recall_mse_curve": history["eval_recall_mse"],
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
