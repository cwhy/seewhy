"""
exp4 — Variations: inverted IMLE with 4-dimensional Gaussian conditioning.

Change from exp3:
  Instead of: 8 binary bits b ~ Bernoulli(0.5)^8
  Now:        4 continuous Gaussian vars c ~ N(0, I_4)
  Reconstruction path: c = 0^4 (mean of Gaussian, same idea as b=0⁸)
  Variation path: sample N_VAR_SAMPLES vectors from N(0,1)^4

This gives a smooth, continuous conditioning space vs the discrete 256-slot space.
N_GAUSS=4 keeps the conditioning dimension small; N_VAR_SAMPLES=16 (doubled vs exp3)
gives richer coverage per step.

Memory fix: B*N_VAR = 256*16 = 4096 decode calls/step → need gradient checkpointing
on the variation decode path to avoid ~15GB activation storage.
With checkpointing: ~5GB, safe on RTX 4090.

Usage:
    uv run python projects/variations/experiments4.py
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

EXP_NAME = "exp4"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM     = 64
N_GAUSS        = 4               # Gaussian conditioning dims (replaces N_BITS=8)
MLP_HIDDEN     = 256
DEC_RANK       = 16
D_R, D_C       = 6, 6
D_V            = 4
D_RC           = D_R * D_C        # 36
D              = D_R * D_C * D_V  # 144
ALPHA          = 1.0
BATCH_SIZE     = 256               # B*N_VAR=4096 decode calls/step — needs checkpointing
N_VAR_SAMPLES  = 16               # doubled vs exp3
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

    IN_DIM = LATENT_DIM + N_GAUSS
    return {
        "enc_W1": n((784, 512),         scale=784**-0.5),
        "enc_b1": jnp.zeros(512),
        "enc_W2": n((512, LATENT_DIM),  scale=512**-0.5),
        "enc_b2": jnp.zeros(LATENT_DIM),
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

# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    return h @ params["enc_W2"] + params["enc_b2"]

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, zc):
    """zc (B, LATENT_DIM+N_GAUSS) → recon (B, 784) in (0,1)."""
    B     = zc.shape[0]
    h     = jax.nn.gelu(zc @ params["dec_W_mlp1"] + params["dec_b_mlp1"])
    A     = (h @ params["dec_W_A"]).reshape(B, DEC_RANK, D)
    B_fac = (h @ params["dec_W_B"]).reshape(B, DEC_RANK, D)
    gram  = jnp.einsum("bri,brj->bij", A, B_fac)

    def decode_pixel(r, c, g):
        re    = params["dec_row_emb"][r]
        ce    = params["dec_col_emb"][c]
        pos_rc = (re[:, None] * ce[None, :]).reshape(D_RC)
        q     = params["dec_W_pos"] @ pos_rc
        f     = g @ q
        h1    = jax.nn.gelu(params["dec_W_val1"] @ f + params["dec_b_val1"])
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

    # Reconstruction path: c = 0^4 (mean of Gaussian)
    zeros_g = jnp.zeros((B, N_GAUSS))
    x_recon = decode(params, jnp.concatenate([z, zeros_g], axis=-1))
    L_recon = ((x_recon - x_norm) ** 2).mean()

    # Variation path: sample N_VAR_SAMPLES Gaussian conditioning vectors
    c_samples = jax.random.normal(key, (N_VAR_SAMPLES, N_GAUSS))          # (N_VAR, N_GAUSS)

    z_rep = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(B * N_VAR_SAMPLES, LATENT_DIM)
    c_rep = jnp.broadcast_to(c_samples[None], (B, N_VAR_SAMPLES, N_GAUSS)).reshape(B * N_VAR_SAMPLES, N_GAUSS)

    # Gradient checkpointing: avoids storing 4096-image activations (~15GB → ~5GB)
    decode_ckpt = jax.checkpoint(decode)
    Y_all_flat  = decode_ckpt(params, jnp.concatenate([z_rep, c_rep], axis=-1))  # (B*N_VAR, 784)
    Y_all = Y_all_flat.reshape(B, N_VAR_SAMPLES, 784)                     # (B, N_VAR, 784)

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
    x_norm  = x_batch / 255.0
    B       = x_batch.shape[0]
    z       = encode(params, x_norm)
    zeros_g = jnp.zeros((B, N_GAUSS))
    x_recon = decode(params, jnp.concatenate([z, zeros_g], axis=-1))
    return ((x_recon - x_norm) ** 2).mean()

def eval_recon_mse(params, X_te):
    mses = [float(_eval_batch(params, jnp.array(X_te[i:i+256])))
            for i in range(0, len(X_te), 256)]
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

    history = {"loss": [], "L_recon": [], "L_var": [], "eval_mse": []}
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
            mse = eval_recon_mse(params, X_te)
            history["eval_mse"].append((epoch, mse))
            eval_str = f"  eval_mse={mse:.5f}"

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
        f"LATENT={LATENT_DIM}  N_GAUSS={N_GAUSS}  N_VAR={N_VAR_SAMPLES}  K_NN={K_NN}"
        f"  B*N_VAR={BATCH_SIZE*N_VAR_SAMPLES} decode/step (checkpointed)"
    )

    params, history, elapsed = train(params, X_train, X_test, knn_idx)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_recon_mse(params, X_test)
    append_result({
        "experiment":      EXP_NAME,
        "name":            f"inverted-IMLE Gaussian  LATENT={LATENT_DIM} N_GAUSS={N_GAUSS} N_VAR={N_VAR_SAMPLES} K_NN={K_NN}",
        "n_params":        n_params(params),
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "final_recon_mse": round(float(final_mse), 6),
        "loss_curve":      [round(v, 6) for v in history["loss"]],
        "L_recon_curve":   [round(v, 6) for v in history["L_recon"]],
        "L_var_curve":     [round(v, 6) for v in history["L_var"]],
        "eval_mse_curve":  history["eval_mse"],
        "latent_dim":      LATENT_DIM,
        "n_gauss":         N_GAUSS,
        "n_var_samples":   N_VAR_SAMPLES,
        "k_nn":            K_NN,
        "dec_rank":        DEC_RANK,
        "mlp_hidden":      MLP_HIDDEN,
        "alpha":           ALPHA,
        "adam_lr":         ADAM_LR,
        "batch_size":      BATCH_SIZE,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s")
