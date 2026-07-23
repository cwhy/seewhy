"""
exp2 — Gram IMLE: basic IMLE with gram decoder (no encoder).

IMLE replaces the VAE encoder with nearest-neighbor matching in output space:

  Outer loop each epoch:
    1. Sample N_SAMPLES latent codes  z_j ~ N(0, I)
    2. Decode (no grad) → soft-mean images  x̃_j = G_θ(z_j)   shape (N_SAMPLES, 784)
    3. For each real image x_i, find nearest sample:
           σ(i) = argmin_j  ||x_i − x̃_j||²   (L2 in pixel space)
    4. Re-decode matched codes with grad, minimise NLL:
           L = −(1/n) Σ_i  log p_θ(x_i | z_{σ(i)})

  No encoder. No KL. No FREE_BITS.

Decoder: identical to vae-gram/exp29 (MLP_HIDDEN=256, DEC_RANK=16, D=144, LATENT_DIM=64).
Matching: L2 on soft-mean decoded images (efficient via dot-product expansion).
Training: cross-entropy NLL on matched latent codes (richer gradient than MSE).

Reference: Aghabozorgi et al., "Adaptive IMLE for Few-shot Pretraining-free
           Generative Modelling", ICML 2023.

Usage:
    uv run python projects/imle-gram/experiments1.py
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
from lib.viz import save_reconstruction_grid, save_sample_grid, save_learning_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp2"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_VAL_BINS  = 32
BATCH_SIZE  = 128
SEED        = 42
MAX_EPOCHS  = 100
N_SAMPLES   = 5000   # latent codes sampled per epoch for NN matching

D_R, D_C, D_V = 6, 6, 4
D    = D_R * D_C * D_V   # 144
D_RC = D_R * D_C         # 36

LATENT_DIM  = 64
DEC_RANK    = 16
MLP_HIDDEN  = 256

ADAM_LR     = 3e-4

EVAL_EVERY  = 5
VIZ_EVERY   = 20
M_EVAL      = 5000   # samples used for eval NN matching (larger → better coverage)

JSONL = Path(__file__).parent / "results.jsonl"
ROWS  = jnp.arange(784) // 28
COLS  = jnp.arange(784) % 28
BIN_CENTERS = (jnp.arange(N_VAL_BINS) + 0.5) / N_VAL_BINS * 255.0

# ── Utilities ─────────────────────────────────────────────────────────────────

def quantize(x):
    return jnp.clip((x / 255.0 * N_VAL_BINS).astype(jnp.int32), 0, N_VAL_BINS - 1)

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
        # Decoder only — no encoder params
        "W_mlp1": n((LATENT_DIM, MLP_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_mlp1": jnp.zeros(MLP_HIDDEN),
        "W_A": n((MLP_HIDDEN, DEC_RANK * D), scale=MLP_HIDDEN**-0.5),
        "W_B": n((MLP_HIDDEN, DEC_RANK * D), scale=MLP_HIDDEN**-0.5),
        "dec_row_emb": n((28, D_R), scale=D_R**-0.5),
        "dec_col_emb": n((28, D_C), scale=D_C**-0.5),
        "W_pos_dec": n((D, D_RC), scale=D_RC**-0.5),
        "W_val1": n((2*D, D), scale=D**-0.5), "b_val1": jnp.zeros(2*D),
        "W_val2": n((N_VAL_BINS, 2*D), scale=(2*D)**-0.5),
    }

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, z):
    """z (B, LATENT_DIM) → logits (B, 784, N_VAL_BINS)."""
    B     = z.shape[0]
    h     = jax.nn.gelu(z @ params["W_mlp1"] + params["b_mlp1"])
    A     = (h @ params["W_A"]).reshape(B, DEC_RANK, D)
    B_fac = (h @ params["W_B"]).reshape(B, DEC_RANK, D)
    gram  = jnp.einsum("bri,brj->bij", A, B_fac)
    def decode_pixel(r, c, g):
        re = params["dec_row_emb"][r]; ce = params["dec_col_emb"][c]
        pos_rc = (re[:, None] * ce[None, :]).reshape(D_RC)
        q  = params["W_pos_dec"] @ pos_rc
        f  = g @ q
        h1 = jax.nn.gelu(params["W_val1"] @ f + params["b_val1"])
        return params["W_val2"] @ h1
    def decode_image(g):
        return jax.vmap(decode_pixel, in_axes=(0, 0, None))(ROWS, COLS, g)
    return jax.vmap(decode_image)(gram)

# ── IMLE sampling & matching ──────────────────────────────────────────────────

def sample_and_decode(params, key, m, chunk=256):
    """
    Sample m latent codes and decode to soft-mean pixel images.
    Returns Z (m, LATENT_DIM) and X_tilde (m, 784) as numpy arrays.
    No gradients — used for NN matching only.
    """
    Z_parts, X_parts = [], []
    for i in range(0, m, chunk):
        bs = min(chunk, m - i)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (bs, LATENT_DIM))
        logits = decode(params, z)
        probs  = jax.nn.softmax(logits, axis=-1)
        x_tilde = (probs * BIN_CENTERS[None, None, :]).sum(-1)
        Z_parts.append(np.array(z))
        X_parts.append(np.array(x_tilde))
    return np.concatenate(Z_parts, axis=0), np.concatenate(X_parts, axis=0)


def find_nearest(X_real, X_tilde):
    """
    For each row in X_real find the index of the nearest row in X_tilde.
    Uses ||a-b||² = ||a||² + ||b||² - 2 a·bᵀ for efficiency.
    X_real:  (B, 784) numpy
    X_tilde: (m, 784) numpy
    Returns: (B,) int32 indices
    """
    d_r = (X_real  ** 2).sum(1, keepdims=True)    # (B, 1)
    d_t = (X_tilde ** 2).sum(1)                    # (m,)
    dists = d_r + d_t - 2.0 * (X_real @ X_tilde.T) # (B, m)
    return np.argmin(dists, axis=1)                 # (B,)

# ── Loss ──────────────────────────────────────────────────────────────────────

def imle_loss(params, x, z_matched):
    """NLL of real images under matched latent codes. Grad flows through decode only."""
    logits = decode(params, z_matched)
    bins   = quantize(x)
    log_p  = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_p, bins[..., None], axis=-1)[..., 0].mean()

# ── Training step ─────────────────────────────────────────────────────────────

def make_train_step(optimizer):
    @jax.jit
    def step(params, opt_state, x, z_matched):
        loss, grads = jax.value_and_grad(imle_loss)(params, x, z_matched)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss
    return step

# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_metrics(params, X_eval, key, m=M_EVAL):
    """
    Sample m codes, find nearest for each eval image.
    Returns match_loss (mean L2/pixel) and bin_acc.
    """
    Z, X_tilde = sample_and_decode(params, key, m)
    match_losses, bin_correct, bin_total = [], 0, 0

    for i in range(0, len(X_eval), 256):
        x_batch = np.array(X_eval[i:i+256])  # (B, 784)
        idx     = find_nearest(x_batch, X_tilde)
        matched = X_tilde[idx]               # (B, 784)
        match_losses.append(((x_batch - matched) ** 2).mean())

        # Bin accuracy: quantize the soft-mean and compare to true bins
        z_jax   = jnp.array(Z[idx])
        logits  = decode(params, z_jax)
        preds   = np.array(jnp.argmax(logits, axis=-1))            # (B, 784)
        true    = np.clip((x_batch / 255.0 * N_VAL_BINS).astype(int), 0, N_VAL_BINS - 1)
        bin_correct += (preds == true).sum()
        bin_total   += preds.size

    return float(np.mean(match_losses)), float(bin_correct / bin_total)

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, X_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    step_fn     = make_train_step(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {"match_loss": [], "match_loss_eval": [], "bin_acc": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key = next(key_gen).get()

        # ── 1. Sample & decode N_SAMPLES codes (no grad) ──────────────────────
        Z_np, X_tilde_np = sample_and_decode(params, key, N_SAMPLES)

        # ── 2. Shuffle training data ───────────────────────────────────────────
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = np.array(X_tr)[perm]

        e_loss = e_match = 0.0
        for b in range(num_batches):
            x_batch = Xs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]          # (B, 784) numpy

            # ── 3. Nearest-neighbor matching (numpy, no grad) ──────────────────
            idx       = find_nearest(x_batch, X_tilde_np)              # (B,)
            z_matched = jnp.array(Z_np[idx])                           # (B, LATENT_DIM)

            # Track L2 matching quality
            e_match += ((x_batch - X_tilde_np[idx]) ** 2).mean()

            # ── 4. Gradient step on NLL with matched codes ─────────────────────
            x_jax = jnp.array(x_batch)
            params, opt_state, loss = step_fn(params, opt_state, x_jax, z_matched)
            e_loss += float(loss)

        e_loss  /= num_batches
        e_match /= num_batches
        history["match_loss"].append(e_match)

        acc_str = eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_key = jax.random.fold_in(key, 9999)
            ml_eval, bin_acc = eval_metrics(params, X_te, eval_key)
            history["match_loss_eval"].append((epoch, ml_eval))
            history["bin_acc"].append((epoch, bin_acc))
            eval_str = f"  eval_match={ml_eval:.1f}  bin_acc={bin_acc:.1%}"

        logging.info(
            f"epoch {epoch:3d}  nll={e_loss:.4f}  match_l2={e_match:.1f}"
            f"{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

        if (epoch + 1) % VIZ_EVERY == 0:
            save_reconstruction_grid(
                params, X_te, decode, LATENT_DIM, BIN_CENTERS,
                EXP_NAME, f"epoch{epoch+1:03d}"
            )

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}  (decoder only, no encoder)")
    logging.info(f"N_SAMPLES={N_SAMPLES}  LATENT_DIM={LATENT_DIM}  MLP_HIDDEN={MLP_HIDDEN}  DEC_RANK={DEC_RANK}")

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)

    save_learning_curves(history, EXP_NAME,
                         subtitle=f"N_SAMPLES={N_SAMPLES} LATENT_DIM={LATENT_DIM}")
    save_reconstruction_grid(params, X_test, decode, LATENT_DIM, BIN_CENTERS,
                             EXP_NAME, "final")
    save_sample_grid(params, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, rows=5, cols=8)

    final_ml, final_acc = eval_metrics(params, X_test, jax.random.PRNGKey(0), m=M_EVAL)
    final_ml, final_acc = float(final_ml), float(final_acc)
    append_result({
        "experiment": EXP_NAME,
        "name": f"gram-imle N_SAMPLES={N_SAMPLES} MLP_HIDDEN={MLP_HIDDEN} DEC_RANK={DEC_RANK}",
        "n_params": n_params(params), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_bin_acc": round(final_acc, 6),
        "final_match_l2": round(final_ml, 4),
        "match_loss_curve": [round(v, 4) for v in history["match_loss"]],
        "bin_acc_curve": history["bin_acc"],
        "d_r": D_R, "d_c": D_C, "d_v": D_V,
        "latent_dim": LATENT_DIM, "dec_rank": DEC_RANK, "mlp_hidden": MLP_HIDDEN,
        "n_val_bins": N_VAL_BINS, "adam_lr": ADAM_LR,
        "n_samples": N_SAMPLES, "m_eval": M_EVAL,
    })
    logging.info(f"Done. match_l2={final_ml:.1f}  bin_acc={final_acc:.1%}  {elapsed:.1f}s")
