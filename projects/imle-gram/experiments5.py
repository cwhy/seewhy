"""
exp5 — Gram IMLE: Adaptive IMLE (Aghabozorgi et al., ICML 2023).

Addresses the 87.1% ceiling of basic IMLE (exp1–4) by replacing the fixed
random pool + uniform matching with adaptive per-example thresholds:

  Algorithm:
    Maintain a persistent pool of N_POOL decoded samples across epochs.
    Each training example i has a threshold τᵢ (initialised to ∞).

    Each epoch:
      1. Partially refresh the pool: replace REFRESH_FRAC of samples with new
         random decodes (keeps diversity without full resample cost).
      2. For each mini-batch:
           a. Find nearest sample for each x_i → distance d_i
           b. Skip x_i if d_i < τᵢ  (already well-covered, no update needed)
           c. For active examples: compute NLL loss on matched codes, gradient step
           d. Update τᵢ ← d_i * CHANGE_COEF  (lower threshold as coverage improves)

  Key hyperparameters (from paper):
    N_POOL       — persistent pool size (paper uses ~10K; we use 2000 for speed)
    REFRESH_FRAC — fraction of pool replaced each epoch (0.2 = 20%)
    CHANGE_COEF  — threshold update factor (0.9 → threshold = 90% of current distance)
    TAU_INIT     — initial threshold (∞ → all examples active at start)

Decoder: exp1 baseline (MLP_HIDDEN=256, DEC_RANK=16, LATENT_DIM=64).

Reference: Aghabozorgi et al., "Adaptive IMLE for Few-shot Pretraining-free
           Generative Modelling", ICML 2023.

Usage:
    uv run python projects/imle-gram/experiments5.py
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

EXP_NAME = "exp5"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_VAL_BINS   = 32
BATCH_SIZE   = 128
SEED         = 42
MAX_EPOCHS   = 100

D_R, D_C, D_V = 6, 6, 4
D    = D_R * D_C * D_V   # 144
D_RC = D_R * D_C         # 36

LATENT_DIM   = 64
DEC_RANK     = 16
MLP_HIDDEN   = 256

ADAM_LR      = 3e-4

# Adaptive IMLE params
N_POOL       = 2000    # persistent decoded sample pool size
REFRESH_FRAC = 0.2     # fraction of pool refreshed each epoch
CHANGE_COEF  = 0.9     # τᵢ ← d_i * CHANGE_COEF after each match

EVAL_EVERY   = 5
VIZ_EVERY    = 20
M_EVAL       = 5000

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
    B     = z.shape[0]
    h     = jax.nn.gelu(z @ params["W_mlp1"] + params["b_mlp1"])
    A     = (h @ params["W_A"]).reshape(B, DEC_RANK, D)
    B_fac = (h @ params["W_B"]).reshape(B, DEC_RANK, D)
    gram  = jnp.einsum("bri,brj->bij", A, B_fac)
    def decode_pixel(r, c, g):
        re = params["dec_row_emb"][r]; ce = params["dec_col_emb"][c]
        pos_rc = (re[:, None] * ce[None, :]).reshape(D_RC)
        q  = params["W_pos_dec"] @ pos_rc; f = g @ q
        h1 = jax.nn.gelu(params["W_val1"] @ f + params["b_val1"])
        return params["W_val2"] @ h1
    def decode_image(g):
        return jax.vmap(decode_pixel, in_axes=(0, 0, None))(ROWS, COLS, g)
    return jax.vmap(decode_image)(gram)

# ── Pool management ───────────────────────────────────────────────────────────

def decode_to_soft_mean(params, z, chunk=256):
    """Decode z → soft-mean pixel images. Returns numpy (n, 784)."""
    parts = []
    for i in range(0, len(z), chunk):
        logits = decode(params, jnp.array(z[i:i+chunk]))
        probs  = jax.nn.softmax(logits, axis=-1)
        parts.append(np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)))
    return np.concatenate(parts, axis=0)


def init_pool(params, key):
    """Sample N_POOL codes and decode to pixel images."""
    Z = np.array(jax.random.normal(key, (N_POOL, LATENT_DIM)))
    X_tilde = decode_to_soft_mean(params, Z)
    return Z, X_tilde


def refresh_pool(params, Z, X_tilde, key):
    """Replace REFRESH_FRAC of pool with fresh samples."""
    n_refresh = max(1, int(N_POOL * REFRESH_FRAC))
    key, subkey = jax.random.split(key)
    idx_replace = np.array(jax.random.choice(key, N_POOL, shape=(n_refresh,), replace=False))
    new_z = np.array(jax.random.normal(subkey, (n_refresh, LATENT_DIM)))
    new_x = decode_to_soft_mean(params, new_z)
    Z[idx_replace]       = new_z
    X_tilde[idx_replace] = new_x
    return Z, X_tilde

# ── Matching ──────────────────────────────────────────────────────────────────

def find_nearest(X_real, X_tilde):
    """L2 nearest neighbor. Returns (indices, distances²)."""
    d_r   = (X_real  ** 2).sum(1, keepdims=True)
    d_t   = (X_tilde ** 2).sum(1)
    dists = d_r + d_t - 2.0 * (X_real @ X_tilde.T)
    idx   = np.argmin(dists, axis=1)
    return idx, dists[np.arange(len(idx)), idx]

# ── Loss ──────────────────────────────────────────────────────────────────────

def imle_loss(params, x, z_matched):
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
    Z = np.array(jax.random.normal(key, (m, LATENT_DIM)))
    X_tilde = decode_to_soft_mean(params, Z)
    match_losses, bin_correct, bin_total = [], 0, 0
    for i in range(0, len(X_eval), 256):
        x_batch = np.array(X_eval[i:i+256])
        idx, dists = find_nearest(x_batch, X_tilde)
        match_losses.append(dists.mean())
        logits = decode(params, jnp.array(Z[idx]))
        preds  = np.array(jnp.argmax(logits, axis=-1))
        true   = np.clip((x_batch / 255.0 * N_VAL_BINS).astype(int), 0, N_VAL_BINS - 1)
        bin_correct += (preds == true).sum()
        bin_total   += preds.size
    return float(np.mean(match_losses)), float(bin_correct / bin_total)

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, X_te):
    n_train     = len(X_tr)
    num_steps   = MAX_EPOCHS * (n_train // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    step_fn     = make_train_step(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = n_train // BATCH_SIZE

    # Per-example adaptive thresholds (initialised to 0 → all active,
    # since active = dist > tau and any dist > 0 is true initially)
    tau = np.zeros(n_train, dtype=np.float32)

    # Initialise persistent pool
    init_key = next(key_gen).get()
    Z_pool, X_pool = init_pool(params, init_key)
    logging.info(f"Pool initialised: {N_POOL} samples")

    history = {"match_loss": [], "match_loss_eval": [], "bin_acc": [], "active_frac": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key = next(key_gen).get()

        # ── Partially refresh pool ─────────────────────────────────────────────
        Z_pool, X_pool = refresh_pool(params, Z_pool, X_pool, key)

        # ── Shuffle training data ──────────────────────────────────────────────
        perm = np.array(jax.random.permutation(key, n_train))
        Xs   = np.array(X_tr)[perm]
        tau_perm = tau[perm]

        e_loss = e_match = 0.0
        n_active = n_total = 0

        for b in range(num_batches):
            x_batch = Xs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            tau_b   = tau_perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            # ── Nearest-neighbor matching ──────────────────────────────────────
            idx, dists = find_nearest(x_batch, X_pool)    # dists: L2²/pixel * 784
            e_match += dists.mean()
            n_total += len(x_batch)

            # ── Adaptive gating: skip well-covered examples ────────────────────
            active = dists > tau_b
            n_active += active.sum()

            # Update thresholds for all examples (whether active or not)
            tau_perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE] = np.where(
                active, dists * CHANGE_COEF, tau_b * CHANGE_COEF
            )

            if active.sum() == 0:
                continue   # entire batch already well-covered

            x_active = jnp.array(x_batch[active])
            z_active = jnp.array(Z_pool[idx[active]])
            params, opt_state, loss = step_fn(params, opt_state, x_active, z_active)
            e_loss += float(loss)

        # Write thresholds back (un-permute)
        tau[perm] = tau_perm

        e_match  /= num_batches
        active_frac = n_active / n_total
        history["match_loss"].append(e_match)
        history["active_frac"].append(active_frac)

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_key = jax.random.fold_in(key, 9999)
            ml_eval, bin_acc = eval_metrics(params, X_te, eval_key)
            history["match_loss_eval"].append((epoch, ml_eval))
            history["bin_acc"].append((epoch, bin_acc))
            eval_str = f"  eval_match={ml_eval:.1f}  bin_acc={bin_acc:.1%}"

        logging.info(
            f"epoch {epoch:3d}  match_l2={e_match:.1f}  active={active_frac:.1%}"
            f"{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

        if (epoch + 1) % VIZ_EVERY == 0:
            save_reconstruction_grid(params, X_te, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, f"epoch{epoch+1:03d}")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}  N_POOL={N_POOL}  CHANGE_COEF={CHANGE_COEF}")

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)

    save_learning_curves(history, EXP_NAME, subtitle=f"Adaptive IMLE N_POOL={N_POOL} CHANGE_COEF={CHANGE_COEF}")
    save_reconstruction_grid(params, X_test, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, "final")
    save_sample_grid(params, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, rows=5, cols=8)

    final_ml, final_acc = eval_metrics(params, X_test, jax.random.PRNGKey(0), m=M_EVAL)
    append_result({
        "experiment": EXP_NAME,
        "name": f"gram-adaptive-imle N_POOL={N_POOL} CHANGE_COEF={CHANGE_COEF}",
        "n_params": n_params(params), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_bin_acc": round(float(final_acc), 6),
        "final_match_l2": round(float(final_ml), 4),
        "match_loss_curve": [round(float(v), 4) for v in history["match_loss"]],
        "active_frac_curve": [round(float(v), 4) for v in history["active_frac"]],
        "bin_acc_curve": history["bin_acc"],
        "latent_dim": LATENT_DIM, "dec_rank": DEC_RANK, "mlp_hidden": MLP_HIDDEN,
        "n_val_bins": N_VAL_BINS, "adam_lr": ADAM_LR,
        "n_pool": N_POOL, "refresh_frac": REFRESH_FRAC, "change_coef": CHANGE_COEF,
    })
    logging.info(f"Done. match_l2={final_ml:.1f}  bin_acc={final_acc:.1%}  {elapsed:.1f}s")
