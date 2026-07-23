"""
exp9 — IMLE with shared codebook + one fixed-zero slot.

Changes from exp8:
  - Single shared codebook (N_CODES=16, D_EMB=16) instead of per-variable codebooks.
  - D_EMB increased 8→16 to compensate for sharing.
  - Variable 0 is always fixed to code 0; the remaining N_VARS-1 are sampled freely.

  codebook: (N_CODES=16, D_EMB=16)   — shared across all N_VARS slots
  indices:  (m, N_VARS)   where indices[:,0] == 0  always

Soft lookup during training:
  w = softmax(one_hot(indices) * TEMP)   # (B, N_VARS, N_CODES)
  z = (w @ codebook).reshape(B, -1)      # (B, N_VARS*D_EMB)

Usage:
    uv run python projects/imle-gram/experiments9.py
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

EXP_NAME = "exp9"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_VAL_BINS = 32
BATCH_SIZE = 128
SEED       = 42
MAX_EPOCHS = 100
N_SAMPLES  = 1000

D_R, D_C   = 6, 6
D_RC       = D_R * D_C   # 36

N_VARS   = 16    # latent slots; slot 0 is fixed to code 0
N_CODES  = 16    # shared codebook size
D_EMB    = 16    # embedding dim (increased from 8 to compensate for sharing)

LATENT_DIM = N_VARS * D_EMB   # 256
MLP_HIDDEN = 64                # f_dim = 6*6*64 = 2304
TEMP       = 10.0

ADAM_LR    = 3e-4

EVAL_EVERY = 5
VIZ_EVERY  = 20
M_EVAL     = 5000

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
    f_dim = D_R * D_C * MLP_HIDDEN   # 2304
    return {
        # single shared codebook
        "codebook": n((N_CODES, D_EMB), scale=D_EMB**-0.5),
        # latent → h
        "W_mlp1": n((LATENT_DIM, MLP_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_mlp1": jnp.zeros(MLP_HIDDEN),
        # positional embeddings
        "row_emb": n((28, D_R), scale=D_R**-0.5),
        "col_emb": n((28, D_C), scale=D_C**-0.5),
        # per-pixel head: (r⊗c⊗h) → logits
        "W_val1": n((MLP_HIDDEN, f_dim), scale=f_dim**-0.5),
        "b_val1": jnp.zeros(MLP_HIDDEN),
        "W_val2": n((N_VAL_BINS, MLP_HIDDEN), scale=MLP_HIDDEN**-0.5),
        "b_val2": jnp.zeros(N_VAL_BINS),
    }

# ── Codebook lookup ────────────────────────────────────────────────────────────

def hard_lookup(params, indices):
    """indices (B, N_VARS) int → z (B, LATENT_DIM) hard embedding."""
    z = params["codebook"][indices, :]        # (B, N_VARS, D_EMB)
    return z.reshape(indices.shape[0], LATENT_DIM)

def soft_lookup(params, indices):
    """indices (B, N_VARS) int → z (B, LATENT_DIM) softmax-relaxed."""
    oh = jax.nn.one_hot(indices, N_CODES)     # (B, N_VARS, N_CODES)
    w  = jax.nn.softmax(oh * TEMP, axis=-1)   # (B, N_VARS, N_CODES)
    z  = w @ params["codebook"]               # (B, N_VARS, D_EMB)
    return z.reshape(indices.shape[0], LATENT_DIM)

def sample_indices(key, m):
    """Sample m index vectors; slot 0 is always code 0."""
    free  = jax.random.randint(key, (m, N_VARS - 1), 0, N_CODES)
    zeros = jnp.zeros((m, 1), dtype=jnp.int32)
    return jnp.concatenate([zeros, free], axis=1)   # (m, N_VARS)

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, z):
    """z (B, LATENT_DIM) → logits (B, 784, N_VAL_BINS)."""
    h = jax.nn.gelu(z @ params["W_mlp1"] + params["b_mlp1"])   # (B, MLP_HIDDEN)

    def decode_image(h_i):
        def decode_pixel(r, c):
            re = params["row_emb"][r]
            ce = params["col_emb"][c]
            f  = (re[:, None, None] * ce[None, :, None] * h_i[None, None, :]).reshape(-1)
            h1 = jax.nn.gelu(params["W_val1"] @ f + params["b_val1"])
            return params["W_val2"] @ h1 + params["b_val2"]
        return jax.vmap(decode_pixel, in_axes=(0, 0))(ROWS, COLS)

    return jax.vmap(decode_image)(h)

# ── IMLE sampling & matching ──────────────────────────────────────────────────

def sample_and_decode(params, key, m, chunk=256):
    I_parts, X_parts = [], []
    for i in range(0, m, chunk):
        bs = min(chunk, m - i)
        key, subkey = jax.random.split(key)
        idx = sample_indices(subkey, bs)
        z   = hard_lookup(params, idx)
        logits  = decode(params, z)
        probs   = jax.nn.softmax(logits, axis=-1)
        x_tilde = (probs * BIN_CENTERS[None, None, :]).sum(-1)
        I_parts.append(np.array(idx))
        X_parts.append(np.array(x_tilde))
    return np.concatenate(I_parts, axis=0), np.concatenate(X_parts, axis=0)


def find_nearest(X_real, X_tilde):
    d_r   = (X_real  ** 2).sum(1, keepdims=True)
    d_t   = (X_tilde ** 2).sum(1)
    dists = d_r + d_t - 2.0 * (X_real @ X_tilde.T)
    return np.argmin(dists, axis=1)

# ── Loss & training step ──────────────────────────────────────────────────────

def imle_loss(params, x, matched_indices):
    z_soft = soft_lookup(params, matched_indices)
    logits = decode(params, z_soft)
    bins   = quantize(x)
    log_p  = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_p, bins[..., None], axis=-1)[..., 0].mean()

def make_train_step(optimizer):
    @jax.jit
    def step(params, opt_state, x, matched_indices):
        loss, grads = jax.value_and_grad(imle_loss)(params, x, matched_indices)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss
    return step

# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_metrics(params, X_eval, key, m=M_EVAL):
    I_pool, X_tilde = sample_and_decode(params, key, m)
    match_losses, bin_correct, bin_total = [], 0, 0
    for i in range(0, len(X_eval), 256):
        x_batch = np.array(X_eval[i:i+256])
        idx     = find_nearest(x_batch, X_tilde)
        match_losses.append(((x_batch - X_tilde[idx]) ** 2).mean())
        logits  = decode(params, hard_lookup(params, jnp.array(I_pool[idx])))
        preds   = np.array(jnp.argmax(logits, axis=-1))
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
        key  = next(key_gen).get()
        I_np, X_tilde_np = sample_and_decode(params, key, N_SAMPLES)
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = np.array(X_tr)[perm]

        e_loss = e_match = 0.0
        for b in range(num_batches):
            x_batch   = Xs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            idx       = find_nearest(x_batch, X_tilde_np)
            e_match  += ((x_batch - X_tilde_np[idx]) ** 2).mean()
            params, opt_state, loss = step_fn(
                params, opt_state, jnp.array(x_batch), jnp.array(I_np[idx])
            )
            e_loss += float(loss)

        e_loss /= num_batches; e_match /= num_batches
        history["match_loss"].append(e_match)

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            ml_eval, bin_acc = eval_metrics(params, X_te, jax.random.fold_in(key, 9999))
            history["match_loss_eval"].append((epoch, ml_eval))
            history["bin_acc"].append((epoch, bin_acc))
            eval_str = f"  eval_match={ml_eval:.1f}  bin_acc={bin_acc:.1%}"

        logging.info(f"epoch {epoch:3d}  nll={e_loss:.4f}  match_l2={e_match:.1f}{eval_str}  {time.perf_counter()-t0:.1f}s")

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
    logging.info(f"Parameters: {n_params(params):,}  (shared codebook N_CODES={N_CODES} D_EMB={D_EMB} slot0=fixed)")
    logging.info(f"Prior space: {N_CODES}^{N_VARS-1} = {N_CODES**(N_VARS-1):.2e} discrete codes")

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)

    save_learning_curves(history, EXP_NAME, subtitle=f"shared-codebook N_CODES={N_CODES} D_EMB={D_EMB} slot0=0")
    save_reconstruction_grid(params, X_test, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, "final")
    save_sample_grid(params, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, rows=5, cols=8)

    final_ml, final_acc = eval_metrics(params, X_test, jax.random.PRNGKey(0), m=M_EVAL)
    append_result({
        "experiment": EXP_NAME,
        "name": f"imle-shared-cb N_CODES={N_CODES} D_EMB={D_EMB} N_VARS={N_VARS} TEMP={TEMP}",
        "n_params": n_params(params), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_bin_acc": round(float(final_acc), 6), "final_match_l2": round(float(final_ml), 4),
        "match_loss_curve": [round(float(v), 4) for v in history["match_loss"]],
        "bin_acc_curve": history["bin_acc"],
        "n_vars": N_VARS, "n_codes": N_CODES, "d_emb": D_EMB, "temp": TEMP,
        "n_val_bins": N_VAL_BINS, "adam_lr": ADAM_LR, "n_samples": N_SAMPLES, "m_eval": M_EVAL,
    })
    logging.info(f"Done. match_l2={final_ml:.1f}  bin_acc={final_acc:.1%}  {elapsed:.1f}s")
