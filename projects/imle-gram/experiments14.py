"""
exp14 — Hebbian aggregation with exact 1-or-2 active slots.

Same Hebbian architecture as exp13. Sampling is more restrictive and cleaner:
  - Exactly 1 or 2 non-zero free slots per sample (never all-zero, never 3+).
  - n_active ~ Uniform({1, 2})
  - slot_1  ~ Uniform(1, N_VARS-1)
  - slot_2  ~ Uniform({1..N_VARS-1} \ {slot_1})   # without replacement, skip trick
  - code_i  ~ Uniform(1, N_CODES-1)               # always non-zero code

This is the regime where the Hebbian interpretation is clearest:
  - 1 active: M = u_1 ⊗ v_1  (pure rank-1)
  - 2 active: M = u_1 ⊗ v_1 + u_2 ⊗ v_2  (rank-2 superposition)

Usage:
    uv run python projects/imle-gram/experiments14.py
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

EXP_NAME = "exp14"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_VAL_BINS = 32
BATCH_SIZE = 128
SEED       = 42
MAX_EPOCHS = 100
N_SAMPLES  = 1000

D_R, D_C   = 6, 6
D_RC       = D_R * D_C   # 36

N_VARS   = 16    # slots; slot 0 fixed to code 0
N_CODES  = 32    # shared codebook size (doubled — no position info)
D_EMB    = 16    # embedding dim per code
D_U      = 16    # u_i / v_i dim; M is (D_U, D_U) → flattened to D_U²=256

LATENT_DIM = D_U * D_U    # 256 — flattened Gram matrix M
MLP_HIDDEN = 64            # f_dim = 6*6*64 = 2304
TEMP       = 10.0
# No P_ZERO — sampling is exact: 1 or 2 active slots per sample

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
        # shared codebook
        "codebook": n((N_CODES, D_EMB), scale=D_EMB**-0.5),
        # Hebbian projections: z_i → (u_i, v_i)
        "W_u": n((D_EMB, D_U), scale=D_EMB**-0.5),
        "b_u": jnp.zeros(D_U),
        "W_v": n((D_EMB, D_U), scale=D_EMB**-0.5),
        "b_v": jnp.zeros(D_U),
        # M.reshape(-1) → h  (LATENT_DIM = D_U²)
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

# ── Hebbian aggregation ────────────────────────────────────────────────────────

def _to_gram(Z, params):
    """Z (B, N_VARS, D_EMB) → M_flat (B, D_U²)."""
    U = Z @ params["W_u"] + params["b_u"]             # (B, N_VARS, D_U)
    V = Z @ params["W_v"] + params["b_v"]             # (B, N_VARS, D_U)
    M = jnp.einsum("bni,bnj->bij", U, V)              # (B, D_U, D_U)
    return M.reshape(Z.shape[0], -1)                   # (B, D_U²)

def hard_lookup(params, indices):
    """indices (B, N_VARS) → z_flat (B, LATENT_DIM) via hard Hebbian sum."""
    Z = params["codebook"][indices, :]                 # (B, N_VARS, D_EMB)
    return _to_gram(Z, params)

def soft_lookup(params, indices):
    """indices (B, N_VARS) → z_flat (B, LATENT_DIM) via soft Hebbian sum."""
    oh = jax.nn.one_hot(indices, N_CODES)              # (B, N_VARS, N_CODES)
    w  = jax.nn.softmax(oh * TEMP, axis=-1)            # (B, N_VARS, N_CODES)
    Z  = w @ params["codebook"]                        # (B, N_VARS, D_EMB)
    return _to_gram(Z, params)

def sample_indices(key, m):
    """Exactly 1 or 2 active (non-zero) free slots; slot 0 always 0.

    slot_1  ~ Uniform(1, N_VARS-1)
    slot_2  ~ Uniform({1..N_VARS-1} minus {slot_1})  via skip trick
    code_i  ~ Uniform(1, N_CODES-1)
    n_active ~ Uniform({1, 2}) — if 1, slot_2 stays 0
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    n_active = jax.random.randint(k1, (m,), 1, 3)           # 1 or 2
    slot_1   = jax.random.randint(k2, (m,), 1, N_VARS)      # in {1..N_VARS-1}
    slot_2_r = jax.random.randint(k3, (m,), 1, N_VARS - 1)  # in {1..N_VARS-2}
    slot_2   = slot_2_r + (slot_2_r >= slot_1).astype(jnp.int32)  # skip slot_1
    code_1   = jax.random.randint(k4, (m,), 1, N_CODES)
    code_2   = jax.random.randint(k5, (m,), 1, N_CODES)
    idx = jnp.zeros((m, N_VARS), dtype=jnp.int32)
    idx = idx.at[jnp.arange(m), slot_1].set(code_1)
    idx = idx.at[jnp.arange(m), slot_2].set(jnp.where(n_active == 2, code_2, 0))
    return idx

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
    logging.info(f"Parameters: {n_params(params):,}  (Hebbian N_CODES={N_CODES} D_EMB={D_EMB} D_U={D_U} n_active=1-2)")

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)

    save_learning_curves(history, EXP_NAME, subtitle=f"Hebbian N_CODES={N_CODES} D_U={D_U} n_active=1-2")
    save_reconstruction_grid(params, X_test, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, "final")
    save_sample_grid(params, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, rows=5, cols=8)

    final_ml, final_acc = eval_metrics(params, X_test, jax.random.PRNGKey(0), m=M_EVAL)
    append_result({
        "experiment": EXP_NAME,
        "name": f"imle-hebbian N_CODES={N_CODES} D_EMB={D_EMB} D_U={D_U} n_active=1-2",
        "n_params": n_params(params), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_bin_acc": round(float(final_acc), 6), "final_match_l2": round(float(final_ml), 4),
        "match_loss_curve": [round(float(v), 4) for v in history["match_loss"]],
        "bin_acc_curve": history["bin_acc"],
        "n_vars": N_VARS, "n_codes": N_CODES, "d_emb": D_EMB, "d_u": D_U, "n_active": "1-2",
        "n_val_bins": N_VAL_BINS, "adam_lr": ADAM_LR, "n_samples": N_SAMPLES, "m_eval": M_EVAL,
    })
    logging.info(f"Done. match_l2={final_ml:.1f}  bin_acc={final_acc:.1%}  {elapsed:.1f}s")
