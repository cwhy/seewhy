"""
exp23 — DiVeQ with product quantization (G=4 groups, K=512 per group).

Splits the latent vector into G independent sub-vectors and applies DiVeQ to
each group separately. This multiplies the information capacity:

    Total bits = G × log₂(K) = 4 × 9 = 36 bits  (vs 9 bits in exp9)

Each group has its own codebook of shape (K, D_g) where D_g = LATENT_DIM // G.
The G quantized sub-vectors are concatenated to form zq.

Architecture:
    MLP encoder  784 → 512 → 512 → LATENT_DIM (=16)
    DiVeQ-PQ     G=4 groups of 4 dims, each with K=512 codewords → 36 bits
    MLP decoder  LATENT_DIM → 512 → 512 → 784

Usage:
    uv run python projects/diveq/scripts/run_experiments.py exp13
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

EXP_NAME = "exp23"

# ── Hyperparameters ───────────────────────────────────────────────────────────

LATENT_DIM    = 16
N_GROUPS      = 16            # product quantization groups
GROUP_DIM     = LATENT_DIM // N_GROUPS   # = 4 dims per group
CODEBOOK_SIZE = 256          # K per group; total bits = N_GROUPS * log2(K) = 36
SIGMA         = 0.01
HIDDEN        = 512
BATCH_SIZE    = 128
MAX_EPOCHS    = 200
ADAM_LR       = 3e-4
SEED          = 42
EVAL_EVERY    = 10

JSONL = Path(__file__).parent / "results.jsonl"

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    def u(shape, lo, hi):
        return jax.random.uniform(next(key_gen).get(), shape, minval=lo, maxval=hi)

    params = {
        "enc_W1": n((784,    HIDDEN),     scale=784**-0.5),
        "enc_b1": jnp.zeros(HIDDEN),
        "enc_W2": n((HIDDEN, HIDDEN),     scale=HIDDEN**-0.5),
        "enc_b2": jnp.zeros(HIDDEN),
        "enc_W3": n((HIDDEN, LATENT_DIM), scale=HIDDEN**-0.5),
        "enc_b3": jnp.zeros(LATENT_DIM),
        "dec_W1": n((LATENT_DIM, HIDDEN), scale=LATENT_DIM**-0.5),
        "dec_b1": jnp.zeros(HIDDEN),
        "dec_W2": n((HIDDEN, HIDDEN),     scale=HIDDEN**-0.5),
        "dec_b2": jnp.zeros(HIDDEN),
        "dec_W3": n((HIDDEN, 784),        scale=HIDDEN**-0.5),
        "dec_b3": jnp.zeros(784),
    }
    # Codebook init: uniform(-3, 3) covers ±3σ of normalized latent (z ~ N(0,I) after LN)
    params["codebooks"] = u((N_GROUPS, CODEBOOK_SIZE, GROUP_DIM), lo=-3.0, hi=3.0)
    return params

# ── Encoder / Decoder ─────────────────────────────────────────────────────────

def encode(params, x):
    """x (B, 784) → z (B, LATENT_DIM), layer-normalized so z_i ~ N(0,1)."""
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    h = jax.nn.gelu(h @ params["enc_W2"] + params["enc_b2"])
    z = h @ params["enc_W3"] + params["enc_b3"]
    # Normalize across latent dims so each dimension is ~unit-scale — critical for D_g=1
    z_mean = z.mean(axis=-1, keepdims=True)
    z_std  = z.std(axis=-1, keepdims=True) + 1e-6
    return (z - z_mean) / z_std

def decode(params, zq):
    """zq (B, LATENT_DIM) → recon (B, 784) in (0, 1)."""
    h = jax.nn.gelu(zq @ params["dec_W1"] + params["dec_b1"])
    h = jax.nn.gelu(h  @ params["dec_W2"] + params["dec_b2"])
    return jax.nn.sigmoid(h @ params["dec_W3"] + params["dec_b3"])

# ── DiVeQ (single group) ──────────────────────────────────────────────────────

def _diveq_group(C, z_g, key, training):
    """
    C:   (K, D_g) codebook for this group
    z_g: (B, D_g) encoder output slice
    Returns zq_g (B, D_g), indices (B,), qe scalar
    """
    z_sq = (z_g ** 2).sum(-1, keepdims=True)
    c_sq = (C ** 2).sum(-1)
    dists = z_sq + c_sq - 2.0 * (z_g @ C.T)
    indices = jnp.argmin(dists, axis=-1)
    c_star = C[indices]

    qe = ((jax.lax.stop_gradient(z_g) - jax.lax.stop_gradient(c_star)) ** 2).mean()

    if training:
        d_tilde = jax.lax.stop_gradient(c_star) - jax.lax.stop_gradient(z_g)
        v  = jax.random.normal(key, z_g.shape) * SIGMA
        vd = v + d_tilde
        norm_vd   = jnp.linalg.norm(vd, axis=-1, keepdims=True)
        direction = jax.lax.stop_gradient(vd / (norm_vd + 1e-8))
        magnitude = jnp.linalg.norm(c_star - z_g, axis=-1, keepdims=True)
        zq_g = z_g + magnitude * direction
    else:
        zq_g = c_star

    return zq_g, indices, qe

# ── DiVeQ-PQ layer ───────────────────────────────────────────────────────────

def diveq_pq(params, z, key, training=True):
    """
    z:   (B, LATENT_DIM) encoder output
    Returns:
        zq:      (B, LATENT_DIM) product-quantized output
        indices: (B, G)          per-group codeword indices
        qe:      scalar          mean quantization error across groups
    """
    codebooks = params["codebooks"]   # (G, K, D_g)
    zq_groups, idx_groups, qe_groups = [], [], []

    for g in range(N_GROUPS):
        z_g = z[:, g*GROUP_DIM:(g+1)*GROUP_DIM]         # (B, D_g)
        C_g = codebooks[g]                               # (K, D_g)
        key_g = jax.random.fold_in(key, g) if training else key
        zq_g, idx_g, qe_g = _diveq_group(C_g, z_g, key_g, training)
        zq_groups.append(zq_g)
        idx_groups.append(idx_g)
        qe_groups.append(qe_g)

    zq      = jnp.concatenate(zq_groups, axis=-1)      # (B, LATENT_DIM)
    indices = jnp.stack(idx_groups, axis=-1)            # (B, G)
    qe      = sum(qe_groups) / N_GROUPS
    return zq, indices, qe

# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(params, x_batch, key):
    x = x_batch / 255.0
    z  = encode(params, x)
    zq, _, qe = diveq_pq(params, z, key, training=True)
    x_recon = decode(params, zq)
    recon = ((x_recon - x) ** 2).mean()
    return recon, (recon, qe)

# ── Epoch fn ──────────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    @jax.jit
    def epoch_fn(params, opt_state, Xs_batched, epoch_key):
        def scan_body(carry, inputs):
            params, opt_state = carry
            x_batch, key = inputs
            (loss, (recon, qe)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, x_batch, key
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), (loss, recon, qe)

        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (params, opt_state), (losses, recons, qes) = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_batched, keys)
        )
        return params, opt_state, losses.mean(), recons.mean(), qes.mean()

    return epoch_fn

# ── Evaluation ────────────────────────────────────────────────────────────────

@jax.jit
def _eval_batch(params, x_batch):
    x = x_batch / 255.0
    z = encode(params, x)
    dummy_key = jnp.zeros(2, dtype=jnp.uint32)
    zq, indices, qe = diveq_pq(params, z, dummy_key, training=False)
    x_recon = decode(params, zq)
    recon_mse = ((x_recon - x) ** 2).mean()
    return recon_mse, indices, qe

def eval_metrics(params, X_te):
    mses, qes, all_indices = [], [], []
    for i in range(0, len(X_te), 256):
        batch = jnp.array(X_te[i:i+256])
        mse, indices, qe = _eval_batch(params, batch)
        mses.append(float(mse))
        qes.append(float(qe))
        all_indices.append(np.array(indices))   # (batch, G)
    all_indices = np.concatenate(all_indices, axis=0)   # (N_te, G)
    # Per-group codebook utilization
    n_used_per_group = [len(np.unique(all_indices[:, g])) for g in range(N_GROUPS)]
    return float(np.mean(mses)), float(np.mean(qes)), n_used_per_group

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, X_te):
    num_batches = len(X_tr) // BATCH_SIZE
    num_steps   = MAX_EPOCHS * num_batches
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    history = {"loss": [], "recon": [], "qe": [], "eval_mse": [], "eval_qe": [], "codebook_use": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        n    = num_batches * BATCH_SIZE
        Xs_batched = jnp.array(X_tr[perm[:n]].reshape(num_batches, BATCH_SIZE, 784))

        params, opt_state, loss, recon, qe = epoch_fn(params, opt_state, Xs_batched, key)
        history["loss"].append(float(loss))
        history["recon"].append(float(recon))
        history["qe"].append(float(qe))

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_mse, eval_qe, n_used = eval_metrics(params, X_te)
            history["eval_mse"].append((epoch, eval_mse))
            history["eval_qe"].append((epoch, eval_qe))
            history["codebook_use"].append((epoch, n_used))
            eval_str = f"  eval_mse={eval_mse:.5f}  cb={n_used}"

        logging.info(
            f"epoch {epoch:3d}  loss={float(loss):.5f}"
            f"  recon={float(recon):.5f}  qe={float(qe):.5f}"
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

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    total_bits = N_GROUPS * int(np.log2(CODEBOOK_SIZE))
    logging.info(
        f"Parameters: {n_params(params):,}  "
        f"LATENT={LATENT_DIM}  G={N_GROUPS}  D_g={GROUP_DIM}  K={CODEBOOK_SIZE}  "
        f"bits={total_bits}  sigma={SIGMA}  HIDDEN={HIDDEN}"
    )

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse, final_qe, final_n_used = eval_metrics(params, X_test)
    total_bits = N_GROUPS * int(np.log2(CODEBOOK_SIZE))
    append_result({
        "experiment":       EXP_NAME,
        "name":             f"DiVeQ-PQ G={N_GROUPS} K={CODEBOOK_SIZE} D_g={GROUP_DIM} bits={total_bits}",
        "n_params":         n_params(params),
        "epochs":           MAX_EPOCHS,
        "time_s":           round(elapsed, 1),
        "final_recon_mse":  round(float(final_mse), 6),
        "final_qe":         round(float(final_qe), 6),
        "final_codebook_use": sum(final_n_used),
        "final_codebook_per_group": final_n_used,
        "loss_curve":       [round(v, 6) for v in history["loss"]],
        "recon_curve":      [round(v, 6) for v in history["recon"]],
        "qe_curve":         [round(v, 6) for v in history["qe"]],
        "eval_mse_curve":   history["eval_mse"],
        "n_groups":         N_GROUPS,
        "group_dim":        GROUP_DIM,
        "codebook_size":    CODEBOOK_SIZE,
        "total_bits":       total_bits,
        "latent_dim":       LATENT_DIM,
        "sigma":            SIGMA,
        "hidden":           HIDDEN,
        "adam_lr":          ADAM_LR,
        "batch_size":       BATCH_SIZE,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  qe={final_qe:.5f}  cb_per_group={final_n_used}  {elapsed:.1f}s")
