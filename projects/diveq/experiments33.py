"""
exp33 — DiVeQ-PQ LATENT=32, G=16, D_g=2, K=512 + batch normalization at bottleneck.

Previous approaches (interleaved grouping, learned projection) failed to eliminate
lazy groups because the encoder concentrates variance in ~10-16 dims out of 32.
The projection layer gets stuck in a bad equilibrium.

Root cause: the MLP encoder doesn't receive gradient pressure to use all 32 dims
equally — it can minimize MSE by using only 10-16 "active" dims.

Fix: batch normalization at the encoder bottleneck forces each of the 32 latent
dims to have zero mean and unit variance across the batch. This makes every dim
equally informative before PQ, eliminating the lazy groups problem structurally.

Implementation: simple approximate BN (no tracked running stats):
    z_norm = (z - mean(z, dim=0)) / (std(z, dim=0) + eps)

During eval we use batch statistics over 256 samples (close to population stats).
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

EXP_NAME = "exp33"

# ── Hyperparameters ───────────────────────────────────────────────────────────

LATENT_DIM    = 32
N_GROUPS      = 16
GROUP_DIM     = LATENT_DIM // N_GROUPS   # = 2
CODEBOOK_SIZE = 512
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
        # BN affine params (scale and shift per latent dim)
        "bn_gamma": jnp.ones(LATENT_DIM),
        "bn_beta":  jnp.zeros(LATENT_DIM),
        "dec_W1": n((LATENT_DIM, HIDDEN), scale=LATENT_DIM**-0.5),
        "dec_b1": jnp.zeros(HIDDEN),
        "dec_W2": n((HIDDEN, HIDDEN),     scale=HIDDEN**-0.5),
        "dec_b2": jnp.zeros(HIDDEN),
        "dec_W3": n((HIDDEN, 784),        scale=HIDDEN**-0.5),
        "dec_b3": jnp.zeros(784),
    }
    params["codebooks"] = u(
        (N_GROUPS, CODEBOOK_SIZE, GROUP_DIM),
        lo=-1.0/CODEBOOK_SIZE, hi=1.0/CODEBOOK_SIZE
    )
    return params

# ── Encoder / Decoder ─────────────────────────────────────────────────────────

def encode(params, x):
    """x (B, 784) → z_bn (B, LATENT_DIM), batch-normalized per dim."""
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    h = jax.nn.gelu(h @ params["enc_W2"] + params["enc_b2"])
    z = h @ params["enc_W3"] + params["enc_b3"]
    # Batch normalization across batch dimension per latent dim
    z_mean = z.mean(axis=0, keepdims=True)             # (1, LATENT_DIM)
    z_std  = z.std(axis=0, keepdims=True) + 1e-6
    z_hat  = (z - z_mean) / z_std
    return z_hat * params["bn_gamma"] + params["bn_beta"]

def decode(params, zq):
    h = jax.nn.gelu(zq @ params["dec_W1"] + params["dec_b1"])
    h = jax.nn.gelu(h  @ params["dec_W2"] + params["dec_b2"])
    return jax.nn.sigmoid(h @ params["dec_W3"] + params["dec_b3"])

# ── DiVeQ (single group) ──────────────────────────────────────────────────────

def _diveq_group(C, z_g, key, training):
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
    codebooks = params["codebooks"]
    zq_groups, idx_groups, qe_groups = [], [], []

    for g in range(N_GROUPS):
        z_g = z[:, g*GROUP_DIM:(g+1)*GROUP_DIM]
        C_g = codebooks[g]
        key_g = jax.random.fold_in(key, g) if training else key
        zq_g, idx_g, qe_g = _diveq_group(C_g, z_g, key_g, training)
        zq_groups.append(zq_g)
        idx_groups.append(idx_g)
        qe_groups.append(qe_g)

    zq      = jnp.concatenate(zq_groups, axis=-1)
    indices = jnp.stack(idx_groups, axis=-1)
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
        all_indices.append(np.array(indices))
    all_indices = np.concatenate(all_indices, axis=0)
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
        f"bits={total_bits}  sigma={SIGMA}  HIDDEN={HIDDEN}  (batch norm bottleneck)"
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
        "name":             f"DiVeQ-PQ+BN G={N_GROUPS} K={CODEBOOK_SIZE} D_g={GROUP_DIM} bits={total_bits}",
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
        "bottleneck":       "batch_norm",
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  qe={final_qe:.5f}  cb_per_group={final_n_used}  {elapsed:.1f}s")
