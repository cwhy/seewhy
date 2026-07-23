"""
exp4 — DiVeQ on MNIST: VQ-VAE with Distributional Vector Quantization.

DiVeQ replaces the standard VQ + straight-through estimator + commitment loss
with a single gradient path through the quantization error magnitude:

    zq = z + ‖c* − z‖₂ · sg[normalize(v + d̃)]

where c* = nearest codeword, v ~ N(0, σ²I), d̃ = sg[c*] − sg[z].
The direction sg[·] is fully stop-gradiented; gradient flows only through the
scalar ‖c* − z‖₂, which is differentiable w.r.t. both z and c*.
No commitment loss; no STE; one reconstruction loss drives everything.

Architecture:
    MLP encoder  784 → 512 → 512 → LATENT_DIM
    DiVeQ        LATENT_DIM → LATENT_DIM  (codebook size K)
    MLP decoder  LATENT_DIM → 512 → 512 → 784

Usage:
    uv run python projects/diveq/scripts/run_experiments.py exp1
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

# ── Hyperparameters ───────────────────────────────────────────────────────────

LATENT_DIM    = 32
CODEBOOK_SIZE = 256        # K codewords (7 bits)
SIGMA         = 0.01       # noise std; paper shows accuracy improves as σ² → 0
HIDDEN        = 512
BATCH_SIZE    = 128
MAX_EPOCHS    = 50
ADAM_LR       = 3e-4
SEED          = 42
EVAL_EVERY    = 5

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

    return {
        # Encoder: 784 → HIDDEN → HIDDEN → LATENT_DIM
        "enc_W1": n((784,    HIDDEN),     scale=784**-0.5),
        "enc_b1": jnp.zeros(HIDDEN),
        "enc_W2": n((HIDDEN, HIDDEN),     scale=HIDDEN**-0.5),
        "enc_b2": jnp.zeros(HIDDEN),
        "enc_W3": n((HIDDEN, LATENT_DIM), scale=HIDDEN**-0.5),
        "enc_b3": jnp.zeros(LATENT_DIM),
        # Codebook: (K, D) — uniform in [-1/K, 1/K] per paper
        "codebook": u((CODEBOOK_SIZE, LATENT_DIM), lo=-1.0/CODEBOOK_SIZE, hi=1.0/CODEBOOK_SIZE),
        # Decoder: LATENT_DIM → HIDDEN → HIDDEN → 784
        "dec_W1": n((LATENT_DIM, HIDDEN), scale=LATENT_DIM**-0.5),
        "dec_b1": jnp.zeros(HIDDEN),
        "dec_W2": n((HIDDEN, HIDDEN),     scale=HIDDEN**-0.5),
        "dec_b2": jnp.zeros(HIDDEN),
        "dec_W3": n((HIDDEN, 784),        scale=HIDDEN**-0.5),
        "dec_b3": jnp.zeros(784),
    }

# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(params, x):
    """x (B, 784) in [0,1] → z (B, LATENT_DIM)."""
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    h = jax.nn.gelu(h @ params["enc_W2"] + params["enc_b2"])
    return               h @ params["enc_W3"] + params["enc_b3"]

# ── DiVeQ layer ───────────────────────────────────────────────────────────────

def diveq(params, z, key, training=True):
    """
    z:        (B, D) encoder output
    key:      PRNGKey for noise (only used when training=True)
    training: Python bool — traced at JIT-compile time, no overhead

    Returns:
        zq:      (B, D) quantized output
        indices: (B,)   nearest-codeword indices
        qe:      scalar quantization error (MSE, stop-gradiented, for logging)
    """
    C = params["codebook"]  # (K, D)

    # Nearest-codeword lookup via ‖z − c‖² = ‖z‖² + ‖c‖² − 2 z·cᵀ
    z_sq = (z ** 2).sum(-1, keepdims=True)   # (B, 1)
    c_sq = (C ** 2).sum(-1)                  # (K,)
    dists = z_sq + c_sq - 2.0 * (z @ C.T)   # (B, K)
    indices = jnp.argmin(dists, axis=-1)      # (B,)
    c_star  = C[indices]                      # (B, D)

    qe = ((jax.lax.stop_gradient(z) - jax.lax.stop_gradient(c_star)) ** 2).mean()

    if training:
        # d̃ = sg[c*] − sg[z]  (both stop-gradiented; whole direction is sg'd anyway)
        d_tilde = jax.lax.stop_gradient(c_star) - jax.lax.stop_gradient(z)  # (B, D)

        v  = jax.random.normal(key, z.shape) * SIGMA   # (B, D)
        vd = v + d_tilde                                # (B, D)

        # Unit direction — fully stop-gradiented (the sg[·] from the paper)
        norm_vd   = jnp.linalg.norm(vd, axis=-1, keepdims=True)
        direction = jax.lax.stop_gradient(vd / (norm_vd + 1e-8))  # (B, D)

        # Magnitude carries ALL gradient (w.r.t. z and c*)
        magnitude = jnp.linalg.norm(c_star - z, axis=-1, keepdims=True)  # (B, 1)

        zq = z + magnitude * direction
    else:
        # Inference: hard quantization, no noise
        zq = c_star

    return zq, indices, qe

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, zq):
    """zq (B, LATENT_DIM) → recon (B, 784) in (0, 1)."""
    h = jax.nn.gelu(zq @ params["dec_W1"] + params["dec_b1"])
    h = jax.nn.gelu(h  @ params["dec_W2"] + params["dec_b2"])
    return jax.nn.sigmoid(h @ params["dec_W3"] + params["dec_b3"])

# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(params, x_batch, key):
    """x_batch (B, 784) in [0, 255] → (recon_loss, (recon_loss, qe))."""
    x = x_batch / 255.0
    z  = encode(params, x)
    zq, _, qe = diveq(params, z, key, training=True)
    x_recon = decode(params, zq)
    recon = ((x_recon - x) ** 2).mean()
    return recon, (recon, qe)

# ── Epoch fn ──────────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    @jax.jit
    def epoch_fn(params, opt_state, Xs_batched, epoch_key):
        """Xs_batched: (num_batches, BATCH_SIZE, 784)"""
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
    zq, indices, qe = diveq(params, z, dummy_key, training=False)
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
    n_used = len(np.unique(np.concatenate(all_indices)))
    return float(np.mean(mses)), float(np.mean(qes)), n_used

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
            eval_str = f"  eval_mse={eval_mse:.5f}  codebook={n_used}/{CODEBOOK_SIZE}"

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
    logging.info(
        f"Parameters: {n_params(params):,}  "
        f"LATENT={LATENT_DIM}  K={CODEBOOK_SIZE}  sigma={SIGMA}  HIDDEN={HIDDEN}"
    )

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse, final_qe, final_n_used = eval_metrics(params, X_test)
    append_result({
        "experiment":      EXP_NAME,
        "name":            f"DiVeQ MLP LATENT={LATENT_DIM} K={CODEBOOK_SIZE} sigma={SIGMA}",
        "n_params":        n_params(params),
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "final_recon_mse": round(float(final_mse), 6),
        "final_qe":        round(float(final_qe), 6),
        "final_codebook_use": final_n_used,
        "loss_curve":      [round(v, 6) for v in history["loss"]],
        "recon_curve":     [round(v, 6) for v in history["recon"]],
        "qe_curve":        [round(v, 6) for v in history["qe"]],
        "eval_mse_curve":  history["eval_mse"],
        "eval_qe_curve":   history["eval_qe"],
        "codebook_use":    history["codebook_use"],
        "latent_dim":      LATENT_DIM,
        "codebook_size":   CODEBOOK_SIZE,
        "sigma":           SIGMA,
        "hidden":          HIDDEN,
        "adam_lr":         ADAM_LR,
        "batch_size":      BATCH_SIZE,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  qe={final_qe:.5f}  codebook_use={final_n_used}/{CODEBOOK_SIZE}  {elapsed:.1f}s")
