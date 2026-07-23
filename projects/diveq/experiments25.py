"""
exp25 — Standard VQ-PQ baseline (straight-through estimator + commitment loss).

Same architecture as exp17 (G=8, D_g=2, K=512 per group) but uses the
classic VQ-VAE objective instead of DiVeQ:

    zq = z + (c* − z).detach()        # straight-through estimator
    loss = recon + α * codebook_loss + β * commitment_loss
         = MSE(x̂, x)
           + ‖sg[z] − c*‖²            # codebook loss (moves c* toward z)
           + β * ‖z − sg[c*]‖²        # commitment loss (moves z toward c*)

β = 0.25 (standard from original VQ-VAE paper).

Purpose: direct comparison with exp17/21 (DiVeQ-PQ) to isolate the benefit
of the DiVeQ gradient path vs STE + commitment loss.

Usage:
    uv run python projects/diveq/scripts/run_experiments.py exp24
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

EXP_NAME = "exp25"

# ── Hyperparameters ───────────────────────────────────────────────────────────

LATENT_DIM    = 16
N_GROUPS      = 8
GROUP_DIM     = LATENT_DIM // N_GROUPS   # = 2
CODEBOOK_SIZE = 512
BETA          = 0.25       # commitment loss weight (standard VQ-VAE value)
HIDDEN        = 512
BATCH_SIZE    = 128
MAX_EPOCHS    = 500
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
    params["codebooks"] = u(
        (N_GROUPS, CODEBOOK_SIZE, GROUP_DIM),
        lo=-1.0/CODEBOOK_SIZE, hi=1.0/CODEBOOK_SIZE
    )
    return params

# ── Encoder / Decoder ─────────────────────────────────────────────────────────

def encode(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    h = jax.nn.gelu(h @ params["enc_W2"] + params["enc_b2"])
    return               h @ params["enc_W3"] + params["enc_b3"]

def decode(params, zq):
    h = jax.nn.gelu(zq @ params["dec_W1"] + params["dec_b1"])
    h = jax.nn.gelu(h  @ params["dec_W2"] + params["dec_b2"])
    return jax.nn.sigmoid(h @ params["dec_W3"] + params["dec_b3"])

# ── Standard VQ (STE) per group ───────────────────────────────────────────────

def _vq_group(C, z_g):
    """Standard VQ: STE forward + codebook/commitment losses."""
    z_sq = (z_g ** 2).sum(-1, keepdims=True)
    c_sq = (C ** 2).sum(-1)
    dists = z_sq + c_sq - 2.0 * (z_g @ C.T)
    indices = jnp.argmin(dists, axis=-1)
    c_star = C[indices]

    # Straight-through estimator: zq = z + (c* - z).detach()
    zq_g = z_g + jax.lax.stop_gradient(c_star - z_g)

    # Codebook loss: ‖sg[z] − c*‖²  (moves c* toward z)
    codebook_loss = ((jax.lax.stop_gradient(z_g) - c_star) ** 2).mean()

    # Commitment loss: ‖z − sg[c*]‖²  (moves z toward c*)
    commitment_loss = ((z_g - jax.lax.stop_gradient(c_star)) ** 2).mean()

    qe = ((jax.lax.stop_gradient(z_g) - jax.lax.stop_gradient(c_star)) ** 2).mean()

    return zq_g, indices, codebook_loss, commitment_loss, qe

# ── Standard VQ-PQ layer ──────────────────────────────────────────────────────

def vq_pq(params, z):
    codebooks = params["codebooks"]
    zq_groups, idx_groups = [], []
    codebook_loss, commitment_loss, qe_total = 0.0, 0.0, 0.0

    for g in range(N_GROUPS):
        z_g = z[:, g*GROUP_DIM:(g+1)*GROUP_DIM]
        C_g = codebooks[g]
        zq_g, idx_g, cb_loss, cm_loss, qe = _vq_group(C_g, z_g)
        zq_groups.append(zq_g)
        idx_groups.append(idx_g)
        codebook_loss  += cb_loss
        commitment_loss += cm_loss
        qe_total       += qe

    zq      = jnp.concatenate(zq_groups, axis=-1)
    indices = jnp.stack(idx_groups, axis=-1)
    return zq, indices, codebook_loss / N_GROUPS, commitment_loss / N_GROUPS, qe_total / N_GROUPS

# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(params, x_batch):
    x = x_batch / 255.0
    z = encode(params, x)
    zq, _, cb_loss, cm_loss, qe = vq_pq(params, z)
    x_recon = decode(params, zq)
    recon = ((x_recon - x) ** 2).mean()
    total = recon + cb_loss + BETA * cm_loss
    return total, (recon, cb_loss, cm_loss, qe)

# ── Epoch fn ──────────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    @jax.jit
    def epoch_fn(params, opt_state, Xs_batched):
        def scan_body(carry, x_batch):
            params, opt_state = carry
            (loss, (recon, cb, cm, qe)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, x_batch
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), (loss, recon, cb, cm, qe)

        (params, opt_state), stats = jax.lax.scan(
            scan_body, (params, opt_state), Xs_batched
        )
        return params, opt_state, *[s.mean() for s in stats]

    return epoch_fn

# ── Evaluation ────────────────────────────────────────────────────────────────

@jax.jit
def _eval_batch(params, x_batch):
    x = x_batch / 255.0
    z = encode(params, x)
    zq, indices, _, _, qe = vq_pq(params, z)
    x_recon = decode(params, zq)
    return ((x_recon - x) ** 2).mean(), indices, qe

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

    history = {"loss": [], "recon": [], "cb": [], "cm": [], "qe": [], "eval_mse": [], "codebook_use": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        n    = num_batches * BATCH_SIZE
        Xs_batched = jnp.array(X_tr[perm[:n]].reshape(num_batches, BATCH_SIZE, 784))

        params, opt_state, loss, recon, cb, cm, qe = epoch_fn(params, opt_state, Xs_batched)
        history["loss"].append(float(loss))
        history["recon"].append(float(recon))
        history["cb"].append(float(cb))
        history["cm"].append(float(cm))
        history["qe"].append(float(qe))

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_mse, eval_qe, n_used = eval_metrics(params, X_te)
            history["eval_mse"].append((epoch, eval_mse))
            history["codebook_use"].append((epoch, n_used))
            eval_str = f"  eval_mse={eval_mse:.5f}  cb={n_used}"

        logging.info(
            f"epoch {epoch:3d}  loss={float(loss):.5f}"
            f"  recon={float(recon):.5f}  cb={float(cb):.5f}  cm={float(cm):.5f}  qe={float(qe):.5f}"
            f"{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

    return params, history, time.perf_counter() - t0

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
        f"bits={total_bits}  BETA={BETA}  HIDDEN={HIDDEN}  (STE + commitment loss)"
    )

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse, final_qe, final_n_used = eval_metrics(params, X_test)
    append_result({
        "experiment":       EXP_NAME,
        "name":             f"VQ-PQ (STE) G={N_GROUPS} K={CODEBOOK_SIZE} D_g={GROUP_DIM} β={BETA} bits={total_bits}",
        "n_params":         n_params(params),
        "epochs":           MAX_EPOCHS,
        "time_s":           round(elapsed, 1),
        "final_recon_mse":  round(float(final_mse), 6),
        "final_qe":         round(float(final_qe), 6),
        "final_codebook_use": sum(final_n_used),
        "final_codebook_per_group": final_n_used,
        "n_groups":         N_GROUPS,
        "group_dim":        GROUP_DIM,
        "codebook_size":    CODEBOOK_SIZE,
        "total_bits":       total_bits,
        "beta":             BETA,
        "latent_dim":       LATENT_DIM,
        "hidden":           HIDDEN,
        "adam_lr":          ADAM_LR,
        "batch_size":       BATCH_SIZE,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  qe={final_qe:.5f}  cb_per_group={final_n_used}  {elapsed:.1f}s")
