"""
exp34 — DiVeQ-PQ LATENT=32, G=16, D_g=2, K=512 + running-stats batch normalization.

exp33 showed that batch normalization gives perfect codebook utilization (8172/8192)
but very bad eval MSE (0.00721 vs 0.00392 training) due to train-eval mismatch:
training uses per-batch stats (N=128), eval uses per-batch stats (N=256) — same
input produces different z_bn depending on which samples are in the batch.

Fix: track population statistics (running mean/var) as non-gradient state, updated
once per epoch from a representative training sample. Eval uses population stats →
consistent z_bn regardless of batch composition.

Implementation:
  encode_train(params, x)         — batch norm (batch stats, for scan body)
  encode_eval(params, bn_state, x) — population norm (running stats, for eval)

bn_state = {"mean": (32,), "var": (32,)} updated outside JIT after each epoch.
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

EXP_NAME = "exp38"

# ── Hyperparameters ───────────────────────────────────────────────────────────

LATENT_DIM    = 32
N_GROUPS      = 16
GROUP_DIM     = LATENT_DIM // N_GROUPS   # = 2
CODEBOOK_SIZE = 256
SIGMA         = 0.01
HIDDEN        = 512
BATCH_SIZE    = 128
MAX_EPOCHS    = 200
ADAM_LR       = 3e-4
SEED          = 42
EVAL_EVERY    = 10
BN_MOMENTUM   = 0.1      # EMA momentum for running stats update

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
        lo=-1.0/256, hi=1.0/256
    )
    return params


def init_bn_state():
    return {
        "mean": jnp.zeros(LATENT_DIM),
        "var":  jnp.ones(LATENT_DIM),
    }

# ── Raw encoder (no BN) ───────────────────────────────────────────────────────

def _encode_raw(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    h = jax.nn.gelu(h @ params["enc_W2"] + params["enc_b2"])
    return               h @ params["enc_W3"] + params["enc_b3"]

# ── Training encoder: batch norm with current-batch stats ─────────────────────

def encode_train(params, x):
    z = _encode_raw(params, x)
    mean = z.mean(axis=0)
    var  = z.var(axis=0)
    z_hat = (z - mean) / (jnp.sqrt(var) + 1e-6)
    return z_hat * params["bn_gamma"] + params["bn_beta"]

# ── Eval encoder: population norm from running stats ─────────────────────────

def encode_eval(params, bn_state, x):
    z = _encode_raw(params, x)
    z_hat = (z - bn_state["mean"]) / (jnp.sqrt(bn_state["var"]) + 1e-6)
    return z_hat * params["bn_gamma"] + params["bn_beta"]

# ── Running stats update (called once per epoch outside JIT) ──────────────────

@jax.jit
def _compute_batch_stats(params, x_batch):
    z = _encode_raw(params, x_batch / 255.0)
    return z.mean(axis=0), z.var(axis=0)

def update_bn_state(params, bn_state, X_sample):
    """Update running mean/var with EMA using a representative sample."""
    mean, var = _compute_batch_stats(params, jnp.array(X_sample))
    new_mean = (1 - BN_MOMENTUM) * bn_state["mean"] + BN_MOMENTUM * mean
    new_var  = (1 - BN_MOMENTUM) * bn_state["var"]  + BN_MOMENTUM * var
    return {"mean": new_mean, "var": new_var}

# ── Decoder ───────────────────────────────────────────────────────────────────

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

# ── Loss (training encoder — batch BN) ───────────────────────────────────────

def loss_fn(params, x_batch, key):
    x = x_batch / 255.0
    z  = encode_train(params, x)
    zq, _, qe = diveq_pq(params, z, key, training=True)
    x_recon = decode(params, zq)
    recon = ((x_recon - x) ** 2).mean()
    return recon, (recon, qe)

# ── Epoch fn (scan body uses training encoder) ────────────────────────────────

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

# ── Evaluation (uses population BN stats) ────────────────────────────────────

@jax.jit
def _eval_batch(params, bn_state, x_batch):
    x = x_batch / 255.0
    z = encode_eval(params, bn_state, x)
    dummy_key = jnp.zeros(2, dtype=jnp.uint32)
    zq, indices, qe = diveq_pq(params, z, dummy_key, training=False)
    x_recon = decode(params, zq)
    recon_mse = ((x_recon - x) ** 2).mean()
    return recon_mse, indices, qe

def eval_metrics(params, bn_state, X_te):
    mses, qes, all_indices = [], [], []
    for i in range(0, len(X_te), 256):
        batch = jnp.array(X_te[i:i+256])
        mse, indices, qe = _eval_batch(params, bn_state, batch)
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
    bn_state    = init_bn_state()

    # Warm up running stats from first 2000 training samples
    for _ in range(20):
        bn_state = update_bn_state(params, bn_state, X_tr[:2000])

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

        # Update running stats after each epoch (2000 random training samples)
        sample_idx = np.random.choice(len(X_tr), 2000, replace=False)
        bn_state = update_bn_state(params, bn_state, X_tr[sample_idx])

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_mse, eval_qe, n_used = eval_metrics(params, bn_state, X_te)
            history["eval_mse"].append((epoch, eval_mse))
            history["eval_qe"].append((epoch, eval_qe))
            history["codebook_use"].append((epoch, n_used))
            eval_str = f"  eval_mse={eval_mse:.5f}  cb={n_used}"

        logging.info(
            f"epoch {epoch:3d}  loss={float(loss):.5f}"
            f"  recon={float(recon):.5f}  qe={float(qe):.5f}"
            f"{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

    return params, bn_state, history, time.perf_counter() - t0

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
        f"bits={total_bits}  sigma={SIGMA}  HIDDEN={HIDDEN}  (running-stats BN)"
    )

    params, bn_state, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(Path(__file__).parent / f"bn_state_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in bn_state.items()}, f)

    final_mse, final_qe, final_n_used = eval_metrics(params, bn_state, X_test)
    total_bits = N_GROUPS * int(np.log2(CODEBOOK_SIZE))
    append_result({
        "experiment":       EXP_NAME,
        "name":             f"DiVeQ-PQ+BN-running G={N_GROUPS} K={CODEBOOK_SIZE} D_g={GROUP_DIM} bits={total_bits}",
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
        "bottleneck":       "batch_norm_running_stats",
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  qe={final_qe:.5f}  cb_per_group={final_n_used}  {elapsed:.1f}s")
