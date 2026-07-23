"""
exp54–exp61 — Low-bit Pareto extension for MNIST DiVeQ-PQ.

Pushes below the 64-bit floor by extending two LATENT ladders:

  LATENT=16, G=8, D_g=2, no BN:
    exp54: K=128  → 56 bits
    exp55: K=64   → 48 bits
    exp56: K=32   → 40 bits
    exp57: K=16   → 32 bits

  LATENT=8, G=4, D_g=2, no BN:
    exp58: AE ceiling  (no quantization, K=None)
    exp59: K=512  → 36 bits
    exp60: K=64   → 24 bits
    exp61: K=16   → 16 bits

All 500ep, HIDDEN=512, same DiVeQ-PQ mechanism as exp21/53.
BN omitted: LATENT≤16 uses all dims naturally (no lazy-group problem).
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

JSONL      = Path(__file__).parent / "results.jsonl"
HIDDEN     = 512
BATCH_SIZE = 128
MAX_EPOCHS = 500
ADAM_LR    = 3e-4
SEED       = 42
EVAL_EVERY = 10
SIGMA      = 0.01

# (exp_name, latent_dim, n_groups, codebook_size_or_None_for_AE)
CONFIGS = [
    ("exp54", 16, 8, 128),
    ("exp55", 16, 8,  64),
    ("exp56", 16, 8,  32),
    ("exp57", 16, 8,  16),
    ("exp58",  8, 4, None),   # AE ceiling for LATENT=8
    ("exp59",  8, 4, 512),
    ("exp60",  8, 4,  64),
    ("exp61",  8, 4,  16),
]


def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")


def done_set():
    s = set()
    if JSONL.exists():
        for line in JSONL.read_text().strip().splitlines():
            try: s.add(json.loads(line).get("experiment"))
            except Exception: pass
    return s


# ── Parametric model ──────────────────────────────────────────────────────────

def init_params(key_gen, latent_dim, n_groups, codebook_size):
    group_dim = latent_dim // n_groups
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    def u(shape, lo, hi):
        return jax.random.uniform(next(key_gen).get(), shape, minval=lo, maxval=hi)
    params = {
        "enc_W1": n((784,       HIDDEN),     scale=784**-0.5),
        "enc_b1": jnp.zeros(HIDDEN),
        "enc_W2": n((HIDDEN,    HIDDEN),     scale=HIDDEN**-0.5),
        "enc_b2": jnp.zeros(HIDDEN),
        "enc_W3": n((HIDDEN,    latent_dim), scale=HIDDEN**-0.5),
        "enc_b3": jnp.zeros(latent_dim),
        "dec_W1": n((latent_dim, HIDDEN),   scale=latent_dim**-0.5),
        "dec_b1": jnp.zeros(HIDDEN),
        "dec_W2": n((HIDDEN,    HIDDEN),    scale=HIDDEN**-0.5),
        "dec_b2": jnp.zeros(HIDDEN),
        "dec_W3": n((HIDDEN,    784),       scale=HIDDEN**-0.5),
        "dec_b3": jnp.zeros(784),
    }
    if codebook_size is not None:
        params["codebooks"] = u(
            (n_groups, codebook_size, group_dim),
            lo=-1.0/codebook_size, hi=1.0/codebook_size
        )
    return params


def encode(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    h = jax.nn.gelu(h @ params["enc_W2"] + params["enc_b2"])
    return               h @ params["enc_W3"] + params["enc_b3"]


def decode(params, zq):
    h = jax.nn.gelu(zq @ params["dec_W1"] + params["dec_b1"])
    h = jax.nn.gelu(h  @ params["dec_W2"] + params["dec_b2"])
    return jax.nn.sigmoid(h @ params["dec_W3"] + params["dec_b3"])


def _diveq_group(C, z_g, key):
    z_sq = (z_g ** 2).sum(-1, keepdims=True)
    c_sq = (C ** 2).sum(-1)
    dists = z_sq + c_sq - 2.0 * (z_g @ C.T)
    indices = jnp.argmin(dists, axis=-1)
    c_star = C[indices]
    qe = ((jax.lax.stop_gradient(z_g) - jax.lax.stop_gradient(c_star)) ** 2).mean()
    d_tilde   = jax.lax.stop_gradient(c_star) - jax.lax.stop_gradient(z_g)
    v         = jax.random.normal(key, z_g.shape) * SIGMA
    vd        = v + d_tilde
    norm_vd   = jnp.linalg.norm(vd, axis=-1, keepdims=True)
    direction = jax.lax.stop_gradient(vd / (norm_vd + 1e-8))
    magnitude = jnp.linalg.norm(c_star - z_g, axis=-1, keepdims=True)
    zq_g = z_g + magnitude * direction
    return zq_g, indices, qe


def diveq_pq_train(params, z, key, n_groups, group_dim):
    codebooks = params["codebooks"]
    zq_groups, idx_groups, qe_groups = [], [], []
    for g in range(n_groups):
        z_g   = z[:, g*group_dim:(g+1)*group_dim]
        C_g   = codebooks[g]
        key_g = jax.random.fold_in(key, g)
        zq_g, idx_g, qe_g = _diveq_group(C_g, z_g, key_g)
        zq_groups.append(zq_g); idx_groups.append(idx_g); qe_groups.append(qe_g)
    return (jnp.concatenate(zq_groups, axis=-1),
            jnp.stack(idx_groups, axis=-1),
            sum(qe_groups) / n_groups)


def diveq_pq_eval(params, z, n_groups, group_dim):
    codebooks = params["codebooks"]
    zq_groups, idx_groups = [], []
    for g in range(n_groups):
        z_g = z[:, g*group_dim:(g+1)*group_dim]
        C_g = codebooks[g]
        dists = (z_g**2).sum(-1, keepdims=True) + (C_g**2).sum(-1) - 2.0*(z_g@C_g.T)
        idx = jnp.argmin(dists, axis=-1)
        zq_groups.append(C_g[idx]); idx_groups.append(idx)
    return jnp.concatenate(zq_groups, axis=-1), jnp.stack(idx_groups, axis=-1)


# ── Loss and epoch fn (closed over config) ────────────────────────────────────

def make_loss_fn(n_groups, group_dim, is_ae):
    def loss_fn(params, x_batch, key):
        x = x_batch / 255.0
        z = encode(params, x)
        if is_ae:
            zq, qe = z, jnp.array(0.0)
        else:
            zq, _, qe = diveq_pq_train(params, z, key, n_groups, group_dim)
        recon = ((decode(params, zq) - x) ** 2).mean()
        return recon, (recon, qe)
    return loss_fn


def make_epoch_fn(optimizer, loss_fn):
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


def eval_metrics(params, X_te, n_groups, group_dim, is_ae):
    mses, all_indices = [], []
    for i in range(0, len(X_te), 256):
        x = jnp.array(X_te[i:i+256]) / 255.0
        z = encode(params, x)
        if is_ae:
            zq = z
            idx_np = None
        else:
            zq, idx = diveq_pq_eval(params, z, n_groups, group_dim)
            idx_np = np.array(idx)
        xr = decode(params, zq)
        mses.append(float(((xr - x)**2).mean()))
        if idx_np is not None:
            all_indices.append(idx_np)
    mse = float(np.mean(mses))
    if is_ae:
        return mse, []
    all_idx = np.concatenate(all_indices, axis=0)
    n_used = [len(np.unique(all_idx[:, g])) for g in range(n_groups)]
    return mse, n_used


def run_one(exp_name, latent_dim, n_groups, codebook_size, X_train, X_test):
    is_ae     = codebook_size is None
    group_dim = latent_dim // n_groups
    total_bits = 0 if is_ae else n_groups * int(np.log2(codebook_size))

    logging.info(f"\n{'='*60}")
    logging.info(f"{exp_name}: LATENT={latent_dim}  G={n_groups}  D_g={group_dim}  "
                 f"K={codebook_size}  bits={total_bits}  is_ae={is_ae}")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen, latent_dim, n_groups, codebook_size)
    logging.info(f"Parameters: {n_params(params):,}")

    num_batches = len(X_train) // BATCH_SIZE
    num_steps   = MAX_EPOCHS * num_batches
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    loss_fn     = make_loss_fn(n_groups, group_dim, is_ae)
    epoch_fn    = make_epoch_fn(optimizer, loss_fn)
    key_gen2    = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    history = {"loss": [], "recon": [], "qe": [], "eval_mse": [], "codebook_use": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen2).get()
        perm = np.array(jax.random.permutation(key, len(X_train)))
        n    = num_batches * BATCH_SIZE
        Xs   = jnp.array(X_train[perm[:n]].reshape(num_batches, BATCH_SIZE, 784))

        params, opt_state, loss, recon, qe = epoch_fn(params, opt_state, Xs, key)
        history["loss"].append(float(loss))
        history["recon"].append(float(recon))
        history["qe"].append(float(qe))

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_mse, n_used = eval_metrics(params, X_test, n_groups, group_dim, is_ae)
            history["eval_mse"].append((epoch, eval_mse))
            history["codebook_use"].append((epoch, n_used))
            cb_str = f"  cb_min={min(n_used)}" if n_used else ""
            eval_str = f"  eval_mse={eval_mse:.5f}{cb_str}"

        logging.info(
            f"epoch {epoch:3d}  loss={float(loss):.5f}"
            f"  recon={float(recon):.5f}  qe={float(qe):.5f}"
            f"{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

    elapsed = time.perf_counter() - t0
    final_mse, final_n_used = eval_metrics(params, X_test, n_groups, group_dim, is_ae)

    PROJECT = Path(__file__).parent
    with open(PROJECT / f"params_{exp_name}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(PROJECT / f"history_{exp_name}.pkl", "wb") as f:
        pickle.dump(history, f)

    row = {
        "experiment":              exp_name,
        "name":                    (f"AE L={latent_dim}" if is_ae
                                    else f"DiVeQ-PQ G={n_groups} K={codebook_size} D_g={group_dim} bits={total_bits}"),
        "final_recon_mse":         round(float(final_mse), 6),
        "final_qe":                0.0 if is_ae else round(float(history["qe"][-1]), 6),
        "n_groups":                n_groups,
        "group_dim":               group_dim,
        "codebook_size":           codebook_size,
        "total_bits":              total_bits,
        "latent_dim":              latent_dim,
        "epochs":                  MAX_EPOCHS,
        "time_s":                  round(elapsed, 1),
        "bottleneck":              "none",
        "n_params":                n_params(params),
    }
    if final_n_used:
        row["final_codebook_use"]       = sum(final_n_used)
        row["final_codebook_per_group"] = final_n_used
    append_result(row)
    cb_str = f"  cb={final_n_used}" if final_n_used else ""
    logging.info(f"Done {exp_name}. recon_mse={final_mse:.5f}{cb_str}  {elapsed:.1f}s")


if __name__ == "__main__":
    done = done_set()
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    logging.info(f"Loaded MNIST: {len(X_train)} train / {len(X_test)} test")

    for exp_name, latent_dim, n_groups, codebook_size in CONFIGS:
        if exp_name in done:
            logging.info(f"{exp_name} already done — skipping")
            continue
        run_one(exp_name, latent_dim, n_groups, codebook_size, X_train, X_test)

    logging.info("\nAll experiments complete.")
