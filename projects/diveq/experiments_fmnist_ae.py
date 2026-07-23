"""
fmnist_ae16 / fmnist_ae32 — AE ceilings for Fashion-MNIST.

Establishes the no-quantization baseline for LATENT=16 and LATENT=32.
These set the target MSE that DiVeQ-PQ needs to approach.

Run with:  uv run python projects/diveq/experiments_fmnist_ae.py
"""

import json, time, sys
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

JSONL = Path(__file__).parent / "results_fmnist.jsonl"

HIDDEN      = 512
BATCH_SIZE  = 128
MAX_EPOCHS  = 300
ADAM_LR     = 3e-4
SEED        = 42
EVAL_EVERY  = 10


def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

def init_params(key_gen, latent_dim):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "enc_W1": n((784,       HIDDEN),     scale=784**-0.5),
        "enc_b1": jnp.zeros(HIDDEN),
        "enc_W2": n((HIDDEN,    HIDDEN),     scale=HIDDEN**-0.5),
        "enc_b2": jnp.zeros(HIDDEN),
        "enc_W3": n((HIDDEN,    latent_dim), scale=HIDDEN**-0.5),
        "enc_b3": jnp.zeros(latent_dim),
        "dec_W1": n((latent_dim, HIDDEN),    scale=latent_dim**-0.5),
        "dec_b1": jnp.zeros(HIDDEN),
        "dec_W2": n((HIDDEN,    HIDDEN),     scale=HIDDEN**-0.5),
        "dec_b2": jnp.zeros(HIDDEN),
        "dec_W3": n((HIDDEN,    784),        scale=HIDDEN**-0.5),
        "dec_b3": jnp.zeros(784),
    }

def encode(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    h = jax.nn.gelu(h @ params["enc_W2"] + params["enc_b2"])
    return               h @ params["enc_W3"] + params["enc_b3"]

def decode(params, z):
    h = jax.nn.gelu(z @ params["dec_W1"] + params["dec_b1"])
    h = jax.nn.gelu(h @ params["dec_W2"] + params["dec_b2"])
    return jax.nn.sigmoid(h @ params["dec_W3"] + params["dec_b3"])

def loss_fn(params, x_batch):
    x = x_batch / 255.0
    return ((decode(params, encode(params, x)) - x) ** 2).mean()

def make_epoch_fn(optimizer):
    @jax.jit
    def epoch_fn(params, opt_state, Xs_batched):
        def scan_body(carry, x_batch):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(params, x_batch)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), loss
        (params, opt_state), losses = jax.lax.scan(scan_body, (params, opt_state), Xs_batched)
        return params, opt_state, losses.mean()
    return epoch_fn

@jax.jit
def _eval_batch(params, x_batch):
    x = x_batch / 255.0
    return ((decode(params, encode(params, x)) - x) ** 2).mean()

def eval_mse(params, X_te):
    return float(np.mean([float(_eval_batch(params, jnp.array(X_te[i:i+256])))
                          for i in range(0, len(X_te), 256)]))

def run_ae(exp_name, latent_dim, X_train, X_test):
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().strip().splitlines():
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
    if exp_name in done:
        logging.info(f"{exp_name} already done — skipping")
        return

    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params    = init_params(key_gen, latent_dim)
    num_batches = len(X_train) // BATCH_SIZE
    num_steps   = MAX_EPOCHS * num_batches
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen2    = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    t0 = time.perf_counter()

    logging.info(f"{exp_name}: LATENT={latent_dim}  params={n_params(params):,}")
    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen2).get()
        perm = np.array(jax.random.permutation(key, len(X_train)))
        n    = num_batches * BATCH_SIZE
        Xs   = jnp.array(X_train[perm[:n]].reshape(num_batches, BATCH_SIZE, 784))
        params, opt_state, loss = epoch_fn(params, opt_state, Xs)
        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            mse = eval_mse(params, X_test)
            eval_str = f"  eval_mse={mse:.5f}"
        logging.info(f"epoch {epoch:3d}  loss={float(loss):.5f}{eval_str}  {time.perf_counter()-t0:.1f}s")

    final_mse = eval_mse(params, X_test)
    elapsed   = time.perf_counter() - t0
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s")
    append_result({
        "experiment":      exp_name,
        "name":            f"FMNIST AE baseline LATENT={latent_dim} HIDDEN={HIDDEN}",
        "final_recon_mse": round(float(final_mse), 6),
        "final_qe":        0.0,
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "latent_dim":      latent_dim,
        "hidden":          HIDDEN,
        "dataset":         "fashion_mnist",
    })


if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("fashion_mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    logging.info(f"Loaded Fashion-MNIST: {len(X_train)} train, {len(X_test)} test")

    run_ae("fmnist_ae16", 16, X_train, X_test)
    run_ae("fmnist_ae32", 32, X_train, X_test)
