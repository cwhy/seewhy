"""
exp20 — Continuous MLP autoencoder baseline (no quantization).

Same encoder/decoder as the DiVeQ experiments but without the VQ layer.
Shows the architecture ceiling: how low can MSE go with LATENT=16 and HIDDEN=512?
Comparing against this tells us how much reconstruction quality is lost to quantization.

Usage:
    uv run python projects/diveq/scripts/run_experiments.py exp20
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

EXP_NAME = "exp20"

# ── Hyperparameters ───────────────────────────────────────────────────────────

LATENT_DIM    = 16
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
    return {
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
    z = encode(params, x)
    x_recon = decode(params, z)
    return ((x_recon - x) ** 2).mean()

def make_epoch_fn(optimizer):
    @jax.jit
    def epoch_fn(params, opt_state, Xs_batched):
        def scan_body(carry, x_batch):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(params, x_batch)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            scan_body, (params, opt_state), Xs_batched
        )
        return params, opt_state, losses.mean()

    return epoch_fn

@jax.jit
def _eval_batch(params, x_batch):
    x = x_batch / 255.0
    z = encode(params, x)
    x_recon = decode(params, z)
    return ((x_recon - x) ** 2).mean()

def eval_mse(params, X_te):
    mses = [float(_eval_batch(params, jnp.array(X_te[i:i+256])))
            for i in range(0, len(X_te), 256)]
    return float(np.mean(mses))

def train(params, X_tr, X_te):
    num_batches = len(X_tr) // BATCH_SIZE
    num_steps   = MAX_EPOCHS * num_batches
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    history = {"loss": [], "eval_mse": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        n    = num_batches * BATCH_SIZE
        Xs_batched = jnp.array(X_tr[perm[:n]].reshape(num_batches, BATCH_SIZE, 784))

        params, opt_state, loss = epoch_fn(params, opt_state, Xs_batched)
        history["loss"].append(float(loss))

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            mse = eval_mse(params, X_te)
            history["eval_mse"].append((epoch, mse))
            eval_str = f"  eval_mse={mse:.5f}"

        logging.info(f"epoch {epoch:3d}  loss={float(loss):.5f}{eval_str}  {time.perf_counter()-t0:.1f}s")

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
    logging.info(f"Parameters: {n_params(params):,}  LATENT={LATENT_DIM}  HIDDEN={HIDDEN}  (no quantization baseline)")

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_mse(params, X_test)
    append_result({
        "experiment":      EXP_NAME,
        "name":            f"AE baseline LATENT={LATENT_DIM} HIDDEN={HIDDEN} (no quantization)",
        "n_params":        n_params(params),
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "final_recon_mse": round(float(final_mse), 6),
        "final_qe":        0.0,
        "final_codebook_use": 0,
        "loss_curve":      [round(v, 6) for v in history["loss"]],
        "eval_mse_curve":  history["eval_mse"],
        "latent_dim":      LATENT_DIM,
        "hidden":          HIDDEN,
        "adam_lr":         ADAM_LR,
        "batch_size":      BATCH_SIZE,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s  (no quantization ceiling)")
