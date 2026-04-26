"""
exp14_abl — ablation: exp14 architecture trained with reconstruction loss only.

Identical to exp14 (deconvnet decoder, LATENT_DIM=64, BATCH_SIZE=128) but with
the kNN matching (variation) loss removed. Trains as a pure autoencoder.

No kNN precomputation needed. Training step is just: encode → decode with
recon_emb → MSE loss. Cat embeddings are present in the params but receive no
gradient (no variation path).

Purpose: isolate the contribution of the kNN matching loss to embedding quality.
Compare embed-eval metrics against exp14 to see the delta.

Usage:
    uv run python projects/variations/experiments14_abl.py
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
from lib.logging_utils import setup_logging

EXP_NAME = "exp14_abl"

# ── Hyperparameters (identical to exp14) ──────────────────────────────────────

LATENT_DIM  = 64
N_CAT       = 4
VOCAB_SIZE  = 8
E_DIM       = 4
COND_DIM    = N_CAT * E_DIM   # 16
IN_DIM      = LATENT_DIM + COND_DIM  # 80
BATCH_SIZE  = 128
MAX_EPOCHS  = 100
ADAM_LR     = 3e-4
SEED        = 42

DC_STEM     = 256
DC_C1       = 64
DC_C2       = 32

EVAL_EVERY  = 5

JSONL = Path(__file__).parent / "results.jsonl"

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Parameters (same layout as exp14) ─────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "cat_emb":   n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        "recon_emb": n((COND_DIM,),                scale=0.1),
        "enc_W1": n((784, 512),        scale=784**-0.5),
        "enc_b1": jnp.zeros(512),
        "enc_W2": n((512, LATENT_DIM), scale=512**-0.5),
        "enc_b2": jnp.zeros(LATENT_DIM),
        "dc_W1": n((IN_DIM,  DC_STEM),     scale=IN_DIM**-0.5),
        "dc_b1": jnp.zeros(DC_STEM),
        "dc_W2": n((DC_STEM, DC_C1 * 7*7), scale=DC_STEM**-0.5),
        "dc_b2": jnp.zeros(DC_C1 * 7*7),
        "dc_ct1": n((DC_C1, DC_C2, 4, 4), scale=(DC_C1 * 4 * 4)**-0.5),
        "dc_ct2": n((DC_C2,    1, 4, 4),  scale=(DC_C2 * 4 * 4)**-0.5),
    }

# ── Encoder (same as exp14) ────────────────────────────────────────────────────

def encode(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    return h @ params["enc_W2"] + params["enc_b2"]

# ── Decoder (same as exp14) ────────────────────────────────────────────────────

def decode(params, zc):
    B = zc.shape[0]
    h = jax.nn.gelu(zc @ params["dc_W1"] + params["dc_b1"])
    h = jax.nn.gelu(h  @ params["dc_W2"] + params["dc_b2"])
    h = h.reshape(B, DC_C1, 7, 7)
    h = jax.lax.conv_transpose(
        h, params["dc_ct1"],
        strides=(2, 2), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )
    h = jax.nn.gelu(h)
    h = jax.lax.conv_transpose(
        h, params["dc_ct2"],
        strides=(2, 2), padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )
    return jax.nn.sigmoid(h.reshape(B, 784))

# ── Loss: reconstruction only ─────────────────────────────────────────────────

def loss_fn(params, x_batch):
    B      = x_batch.shape[0]
    x_norm = x_batch / 255.0
    z      = encode(params, x_norm)
    zc     = jnp.concatenate([z, jnp.broadcast_to(params["recon_emb"], (B, COND_DIM))], axis=-1)
    x_recon = decode(params, zc)
    return ((x_recon - x_norm) ** 2).mean()

# ── Training ──────────────────────────────────────────────────────────────────

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
    x_norm  = x_batch / 255.0
    B       = x_batch.shape[0]
    z       = encode(params, x_norm)
    zc      = jnp.concatenate([z, jnp.broadcast_to(params["recon_emb"], (B, COND_DIM))], axis=-1)
    x_recon = decode(params, zc)
    return ((x_recon - x_norm) ** 2).mean()

def eval_recon_mse(params, X_te):
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
            mse = eval_recon_mse(params, X_te)
            history["eval_mse"].append((epoch, mse))
            eval_str = f"  eval_mse={mse:.5f}"

        logging.info(
            f"epoch {epoch:3d}  loss={float(loss):.5f}{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging(EXP_NAME)
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
        f"LATENT={LATENT_DIM}  decoder=deconvnet(C1={DC_C1},C2={DC_C2},stem={DC_STEM})"
        f"  ablation: recon-only (no kNN, no variation loss)"
    )

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_mse = eval_recon_mse(params, X_test)
    append_result({
        "experiment":      EXP_NAME,
        "name":            "exp14 ablation — recon loss only (no kNN variation)",
        "n_params":        n_params(params),
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "final_recon_mse": round(float(final_mse), 6),
        "loss_curve":      [round(v, 6) for v in history["loss"]],
        "eval_mse_curve":  history["eval_mse"],
        "latent_dim":      LATENT_DIM,
        "decoder":         "deconvnet",
        "variation_loss":  False,
        "batch_size":      BATCH_SIZE,
        "adam_lr":         ADAM_LR,
    })
    logging.info(f"Done. recon_mse={final_mse:.5f}  {elapsed:.1f}s")
