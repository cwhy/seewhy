"""
Generate AE, DAE, and VAE experiment files for all 9 architectures.

Each file is a self-contained experiment with:
  - same encoder architecture as the SIGReg/EMA experiments
  - shared 2-layer MLP decoder (LATENT_DIM -> 512 -> 784)
  - AE: standard autoencoder, MSE in [0,1] space
  - DAE: denoising autoencoder, same 4 augmentations as SIGReg/EMA
  - VAE: variational autoencoder, beta-VAE with BETA=0.001

Outputs (written to projects/ssl/):
    exp_ae_{mlp,mlp2,conv,conv2,vit,vit_patch,gram_dual,gram_single,gram_glu}.py
    exp_dae_{mlp,mlp2,conv,conv2,vit,vit_patch,gram_dual,gram_single,gram_glu}.py
    exp_vae_{mlp,mlp2,conv,conv2,vit,vit_patch,gram_dual,gram_single,gram_glu}.py

Usage:
    uv run python projects/ssl/scripts/tmp/mk_exp_ae_dae_vae.py
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent

# ── Shared header ─────────────────────────────────────────────────────────────

HEADER = '''\
"""
{exp_name} — {arch_label} encoder + {training_upper}, LATENT_DIM=1024.

Training: {training_desc}

Usage:
    uv run python projects/ssl/{exp_name}.py
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

EXP_NAME = "{exp_name}"

# ── Hyperparameters ────────────────────────────────────────────────────────────

BATCH_SIZE   = 256
SEED         = 42
MAX_EPOCHS   = 200
ADAM_LR      = 3e-4
EVAL_EVERY   = 10
PROBE_EPOCHS = 10

LATENT_DIM   = 1024
DEC_HIDDEN   = 512

NOISE_STD    = 75.0
MASK_RATIOS  = [0.25, 0.50, 0.75]

JSONL = Path(__file__).parent / "results.jsonl"

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\\n")

'''

# ── Decoder block (shared for all archs/types) ────────────────────────────────

DECODER_PARAMS = '''\
        # ── Decoder ───────────────────────────────────────────────────────────
        "W_dec1"      : n((LATENT_DIM, DEC_HIDDEN),   scale=LATENT_DIM**-0.5),
        "b_dec1"      : jnp.zeros(DEC_HIDDEN),
        "W_dec2"      : n((DEC_HIDDEN, 784),           scale=DEC_HIDDEN**-0.5),
        "b_dec2"      : jnp.zeros(784),'''

VAE_PARAMS = '''\
        # ── VAE moments projection ─────────────────────────────────────────────
        "W_vae"       : jnp.zeros((LATENT_DIM, 2 * LATENT_DIM)),
        "b_vae"       : jnp.zeros(2 * LATENT_DIM),'''

DECODER_FN = '''\

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, z):
    """z: (B, LATENT_DIM) -> (B, 784) in [0,1] space."""
    h = jax.nn.gelu(z @ params["W_dec1"] + params["b_dec1"])
    return h @ params["W_dec2"] + params["b_dec2"]

'''

# ── Patch mask helper ─────────────────────────────────────────────────────────

PATCH_MASK_FN = '''\
def _patch_mask(x, key, mask_ratio):
    """Zero entire 4x4 patches at given ratio.  x: (B, 784) -> (B, 784)."""
    B       = x.shape[0]
    patches = x.reshape(B, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(B, 49, 16)
    keep    = jax.random.bernoulli(key, 1.0 - mask_ratio, shape=(B, 49))
    patches = patches * keep[:, :, None]
    return patches.reshape(B, 7, 7, 4, 4).transpose(0, 1, 3, 2, 4).reshape(B, 784)

'''

# ── Loss functions ─────────────────────────────────────────────────────────────

AE_LOSS = '''\
# ── Loss ──────────────────────────────────────────────────────────────────────

def total_loss(params, x, _key):
    """x: (B, 784).  Returns scalar MSE loss."""
    x_norm = x / 255.0
    recon  = decode(params, encode(params, x))
    return ((recon - x_norm) ** 2).mean()

'''

DAE_LOSS_PIXEL = '''\
# ── Loss ──────────────────────────────────────────────────────────────────────

def total_loss(params, x, key):
    """x: (B, 784).  Returns (loss, None).  One random augmentation per sample."""
    k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)
    B   = x.shape[0]
    idx = jax.random.randint(k_idx, (B,), 0, 4)
    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))
    recon = decode(params, encode(params, x_aug))
    return ((recon - x / 255.0) ** 2).mean(), None

'''

DAE_LOSS_PATCH = '''\
# ── Loss ──────────────────────────────────────────────────────────────────────

def total_loss(params, x, key):
    """x: (B, 784).  Returns (loss, None).  One random augmentation per sample (patch masking)."""
    k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)
    B   = x.shape[0]
    idx = jax.random.randint(k_idx, (B,), 0, 4)
    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
    x_m25 = _patch_mask(x, k_m1, MASK_RATIOS[0])
    x_m50 = _patch_mask(x, k_m2, MASK_RATIOS[1])
    x_m75 = _patch_mask(x, k_m3, MASK_RATIOS[2])
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))
    recon = decode(params, encode(params, x_aug))
    return ((recon - x / 255.0) ** 2).mean(), None

'''

VAE_LOSS = '''\
# ── VAE encode ────────────────────────────────────────────────────────────────

BETA = 0.001

def encode_base(params, x):
    """Encoder without VAE projection.  x: (B, 784) -> (B, LATENT_DIM)."""
    return encode(params, x)

# ── Loss ──────────────────────────────────────────────────────────────────────

def total_loss(params, x, key):
    """x: (B, 784).  Returns (loss, (recon_loss, kl_loss))."""
    x_norm  = x / 255.0
    h       = encode_base(params, x)
    moments = h @ params["W_vae"] + params["b_vae"]
    mu, lv  = moments[:, :LATENT_DIM], moments[:, LATENT_DIM:]
    lv      = jnp.clip(lv, -10.0, 10.0)
    eps     = jax.random.normal(key, mu.shape)
    z       = mu + eps * jnp.exp(0.5 * lv)
    recon_loss = ((decode(params, z) - x_norm) ** 2).mean()
    kl_loss    = -0.5 * (1 + lv - mu ** 2 - jnp.exp(lv)).mean()
    loss       = recon_loss + BETA * kl_loss
    return loss, (recon_loss, kl_loss)

'''

# ── Epoch functions ────────────────────────────────────────────────────────────

AE_EPOCH_FN = '''\
# ── Epoch function ────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    def scan_body(carry, inputs):
        params, opt_state = carry
        x_batch, key = inputs
        loss, grads = jax.value_and_grad(lambda p: total_loss(p, x_batch, key))(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), loss

    def epoch_fn(params, opt_state, Xs_batched, epoch_key):
        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (params, opt_state), losses = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_batched, keys)
        )
        return params, opt_state, losses.mean()

    return jax.jit(epoch_fn)

'''

DAE_EPOCH_FN = '''\
# ── Epoch function ────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    def scan_body(carry, inputs):
        params, opt_state = carry
        x_batch, key = inputs
        (loss, _), grads = jax.value_and_grad(total_loss, has_aux=True)(params, x_batch, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), loss

    def epoch_fn(params, opt_state, Xs_batched, epoch_key):
        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (params, opt_state), losses = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_batched, keys)
        )
        return params, opt_state, losses.mean()

    return jax.jit(epoch_fn)

'''

VAE_EPOCH_FN = '''\
# ── Epoch function ────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    def scan_body(carry, inputs):
        params, opt_state = carry
        x_batch, key = inputs
        (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(params, x_batch, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), (loss, *aux)

    def epoch_fn(params, opt_state, Xs_batched, epoch_key):
        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (params, opt_state), stats = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_batched, keys)
        )
        return params, opt_state, *[s.mean() for s in stats]

    return jax.jit(epoch_fn)

'''

# ── Linear probe (shared) ─────────────────────────────────────────────────────

LINEAR_PROBE = '''\
# ── Linear probe ──────────────────────────────────────────────────────────────

def linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te):
    def encode_all(X):
        parts = []
        for i in range(0, len(X), 512):
            parts.append(np.array(encode(params, jnp.array(np.array(X[i:i+512])))))
        return np.concatenate(parts, 0)

    E_tr = encode_all(X_tr)
    E_te = encode_all(X_te)
    n_tr = (len(E_tr) // BATCH_SIZE) * BATCH_SIZE
    E_b  = jnp.array(E_tr[:n_tr].reshape(-1, BATCH_SIZE, LATENT_DIM))
    Y_b  = jnp.array(np.array(Y_tr[:n_tr]).reshape(-1, BATCH_SIZE))

    W, b        = jnp.zeros((LATENT_DIM, 10)), jnp.zeros(10)
    probe_opt   = optax.adam(1e-3)
    probe_state = probe_opt.init((W, b))

    def probe_step(carry, inputs):
        (W, b), state = carry
        e, y = inputs
        def ce(wb): return optax.softmax_cross_entropy_with_integer_labels(e @ wb[0] + wb[1], y).mean()
        loss, grads = jax.value_and_grad(ce)((W, b))
        updates, new_state = probe_opt.update(grads, state, (W, b))
        return (optax.apply_updates((W, b), updates), new_state), loss

    @jax.jit
    def probe_epoch(W, b, state):
        (W, b), state = jax.lax.scan(probe_step, ((W, b), state), (E_b, Y_b))[0]
        return W, b, state

    for _ in range(PROBE_EPOCHS):
        W, b, probe_state = probe_epoch(W, b, probe_state)

    logits = jnp.array(E_te) @ W + b
    return float((jnp.argmax(logits, 1) == jnp.array(np.array(Y_te))).mean())

'''

# ── Training loops ────────────────────────────────────────────────────────────

AE_TRAIN = '''\
# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, Y_tr, X_te, Y_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {{"loss": [], "probe_acc": []}}
    t0          = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(np.array(X_tr)[perm][:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784))
        params, opt_state, loss = epoch_fn(params, opt_state, Xs, jax.random.fold_in(key, epoch))
        loss = float(loss)
        history["loss"].append(loss)

        probe_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te)
            history["probe_acc"].append((epoch, acc))
            probe_str = f"  probe_acc={{acc:.1%}}"

        logging.info(f"epoch {{epoch:3d}}  loss={{loss:.4f}}{{probe_str}}  {{time.perf_counter()-t0:.1f}}s")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {{jax.devices()}}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_train = np.array(data.y)
    Y_test  = np.array(data.y_test)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {{n_params(params):,}}")

    params, history, elapsed = train(params, X_train, Y_train, X_test, Y_test)

    with open(Path(__file__).parent / f"params_{{EXP_NAME}}.pkl", "wb") as f:
        pickle.dump({{k: np.array(v) for k, v in params.items()}}, f)
    with open(Path(__file__).parent / f"history_{{EXP_NAME}}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_acc = linear_probe_accuracy(params, X_train, Y_train, X_test, Y_test)
    append_result({{
        "experiment"      : EXP_NAME,
        "n_params"        : n_params(params),
        "epochs"          : MAX_EPOCHS,
        "time_s"          : round(elapsed, 1),
        "final_probe_acc" : round(final_acc, 6),
        "loss_curve"      : [round(v, 6) for v in history["loss"]],
        "probe_acc_curve" : history["probe_acc"],
        "latent_dim"      : LATENT_DIM,
        "adam_lr"         : ADAM_LR,
        "batch_size"      : BATCH_SIZE,
        "training"        : "ae",
        "encoder"         : "{arch_label}",
    }})
    logging.info(f"Done. probe_acc={{final_acc:.1%}}  {{elapsed:.1f}}s")
'''

DAE_TRAIN = '''\
# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, Y_tr, X_te, Y_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {{"loss": [], "probe_acc": []}}
    t0          = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(np.array(X_tr)[perm][:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784))
        params, opt_state, loss = epoch_fn(params, opt_state, Xs, jax.random.fold_in(key, epoch))
        loss = float(loss)
        history["loss"].append(loss)

        probe_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te)
            history["probe_acc"].append((epoch, acc))
            probe_str = f"  probe_acc={{acc:.1%}}"

        logging.info(f"epoch {{epoch:3d}}  loss={{loss:.4f}}{{probe_str}}  {{time.perf_counter()-t0:.1f}}s")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {{jax.devices()}}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_train = np.array(data.y)
    Y_test  = np.array(data.y_test)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {{n_params(params):,}}")

    params, history, elapsed = train(params, X_train, Y_train, X_test, Y_test)

    with open(Path(__file__).parent / f"params_{{EXP_NAME}}.pkl", "wb") as f:
        pickle.dump({{k: np.array(v) for k, v in params.items()}}, f)
    with open(Path(__file__).parent / f"history_{{EXP_NAME}}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_acc = linear_probe_accuracy(params, X_train, Y_train, X_test, Y_test)
    append_result({{
        "experiment"      : EXP_NAME,
        "n_params"        : n_params(params),
        "epochs"          : MAX_EPOCHS,
        "time_s"          : round(elapsed, 1),
        "final_probe_acc" : round(final_acc, 6),
        "loss_curve"      : [round(v, 6) for v in history["loss"]],
        "probe_acc_curve" : history["probe_acc"],
        "latent_dim"      : LATENT_DIM,
        "adam_lr"         : ADAM_LR,
        "batch_size"      : BATCH_SIZE,
        "training"        : "dae",
        "encoder"         : "{arch_label}",
    }})
    logging.info(f"Done. probe_acc={{final_acc:.1%}}  {{elapsed:.1f}}s")
'''

VAE_TRAIN = '''\
# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, Y_tr, X_te, Y_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {{"loss": [], "recon_loss": [], "kl_loss": [], "probe_acc": []}}
    t0          = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(np.array(X_tr)[perm][:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784))
        params, opt_state, loss, recon_loss, kl_loss = epoch_fn(params, opt_state, Xs, jax.random.fold_in(key, epoch))
        loss, recon_loss, kl_loss = float(loss), float(recon_loss), float(kl_loss)
        history["loss"].append(loss)
        history["recon_loss"].append(recon_loss)
        history["kl_loss"].append(kl_loss)

        probe_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te)
            history["probe_acc"].append((epoch, acc))
            probe_str = f"  probe_acc={{acc:.1%}}"

        logging.info(f"epoch {{epoch:3d}}  loss={{loss:.4f}}  recon={{recon_loss:.4f}}  kl={{kl_loss:.4f}}{{probe_str}}  {{time.perf_counter()-t0:.1f}}s")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {{jax.devices()}}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_train = np.array(data.y)
    Y_test  = np.array(data.y_test)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {{n_params(params):,}}")

    params, history, elapsed = train(params, X_train, Y_train, X_test, Y_test)

    with open(Path(__file__).parent / f"params_{{EXP_NAME}}.pkl", "wb") as f:
        pickle.dump({{k: np.array(v) for k, v in params.items()}}, f)
    with open(Path(__file__).parent / f"history_{{EXP_NAME}}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_acc = linear_probe_accuracy(params, X_train, Y_train, X_test, Y_test)
    append_result({{
        "experiment"        : EXP_NAME,
        "n_params"          : n_params(params),
        "epochs"            : MAX_EPOCHS,
        "time_s"            : round(elapsed, 1),
        "final_probe_acc"   : round(final_acc, 6),
        "loss_curve"        : [round(v, 6) for v in history["loss"]],
        "recon_loss_curve"  : [round(v, 6) for v in history["recon_loss"]],
        "kl_loss_curve"     : [round(v, 6) for v in history["kl_loss"]],
        "probe_acc_curve"   : history["probe_acc"],
        "latent_dim"        : LATENT_DIM,
        "adam_lr"           : ADAM_LR,
        "batch_size"        : BATCH_SIZE,
        "beta_kl"           : BETA,
        "training"          : "vae",
        "encoder"           : "{arch_label}",
    }})
    logging.info(f"Done. probe_acc={{final_acc:.1%}}  {{elapsed:.1f}}s")
'''

# ── Encoder definitions (stripped of predictor params) ────────────────────────
# Each ENC_* block contains: arch constants, init_params (encoder+decoder only),
# and encode() function. The decoder params are injected via DECODER_PARAMS.

ENC_MLP = '''\
# ── Architecture constants ─────────────────────────────────────────────────────

MLP_ENC_HIDDEN = 512

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "W_enc1"      : n((784, MLP_ENC_HIDDEN),          scale=784**-0.5),
        "b_enc1"      : jnp.zeros(MLP_ENC_HIDDEN),
        "W_enc2"      : n((MLP_ENC_HIDDEN, LATENT_DIM),   scale=MLP_ENC_HIDDEN**-0.5),
        "b_enc2"      : jnp.zeros(LATENT_DIM),
{decoder_params}
{vae_params}    }

def encode(params, x):
    """MLP-shallow.  x: (B, 784) -> (B, LATENT_DIM)."""
    x_norm = x / 255.0
    h = jax.nn.gelu(x_norm @ params["W_enc1"] + params["b_enc1"])
    return h @ params["W_enc2"] + params["b_enc2"]
'''

ENC_MLP2 = '''\
# ── Architecture constants ─────────────────────────────────────────────────────

MLP_ENC_H1 = 1024
MLP_ENC_H2 = 1024

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "W_enc1"      : n((784, MLP_ENC_H1),             scale=784**-0.5),
        "b_enc1"      : jnp.zeros(MLP_ENC_H1),
        "W_enc2"      : n((MLP_ENC_H1, MLP_ENC_H2),      scale=MLP_ENC_H1**-0.5),
        "b_enc2"      : jnp.zeros(MLP_ENC_H2),
        "W_enc3"      : n((MLP_ENC_H2, LATENT_DIM),       scale=MLP_ENC_H2**-0.5),
        "b_enc3"      : jnp.zeros(LATENT_DIM),
{decoder_params}
{vae_params}    }

def encode(params, x):
    """MLP-deep.  x: (B, 784) -> (B, LATENT_DIM)."""
    x_norm = x / 255.0
    h = jax.nn.gelu(x_norm       @ params["W_enc1"] + params["b_enc1"])
    h = jax.nn.gelu(h            @ params["W_enc2"] + params["b_enc2"])
    return h @ params["W_enc3"] + params["b_enc3"]
'''

ENC_CONV = '''\
# ── Architecture constants ─────────────────────────────────────────────────────

CONV_CH = [32, 64, 128]

# ── Helpers ───────────────────────────────────────────────────────────────────

_DIM_NUMS = ("NHWC", "HWIO", "NHWC")

def _conv2d(x, W, b):
    return jax.lax.conv_general_dilated(x, W, (1, 1), "SAME", dimension_numbers=_DIM_NUMS) + b

def _maxpool2x2(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    c1, c2, c3 = CONV_CH
    return {
        "W_conv1"     : n((3, 3,  1, c1), scale=(9*1 )**-0.5),
        "b_conv1"     : jnp.zeros(c1),
        "W_conv2"     : n((3, 3, c1, c2), scale=(9*c1)**-0.5),
        "b_conv2"     : jnp.zeros(c2),
        "W_conv3"     : n((3, 3, c2, c3), scale=(9*c2)**-0.5),
        "b_conv3"     : jnp.zeros(c3),
        "W_fc"        : n((c3, LATENT_DIM),              scale=c3**-0.5),
        "b_fc"        : jnp.zeros(LATENT_DIM),
{decoder_params}
{vae_params}    }

def encode(params, x):
    """ConvNet-v1 [32,64,128]+GAP.  x: (B, 784) -> (B, LATENT_DIM)."""
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv2d(h, params["W_conv1"], params["b_conv1"]))
    h = _maxpool2x2(h)
    h = jax.nn.gelu(_conv2d(h, params["W_conv2"], params["b_conv2"]))
    h = _maxpool2x2(h)
    h = jax.nn.gelu(_conv2d(h, params["W_conv3"], params["b_conv3"]))
    h = h.mean(axis=(1, 2))
    return h @ params["W_fc"] + params["b_fc"]
'''

ENC_CONV2 = '''\
# ── Architecture constants ─────────────────────────────────────────────────────

CONV_CH   = [64, 128, 256]
FC_HIDDEN = 512

# ── Helpers ───────────────────────────────────────────────────────────────────

_DN = ("NHWC", "HWIO", "NHWC")

def _conv(x, W, b):
    return jax.lax.conv_general_dilated(x, W, (1, 1), "SAME", dimension_numbers=_DN) + b

def _maxpool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    c1, c2, c3 = CONV_CH
    return {
        "W_conv1"  : n((3, 3,  1, c1), scale=(9*1 )**-0.5),
        "b_conv1"  : jnp.zeros(c1),
        "W_conv2"  : n((3, 3, c1, c2), scale=(9*c1)**-0.5),
        "b_conv2"  : jnp.zeros(c2),
        "W_conv3"  : n((3, 3, c2, c3), scale=(9*c2)**-0.5),
        "b_conv3"  : jnp.zeros(c3),
        "W_fc1"    : n((c3, FC_HIDDEN),              scale=c3**-0.5),
        "b_fc1"    : jnp.zeros(FC_HIDDEN),
        "W_fc2"    : n((FC_HIDDEN, LATENT_DIM),       scale=FC_HIDDEN**-0.5),
        "b_fc2"    : jnp.zeros(LATENT_DIM),
{decoder_params}
{vae_params}    }

def encode(params, x):
    """ConvNet-v2 [64,128,256]+FC(512).  x: (B, 784) -> (B, LATENT_DIM)."""
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))
    h = h.mean(axis=(1, 2))
    h = jax.nn.gelu(h @ params["W_fc1"] + params["b_fc1"])
    return h @ params["W_fc2"] + params["b_fc2"]
'''

ENC_VIT = '''\
# ── Architecture constants ─────────────────────────────────────────────────────

PATCH_SIZE = 4
N_PATCHES  = 49
PATCH_DIM  = 16
D_MODEL    = 128
N_HEADS    = 4
DH         = D_MODEL // N_HEADS
FFN_DIM    = 256
N_LAYERS   = 3

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    params = {
        "W_patch"   : n((PATCH_DIM, D_MODEL),   scale=PATCH_DIM**-0.5),
        "b_patch"   : jnp.zeros(D_MODEL),
        "pos_emb"   : n((N_PATCHES, D_MODEL),   scale=0.02),
    }
    for i in range(N_LAYERS):
        params.update({
            f"W_q_{i}"   : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"W_k_{i}"   : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"W_v_{i}"   : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"W_o_{i}"   : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"W_ff1_{i}" : n((D_MODEL, FFN_DIM), scale=D_MODEL**-0.5),
            f"b_ff1_{i}" : jnp.zeros(FFN_DIM),
            f"W_ff2_{i}" : n((FFN_DIM, D_MODEL), scale=FFN_DIM**-0.5),
            f"b_ff2_{i}" : jnp.zeros(D_MODEL),
            f"ln1_g_{i}" : jnp.ones(D_MODEL),
            f"ln1_b_{i}" : jnp.zeros(D_MODEL),
            f"ln2_g_{i}" : jnp.ones(D_MODEL),
            f"ln2_b_{i}" : jnp.zeros(D_MODEL),
        })
    params.update({
        "ln_g"      : jnp.ones(D_MODEL),
        "ln_b"      : jnp.zeros(D_MODEL),
        "W_head"    : n((D_MODEL, LATENT_DIM),   scale=D_MODEL**-0.5),
        "b_head"    : jnp.zeros(LATENT_DIM),
{decoder_params}
{vae_params}    })
    return params

def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b

def _mha(params, x, i):
    B, S, D = x.shape
    Q = (x @ params[f"W_q_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)
    K = (x @ params[f"W_k_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)
    V = (x @ params[f"W_v_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)
    attn = jax.nn.softmax(Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(DH), axis=-1)
    return (attn @ V).transpose(0, 2, 1, 3).reshape(B, S, D) @ params[f"W_o_{i}"]

def _ffn(params, x, i):
    h = jax.nn.gelu(x @ params[f"W_ff1_{i}"] + params[f"b_ff1_{i}"])
    return h @ params[f"W_ff2_{i}"] + params[f"b_ff2_{i}"]

def _transformer_layer(params, x, i):
    x = x + _mha(params, _ln(x, params[f"ln1_g_{i}"], params[f"ln1_b_{i}"]), i)
    x = x + _ffn(params, _ln(x, params[f"ln2_g_{i}"], params[f"ln2_b_{i}"]), i)
    return x

def _extract_patches(x):
    B = x.shape[0]
    img = x.reshape(B, 28, 28)
    return img.reshape(B, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(B, 49, 16)

def encode(params, x):
    """mini ViT.  x: (B, 784) -> (B, LATENT_DIM)."""
    patches = _extract_patches(x / 255.0)
    h = patches @ params["W_patch"] + params["b_patch"] + params["pos_emb"][None]
    for i in range(N_LAYERS):
        h = _transformer_layer(params, h, i)
    h = _ln(h, params["ln_g"], params["ln_b"]).mean(axis=1)
    return h @ params["W_head"] + params["b_head"]
'''

# ViT-patch uses identical encoder; masking differs in loss
ENC_VIT_PATCH = ENC_VIT

ENC_GRAM_SINGLE = '''\
# ── Architecture constants ─────────────────────────────────────────────────────

D_R, D_C, D_V = 6, 6, 4
D              = D_R * D_C * D_V
D_RC           = D_R * D_C
DEC_RANK       = 32

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "enc_row_emb" : n((28, D_R),                    scale=D_R**-0.5),
        "enc_col_emb" : n((28, D_C),                    scale=D_C**-0.5),
        "W_pos_enc"   : n((D_RC, D),                    scale=D_RC**-0.5),
        "W_enc"       : n((D, DEC_RANK),                scale=D**-0.5),
        "W_proj"      : n((DEC_RANK**2, LATENT_DIM),    scale=(DEC_RANK**2)**-0.5),
{decoder_params}
{vae_params}    }

def encode(params, x):
    """Gram-single.  x: (B, 784) -> (B, LATENT_DIM)."""
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    z        = H @ params["W_proj"]
    return z * H
'''

ENC_GRAM_GLU = '''\
# ── Architecture constants ─────────────────────────────────────────────────────

D_R, D_C, D_V = 6, 6, 4
D              = D_R * D_C * D_V
D_RC           = D_R * D_C
DEC_RANK       = 32

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "enc_row_emb" : n((28, D_R),                    scale=D_R**-0.5),
        "enc_col_emb" : n((28, D_C),                    scale=D_C**-0.5),
        "W_pos_enc"   : n((D_RC, D),                    scale=D_RC**-0.5),
        "W_enc"       : n((D, DEC_RANK),                scale=D**-0.5),
        "W_enc_A"     : n((DEC_RANK**2, LATENT_DIM),    scale=(DEC_RANK**2)**-0.5),
        "W_enc_B"     : n((DEC_RANK**2, LATENT_DIM),    scale=(DEC_RANK**2)**-0.5),
        "W_gate"      : n((DEC_RANK**2, DEC_RANK**2),   scale=(DEC_RANK**2)**-0.5),
{decoder_params}
{vae_params}    }

def encode(params, x):
    """Gram-GLU.  x: (B, 784) -> (B, LATENT_DIM)."""
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    gate     = jax.nn.sigmoid(H @ params["W_gate"])
    H_g      = H * gate
    a        = H_g @ params["W_enc_A"]
    b        = H_g @ params["W_enc_B"]
    return a * b
'''

# Gram-dual (from exp12a.py): DEC_RANK=64, Hadamard readout
ENC_GRAM_DUAL = '''\
# ── Architecture constants ─────────────────────────────────────────────────────

D_R, D_C, D_V = 6, 6, 4
D              = D_R * D_C * D_V   # 144
D_RC           = D_R * D_C         # 36
DEC_RANK       = 64                # DEC_RANK^2 = 4096 -> projected to LATENT_DIM=1024

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "enc_row_emb" : n((28, D_R),                         scale=D_R**-0.5),
        "enc_col_emb" : n((28, D_C),                         scale=D_C**-0.5),
        "W_pos_enc"   : n((D_RC, D),                         scale=D_RC**-0.5),
        "W_enc"       : n((D, DEC_RANK),                     scale=D**-0.5),
        "W_enc_A"     : n((DEC_RANK**2, LATENT_DIM),         scale=(DEC_RANK**2)**-0.5),
        "W_enc_B"     : n((DEC_RANK**2, LATENT_DIM),         scale=(DEC_RANK**2)**-0.5),
{decoder_params}
{vae_params}    }

def encode(params, x):
    """Gram-dual (Hadamard readout).  x: (B, 784) -> (B, LATENT_DIM)."""
    x_norm = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F = x_norm[:, :, None] * f[None, :, :]
    H = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    a = H @ params["W_enc_A"]
    b = H @ params["W_enc_B"]
    return a * b
'''

# ── Experiment specs ──────────────────────────────────────────────────────────

ARCHITECTURES = [
    dict(arch_key="mlp",        arch_label="MLP-shallow",  encoder=ENC_MLP,        use_patch_mask=False),
    dict(arch_key="mlp2",       arch_label="MLP-deep",     encoder=ENC_MLP2,       use_patch_mask=False),
    dict(arch_key="conv",       arch_label="ConvNet-v1",   encoder=ENC_CONV,       use_patch_mask=False),
    dict(arch_key="conv2",      arch_label="ConvNet-v2",   encoder=ENC_CONV2,      use_patch_mask=False),
    dict(arch_key="vit",        arch_label="ViT",          encoder=ENC_VIT,        use_patch_mask=False),
    dict(arch_key="vit_patch",  arch_label="ViT-patch",    encoder=ENC_VIT_PATCH,  use_patch_mask=True),
    dict(arch_key="gram_dual",  arch_label="Gram-dual",    encoder=ENC_GRAM_DUAL,  use_patch_mask=False),
    dict(arch_key="gram_single",arch_label="Gram-single",  encoder=ENC_GRAM_SINGLE,use_patch_mask=False),
    dict(arch_key="gram_glu",   arch_label="Gram-GLU",     encoder=ENC_GRAM_GLU,   use_patch_mask=False),
]

TRAINING_TYPES = [
    dict(
        training     = "ae",
        training_upper = "AE",
        training_desc  = "Standard autoencoder, MSE loss in [0,1] space.",
        loss_block   = AE_LOSS,
        epoch_fn     = AE_EPOCH_FN,
        train_block  = AE_TRAIN,
        patch_mask_fn_needed = False,
        patch_mask_fn_needed_for_dae = False,
    ),
    dict(
        training     = "dae",
        training_upper = "DAE",
        training_desc  = "Denoising autoencoder, 4 augmentations (gaussian noise + 3 masking levels), MSE to clean target.",
        loss_block   = None,  # set per arch
        epoch_fn     = DAE_EPOCH_FN,
        train_block  = DAE_TRAIN,
        patch_mask_fn_needed = True,  # only for vit_patch
    ),
    dict(
        training     = "vae",
        training_upper = "VAE",
        training_desc  = "Variational autoencoder, beta-VAE (BETA=0.001), encode returns mu at eval.",
        loss_block   = VAE_LOSS,
        epoch_fn     = VAE_EPOCH_FN,
        train_block  = VAE_TRAIN,
        patch_mask_fn_needed = False,
    ),
]

# ── Generate ──────────────────────────────────────────────────────────────────

def build_init_params_block(enc_template, training):
    """Inject decoder + optional VAE params into the encoder template."""
    vae_extra = VAE_PARAMS + "\n" if training == "vae" else ""
    return (enc_template
            .replace("{decoder_params}", DECODER_PARAMS)
            .replace("{vae_params}", vae_extra))

for arch in ARCHITECTURES:
    for ttype in TRAINING_TYPES:
        training    = ttype["training"]
        arch_key    = arch["arch_key"]
        arch_label  = arch["arch_label"]
        exp_name    = f"exp_{training}_{arch_key}"
        use_patch   = arch["use_patch_mask"]

        # Header
        header = HEADER.format(
            exp_name       = exp_name,
            arch_label     = arch_label,
            training_upper = ttype["training_upper"],
            training_desc  = ttype["training_desc"],
        )

        # Encoder block with decoder params injected
        enc_block = build_init_params_block(arch["encoder"], training)

        # Decoder function
        decoder_fn = DECODER_FN

        # Patch mask helper (for DAE vit_patch, prepended before loss)
        patch_prefix = ""
        if training == "dae" and use_patch:
            patch_prefix = PATCH_MASK_FN

        # Loss block
        if training == "ae":
            loss_block = AE_LOSS
        elif training == "dae":
            loss_block = DAE_LOSS_PATCH if use_patch else DAE_LOSS_PIXEL
        else:
            loss_block = VAE_LOSS

        # Epoch fn
        epoch_fn_block = ttype["epoch_fn"]

        # Train block with arch_label substituted
        train_block = ttype["train_block"].format(arch_label=arch_label)

        content = (
            header
            + enc_block
            + decoder_fn
            + patch_prefix
            + loss_block
            + epoch_fn_block
            + LINEAR_PROBE
            + train_block
        )

        out = SSL / f"{exp_name}.py"
        out.write_text(content)
        print(f"Wrote {out.name}  ({len(content)} chars)")

print("Done.")
