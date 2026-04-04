"""
Generate SIGReg experiment files for all 8 non-Gram-dual architectures.

Each file is a port of the corresponding EMA experiment with:
  - same encoder architecture, same hyperparameters, same 4 augmentations
  - FiLM predictor conditioned on action (same as EMA)
  - SIGReg spectral regulariser instead of EMA momentum target encoder
  - single shared parameter dict (no online/target split)

Loss:
    e_full = encode(params, x_clean)
    e_aug  = encode(params, x_augmented)
    e_pred = predict(params, e_aug, action_emb)
    L = MSE(e_pred, e_full) + LAMBDA * sigreg(e_full, key)

Outputs (written to projects/ssl/):
    exp_sigreg_mlp.py
    exp_sigreg_mlp2.py
    exp_sigreg_conv.py
    exp_sigreg_conv2.py
    exp_sigreg_vit.py
    exp_sigreg_vit_patch.py
    exp_sigreg_gram_single.py
    exp_sigreg_gram_glu.py

Usage:
    uv run python projects/ssl/scripts/tmp/mk_exp_sigreg.py
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent

# ── Shared boilerplate ────────────────────────────────────────────────────────

HEADER = '''\
"""
{exp_name} — {arch_label} encoder + SIGReg-JEPA, LATENT_DIM=1024.

Training loss:
    e_full = encode(params, x_clean)
    e_aug  = encode(params, x_augmented)   [4 augmentations, FiLM-conditioned]
    e_pred = predict(params, e_aug, action_emb)
    L = MSE(e_pred, e_full) + LAMBDA * sigreg(e_full, key)

No online/target split — SIGReg prevents representation collapse.
Matches EMA experiment exactly in architecture, augmentations, and LATENT_DIM.
EMA counterpart: {ema_counterpart}

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
MLP_PRED_HIDDEN = 512

N_ACTIONS    = 4
ACTION_DIM   = 32
{action_labels_line}
NOISE_STD    = 75.0
MASK_RATIOS  = [0.25, 0.50, 0.75]

LAMBDA       = 0.1    # SIGReg weight
N_PROJ       = 1024   # SIGReg random projections
SIGREG_KNOTS = 17

JSONL = Path(__file__).parent / "results.jsonl"

# ── SIGReg quadrature constants ────────────────────────────────────────────────

_t_vec         = jnp.linspace(0.0, 3.0, SIGREG_KNOTS)
_dt            = 3.0 / (SIGREG_KNOTS - 1)
_trap          = jnp.full(SIGREG_KNOTS, 2.0 * _dt).at[0].set(_dt).at[-1].set(_dt)
SIGREG_PHI     = jnp.exp(-0.5 * _t_vec ** 2)
SIGREG_WEIGHTS = _trap * SIGREG_PHI

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\\n")

'''

FOOTER = '''\

# ── Predictor (FiLM conditioned on action) ────────────────────────────────────

def predict(params, e, a):
    """e: (B, LATENT_DIM), a: (B, ACTION_DIM) → (B, LATENT_DIM)."""
    h     = jax.nn.gelu(e @ params["W_pred1"] + params["b_pred1"])
    scale = a @ params["W_film_s"] + params["b_film_s"]
    shift = a @ params["W_film_t"] + params["b_film_t"]
    h     = h * (1.0 + scale) + shift
    return h @ params["W_pred2"] + params["b_pred2"]

# ── SIGReg ────────────────────────────────────────────────────────────────────

def sigreg(emb, key):
    """Epps-Pulley test vs N(0,I).  emb: (B, LATENT_DIM) → scalar."""
    B   = emb.shape[0]
    A   = jax.random.normal(key, (LATENT_DIM, N_PROJ))
    A   = A / jnp.linalg.norm(A, axis=0, keepdims=True)
    proj = emb @ A
    x_t  = proj[:, :, None] * _t_vec[None, None, :]
    cos_mean = jnp.cos(x_t).mean(0)
    sin_mean = jnp.sin(x_t).mean(0)
    err  = (cos_mean - SIGREG_PHI[None, :]) ** 2 + sin_mean ** 2
    return (err @ SIGREG_WEIGHTS * B).mean()

# ── Loss ──────────────────────────────────────────────────────────────────────

{augmentation_fn}

def total_loss(params, x, key):
    """x: (B, 784).  Returns (loss, (pred_loss, sreg_loss))."""
    k_idx, k_noise, k_m1, k_m2, k_m3, k_sigreg = jax.random.split(key, 6)
    B     = x.shape[0]
    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)
    a_emb = params["action_emb"][idx]

    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
{mask_lines}
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))

    e_full = encode(params, x)
    e_aug  = encode(params, x_aug)
    e_pred = predict(params, e_aug, a_emb)

    pred_loss = ((e_pred - e_full) ** 2).mean()
    sreg_loss = sigreg(e_full, k_sigreg)
    return pred_loss + LAMBDA * sreg_loss, (pred_loss, sreg_loss)

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

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, Y_tr, X_te, Y_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {{"loss": [], "pred_loss": [], "sreg_loss": [], "probe_acc": []}}
    t0          = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(np.array(X_tr)[perm][:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784))
        params, opt_state, loss, pred_loss, sreg_loss = epoch_fn(params, opt_state, Xs, jax.random.fold_in(key, epoch))
        loss, pred_loss, sreg_loss = float(loss), float(pred_loss), float(sreg_loss)
        history["loss"].append(loss)
        history["pred_loss"].append(pred_loss)
        history["sreg_loss"].append(sreg_loss)

        probe_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te)
            history["probe_acc"].append((epoch, acc))
            probe_str = f"  probe_acc={{acc:.1%}}"

        logging.info(f"epoch {{epoch:3d}}  loss={{loss:.4f}}  pred={{pred_loss:.4f}}  sigreg={{sreg_loss:.4f}}{{probe_str}}  {{time.perf_counter()-t0:.1f}}s")

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
        "pred_loss_curve" : [round(v, 6) for v in history["pred_loss"]],
        "sreg_loss_curve" : [round(v, 6) for v in history["sreg_loss"]],
        "probe_acc_curve" : history["probe_acc"],
        "latent_dim"      : LATENT_DIM,
        "lambda_sigreg"   : LAMBDA,
        "adam_lr"         : ADAM_LR,
        "batch_size"      : BATCH_SIZE,
        "training"        : "sigreg",
        "encoder"         : "{arch_label}",
    }})
    logging.info(f"Done. probe_acc={{final_acc:.1%}}  {{elapsed:.1f}}s")
'''

# ── Pixel masking (flat Bernoulli) ───────────────────────────────────────────

PIXEL_MASK_LINES = '''\
    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)\
'''

PATCH_MASK_LINES = '''\
    x_m25 = _patch_mask(x, k_m1, MASK_RATIOS[0])
    x_m50 = _patch_mask(x, k_m2, MASK_RATIOS[1])
    x_m75 = _patch_mask(x, k_m3, MASK_RATIOS[2])\
'''

PATCH_MASK_FN = '''\
def _patch_mask(x, key, mask_ratio):
    """Zero entire 4x4 patches at given ratio.  x: (B, 784) -> (B, 784)."""
    B       = x.shape[0]
    patches = x.reshape(B, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(B, 49, 16)
    keep    = jax.random.bernoulli(key, 1.0 - mask_ratio, shape=(B, 49))
    patches = patches * keep[:, :, None]
    return patches.reshape(B, 7, 7, 4, 4).transpose(0, 1, 3, 2, 4).reshape(B, 784)

'''

# ── Encoder definitions ────────────────────────────────────────────────────────

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
        "action_emb"  : n((N_ACTIONS, ACTION_DIM),         scale=0.01),
        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN),   scale=LATENT_DIM**-0.5),
        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN),   scale=ACTION_DIM**-0.5),
        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN),   scale=ACTION_DIM**-0.5),
        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM),   scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"     : jnp.zeros(LATENT_DIM),
    }

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
        "action_emb"  : n((N_ACTIONS, ACTION_DIM),         scale=0.01),
        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN),   scale=LATENT_DIM**-0.5),
        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN),   scale=ACTION_DIM**-0.5),
        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN),   scale=ACTION_DIM**-0.5),
        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM),   scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"     : jnp.zeros(LATENT_DIM),
    }

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
        "action_emb"  : n((N_ACTIONS, ACTION_DIM),        scale=0.01),
        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN),  scale=LATENT_DIM**-0.5),
        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN),  scale=ACTION_DIM**-0.5),
        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN),  scale=ACTION_DIM**-0.5),
        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM),  scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"     : jnp.zeros(LATENT_DIM),
    }

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
        "action_emb" : n((N_ACTIONS, ACTION_DIM),     scale=0.01),
        "W_pred1"  : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s" : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s" : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t" : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t" : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"  : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"  : jnp.zeros(LATENT_DIM),
    }

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
        "action_emb": n((N_ACTIONS, ACTION_DIM),  scale=0.01),
        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"   : jnp.zeros(LATENT_DIM),
    })
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

# ViT-patch: same init_params as ViT but different action labels and _patch_mask in loss
ENC_VIT_PATCH = ENC_VIT  # identical encoder; masking differs in loss

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
        "action_emb"  : n((N_ACTIONS, ACTION_DIM),      scale=0.01),
        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"     : jnp.zeros(LATENT_DIM),
    }

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
        "action_emb"  : n((N_ACTIONS, ACTION_DIM),      scale=0.01),
        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"     : jnp.zeros(LATENT_DIM),
    }

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

# ── Experiment specs ──────────────────────────────────────────────────────────

PIXEL_ACTION_LABELS = 'ACTION_LABELS = ["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]'
PATCH_ACTION_LABELS = 'ACTION_LABELS = ["denoise_gaussian_75", "patch_mask_25", "patch_mask_50", "patch_mask_75"]'

EXPERIMENTS = [
    dict(
        exp_name        = "exp_sigreg_mlp",
        arch_label      = "MLP-shallow",
        ema_counterpart = "exp_mlp_ema_1024",
        encoder_block   = ENC_MLP,
        action_labels   = PIXEL_ACTION_LABELS,
        augmentation_fn = "",
        mask_lines      = PIXEL_MASK_LINES,
    ),
    dict(
        exp_name        = "exp_sigreg_mlp2",
        arch_label      = "MLP-deep",
        ema_counterpart = "exp_mlp2_ema_1024",
        encoder_block   = ENC_MLP2,
        action_labels   = PIXEL_ACTION_LABELS,
        augmentation_fn = "",
        mask_lines      = PIXEL_MASK_LINES,
    ),
    dict(
        exp_name        = "exp_sigreg_conv",
        arch_label      = "ConvNet-v1",
        ema_counterpart = "exp_conv_ema_1024",
        encoder_block   = ENC_CONV,
        action_labels   = PIXEL_ACTION_LABELS,
        augmentation_fn = "",
        mask_lines      = PIXEL_MASK_LINES,
    ),
    dict(
        exp_name        = "exp_sigreg_conv2",
        arch_label      = "ConvNet-v2",
        ema_counterpart = "exp_conv2_ema_1024",
        encoder_block   = ENC_CONV2,
        action_labels   = PIXEL_ACTION_LABELS,
        augmentation_fn = "",
        mask_lines      = PIXEL_MASK_LINES,
    ),
    dict(
        exp_name        = "exp_sigreg_vit",
        arch_label      = "ViT",
        ema_counterpart = "exp_vit_ema_1024",
        encoder_block   = ENC_VIT,
        action_labels   = PIXEL_ACTION_LABELS,
        augmentation_fn = "",
        mask_lines      = PIXEL_MASK_LINES,
    ),
    dict(
        exp_name        = "exp_sigreg_vit_patch",
        arch_label      = "ViT-patch",
        ema_counterpart = "exp_vit_patch_ema",
        encoder_block   = ENC_VIT_PATCH,
        action_labels   = PATCH_ACTION_LABELS,
        augmentation_fn = PATCH_MASK_FN,
        mask_lines      = PATCH_MASK_LINES,
    ),
    dict(
        exp_name        = "exp_sigreg_gram_single",
        arch_label      = "Gram-single",
        ema_counterpart = "exp_gram_single_ema_1024",
        encoder_block   = ENC_GRAM_SINGLE,
        action_labels   = PIXEL_ACTION_LABELS,
        augmentation_fn = "",
        mask_lines      = PIXEL_MASK_LINES,
    ),
    dict(
        exp_name        = "exp_sigreg_gram_glu",
        arch_label      = "Gram-GLU",
        ema_counterpart = "exp_gram_glu_ema",
        encoder_block   = ENC_GRAM_GLU,
        action_labels   = PIXEL_ACTION_LABELS,
        augmentation_fn = "",
        mask_lines      = PIXEL_MASK_LINES,
    ),
]

# ── Generate ──────────────────────────────────────────────────────────────────

for spec in EXPERIMENTS:
    content = (
        HEADER.format(
            exp_name        = spec["exp_name"],
            arch_label      = spec["arch_label"],
            ema_counterpart = spec["ema_counterpart"],
            action_labels_line = spec["action_labels"],
        )
        + spec["encoder_block"]
        + FOOTER.format(
            augmentation_fn = spec["augmentation_fn"],
            mask_lines      = spec["mask_lines"],
            arch_label      = spec["arch_label"],
        )
    )
    out = SSL / f"{spec['exp_name']}.py"
    out.write_text(content)
    print(f"Wrote {out.name}  ({len(content)} chars)")

print("Done.")
