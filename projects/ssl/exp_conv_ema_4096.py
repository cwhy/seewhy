"""
exp_conv_ema_4096 — ConvNet encoder + EMA + augmentation, LATENT_DIM=4096.

ConvNet encoder:
  (B,784) → reshape (B,28,28,1) → Conv(32,3×3)+GeLU → MaxPool(2×2)
           → Conv(64,3×3)+GeLU → MaxPool(2×2)
           → Conv(128,3×3)+GeLU → GlobalAvgPool → Linear(LATENT_DIM)

Everything else identical to exp_mlp_ema (4 actions, FiLM predictor, EMA τ=0.996).

Usage:
    uv run python projects/ssl/exp_conv_ema_4096.py
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

EXP_NAME = "exp_conv_ema_4096"

# ── Hyperparameters ────────────────────────────────────────────────────────────

BATCH_SIZE      = 256
SEED            = 42
MAX_EPOCHS      = 200

LATENT_DIM      = 4096
CONV_CH         = [32, 64, 128]   # channels per conv layer
MLP_PRED_HIDDEN = 512             # predictor hidden size

N_ACTIONS    = 4
ACTION_DIM   = 32
ACTION_LABELS = ["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]

NOISE_STD    = 75.0
MASK_RATIOS  = [0.25, 0.50, 0.75]

EMA_TAU      = 0.996

ADAM_LR      = 3e-4

EVAL_EVERY   = 10
PROBE_EPOCHS = 10

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
    c1, c2, c3 = CONV_CH
    return {
        # ── Encoder (ConvNet, HWIO kernel format) ─────────────────────────────
        "W_conv1"     : n((3, 3,  1, c1), scale=(3*3*1 )**-0.5),
        "b_conv1"     : jnp.zeros(c1),
        "W_conv2"     : n((3, 3, c1, c2), scale=(3*3*c1)**-0.5),
        "b_conv2"     : jnp.zeros(c2),
        "W_conv3"     : n((3, 3, c2, c3), scale=(3*3*c2)**-0.5),
        "b_conv3"     : jnp.zeros(c3),
        "W_fc"        : n((c3, LATENT_DIM),            scale=c3**-0.5),
        "b_fc"        : jnp.zeros(LATENT_DIM),
        # ── Action embedding ───────────────────────────────────────────────────
        "action_emb"  : n((N_ACTIONS, ACTION_DIM),      scale=0.01),
        # ── Predictor (FiLM conditioned) ───────────────────────────────────────
        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"     : jnp.zeros(LATENT_DIM),
    }

# ── Encoder (ConvNet) ─────────────────────────────────────────────────────────

_DIM_NUMS = ("NHWC", "HWIO", "NHWC")

def _conv2d(x, W, b):
    """x: (B,H,W,C_in), W: (kH,kW,C_in,C_out) → (B,H,W,C_out), SAME padding."""
    return jax.lax.conv_general_dilated(
        x, W, (1, 1), "SAME", dimension_numbers=_DIM_NUMS
    ) + b

def _maxpool2x2(x):
    """(B,H,W,C) → (B,H/2,W/2,C) via 2×2 max pooling."""
    return jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="VALID",
    )

def encode(params, x):
    """x: (B, 784) float32 raw pixels → (B, LATENT_DIM)."""
    h = (x / 255.0).reshape(-1, 28, 28, 1)          # NHWC
    h = jax.nn.gelu(_conv2d(h, params["W_conv1"], params["b_conv1"]))
    h = _maxpool2x2(h)                                # 28→14
    h = jax.nn.gelu(_conv2d(h, params["W_conv2"], params["b_conv2"]))
    h = _maxpool2x2(h)                                # 14→7
    h = jax.nn.gelu(_conv2d(h, params["W_conv3"], params["b_conv3"]))
    h = h.mean(axis=(1, 2))                           # global avg pool → (B,128)
    return h @ params["W_fc"] + params["b_fc"]

# ── Predictor (FiLM) ──────────────────────────────────────────────────────────

def predict(params, e, a):
    h     = jax.nn.gelu(e @ params["W_pred1"] + params["b_pred1"])
    scale = a @ params["W_film_s"] + params["b_film_s"]
    shift = a @ params["W_film_t"] + params["b_film_t"]
    h     = h * (1.0 + scale) + shift
    return h @ params["W_pred2"] + params["b_pred2"]

# ── EMA update ────────────────────────────────────────────────────────────────

@jax.jit
def ema_update(target_params, online_params):
    return jax.tree_util.tree_map(
        lambda t, o: EMA_TAU * t + (1.0 - EMA_TAU) * o,
        target_params, online_params
    )

# ── Loss ──────────────────────────────────────────────────────────────────────

def total_loss(online_params, target_params, x, key):
    k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)
    B = x.shape[0]

    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)
    a_emb = online_params["action_emb"][idx]

    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))

    e_target = jax.lax.stop_gradient(encode(target_params, x))
    e_aug    = encode(online_params, x_aug)
    e_pred   = predict(online_params, e_aug, a_emb)

    pred_loss = ((e_pred - e_target) ** 2).mean()
    return pred_loss, (pred_loss,)

# ── Epoch function ────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    def scan_body(carry, inputs):
        online_params, target_params, opt_state = carry
        x_batch, key = inputs
        (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
            online_params, target_params, x_batch, key
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, online_params)
        new_online = optax.apply_updates(online_params, updates)
        new_target = ema_update(target_params, new_online)
        return (new_online, new_target, new_opt_state), (loss, *aux)

    def epoch_fn(online_params, target_params, opt_state, Xs_batched, epoch_key):
        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (online_params, target_params, opt_state), stats = jax.lax.scan(
            scan_body, (online_params, target_params, opt_state), (Xs_batched, keys)
        )
        return online_params, target_params, opt_state, *[s.mean() for s in stats]

    return jax.jit(epoch_fn)

# ── Linear-probe evaluation ───────────────────────────────────────────────────

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
        def ce(wb):
            return optax.softmax_cross_entropy_with_integer_labels(e @ wb[0] + wb[1], y).mean()
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

def train(online_params, X_tr, Y_tr, X_te, Y_te):
    num_steps     = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer     = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state     = optimizer.init(online_params)
    target_params = jax.tree_util.tree_map(lambda x: x.copy(), online_params)
    epoch_fn      = make_epoch_fn(optimizer)
    key_gen       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches   = len(X_tr) // BATCH_SIZE
    history       = {"loss": [], "probe_acc": []}
    t0            = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(
            np.array(X_tr)[perm][: num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784)
        )
        online_params, target_params, opt_state, loss, _ = epoch_fn(
            online_params, target_params, opt_state, Xs, jax.random.fold_in(key, epoch)
        )
        loss = float(loss)
        history["loss"].append(loss)

        probe_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = linear_probe_accuracy(online_params, X_tr, Y_tr, X_te, Y_te)
            history["probe_acc"].append((epoch, acc))
            probe_str = f"  probe_acc={acc:.1%}"

        logging.info(
            f"epoch {epoch:3d}  loss={loss:.4f}{probe_str}  {time.perf_counter()-t0:.1f}s"
        )

    return online_params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_train = np.array(data.y)
    Y_test  = np.array(data.y_test)

    key_gen       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    online_params = init_params(key_gen)
    logging.info(f"Parameters: {n_params(online_params):,}")
    logging.info(
        f"LATENT_DIM={LATENT_DIM}  conv_ch={CONV_CH}"
        f"  EMA_TAU={EMA_TAU}  N_ACTIONS={N_ACTIONS}  encoder=conv"
    )

    online_params, history, elapsed = train(online_params, X_train, Y_train, X_test, Y_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in online_params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_acc = linear_probe_accuracy(online_params, X_train, Y_train, X_test, Y_test)
    append_result({
        "experiment"      : EXP_NAME,
        "name"            : f"jepa-conv-ema aug LATENT={LATENT_DIM} conv_ch={CONV_CH}",
        "n_params"        : n_params(online_params),
        "epochs"          : MAX_EPOCHS,
        "time_s"          : round(elapsed, 1),
        "final_probe_acc" : round(final_acc, 6),
        "loss_curve"      : [round(v, 6) for v in history["loss"]],
        "probe_acc_curve" : history["probe_acc"],
        "latent_dim"      : LATENT_DIM,
        "conv_ch"         : CONV_CH,
        "ema_tau"         : EMA_TAU,
        "n_actions"       : N_ACTIONS,
        "noise_std"       : NOISE_STD,
        "mask_ratios"     : MASK_RATIOS,
        "adam_lr"         : ADAM_LR,
        "batch_size"      : BATCH_SIZE,
        "encoder"         : "conv",
        "augmentation"    : True,
    })
    logging.info(f"Done. probe_acc={final_acc:.1%}  {elapsed:.1f}s")
