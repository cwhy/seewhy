"""
exp_sigreg_mlp — MLP-shallow encoder + SIGReg-JEPA, LATENT_DIM=1024.

Training loss:
    e_full = encode(params, x_clean)
    e_aug  = encode(params, x_augmented)   [4 augmentations, FiLM-conditioned]
    e_pred = predict(params, e_aug, action_emb)
    L = MSE(e_pred, e_full) + LAMBDA * sigreg(e_full, key)

No online/target split — SIGReg prevents representation collapse.
Matches EMA experiment exactly in architecture, augmentations, and LATENT_DIM.
EMA counterpart: exp_mlp_ema_1024

Usage:
    uv run python projects/ssl/exp_sigreg_mlp.py
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

EXP_NAME = "exp_sigreg_mlp"

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
ACTION_LABELS = ["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]
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
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

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



def total_loss(params, x, key):
    """x: (B, 784).  Returns (loss, (pred_loss, sreg_loss))."""
    k_idx, k_noise, k_m1, k_m2, k_m3, k_sigreg = jax.random.split(key, 6)
    B     = x.shape[0]
    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)
    a_emb = params["action_emb"][idx]

    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)
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
    history     = {"loss": [], "pred_loss": [], "sreg_loss": [], "probe_acc": []}
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
            probe_str = f"  probe_acc={acc:.1%}"

        logging.info(f"epoch {epoch:3d}  loss={loss:.4f}  pred={pred_loss:.4f}  sigreg={sreg_loss:.4f}{probe_str}  {time.perf_counter()-t0:.1f}s")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_train = np.array(data.y)
    Y_test  = np.array(data.y_test)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}")

    params, history, elapsed = train(params, X_train, Y_train, X_test, Y_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_acc = linear_probe_accuracy(params, X_train, Y_train, X_test, Y_test)
    append_result({
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
        "encoder"         : "MLP-shallow",
    })
    logging.info(f"Done. probe_acc={final_acc:.1%}  {elapsed:.1f}s")
