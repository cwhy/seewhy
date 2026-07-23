"""
exp7c — LeWM JEPA on MNIST: Gaussian noise sweep NOISE_STD=75 epochs=200.

vs exp7 (93.5%, std=50, 100ep): NOISE_STD=75, MAX_EPOCHS=200.

Based on:
  Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
  Architecture from Pixels", 2026. https://arxiv.org/abs/2603.19312

Context (prev)  = MNIST image with 50% of pixels randomly zeroed (per-batch mask)
Target  (after) = complete MNIST image

Architecture:
  encoder  (gram-style, mirrors imle-gram decoder): image (784,) → embedding (LATENT_DIM,)
  predictor (MLP)                                 : masked embedding → predicted full embedding

Loss:
  L = MSE(predictor(encode(x_masked)), encode(x_full))
      + LAMBDA * SIGReg(encode(x_full))

Eval: linear-probe accuracy on frozen embeddings of complete test images.

Usage:
    uv run python projects/ssl/exp7c.py
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

EXP_NAME = "exp7c"

# ── Hyperparameters ────────────────────────────────────────────────────────────

BATCH_SIZE   = 256
SEED         = 42
MAX_EPOCHS   = 200

D_R, D_C, D_V = 6, 6, 4
D              = D_R * D_C * D_V   # 144  (matches imle-gram)
D_RC           = D_R * D_C         # 36

LATENT_DIM   = 128
DEC_RANK     = 32
MLP_HIDDEN   = 512

LAMBDA       = 0.1    # SIGReg weight
N_PROJ       = 1024   # SIGReg: random projection directions
SIGREG_KNOTS = 17     # SIGReg: quadrature knots over [0, 3]

ADAM_LR      = 3e-4

EVAL_EVERY   = 10     # linear-probe eval every N epochs
PROBE_EPOCHS = 10     # gradient epochs for linear probe

JSONL = Path(__file__).parent / "results.jsonl"

# ── Precomputed constants ──────────────────────────────────────────────────────

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28
NOISE_STD  = 75.0   # Gaussian noise std in pixel units (0–255 scale)

# SIGReg quadrature: trapezoidal weights * Gaussian window
_t_vec = jnp.linspace(0.0, 3.0, SIGREG_KNOTS)               # (K,)
_dt    = 3.0 / (SIGREG_KNOTS - 1)
_trap  = jnp.full(SIGREG_KNOTS, 2.0 * _dt).at[0].set(_dt).at[-1].set(_dt)
SIGREG_PHI     = jnp.exp(-0.5 * _t_vec ** 2)                # N(0,1) char. fn   (K,)
SIGREG_WEIGHTS = _trap * SIGREG_PHI                          # quadrature weights (K,)

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
        # ── Encoder (gram-style, mirrors imle-gram decoder structure) ──────────
        "enc_row_emb" : n((28, D_R),               scale=D_R**-0.5),
        "enc_col_emb" : n((28, D_C),               scale=D_C**-0.5),
        "W_pos_enc"   : n((D_RC, D),               scale=D_RC**-0.5),   # (36, 144)
        "W_enc"       : n((D, DEC_RANK),           scale=D**-0.5),      # (144, 32) shared
        "W_enc1"      : n((DEC_RANK**2, MLP_HIDDEN), scale=(DEC_RANK**2)**-0.5),  # (256, 256)
        "b_enc1"      : jnp.zeros(MLP_HIDDEN),
        "W_enc2"      : n((MLP_HIDDEN, LATENT_DIM), scale=MLP_HIDDEN**-0.5),      # (256, 64)
        "b_enc2"      : jnp.zeros(LATENT_DIM),
        # ── Predictor ─────────────────────────────────────────────────────────
        "W_pred1"     : n((LATENT_DIM, MLP_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"     : jnp.zeros(MLP_HIDDEN),
        "W_pred2"     : n((MLP_HIDDEN, LATENT_DIM), scale=MLP_HIDDEN**-0.5),
        "b_pred2"     : jnp.zeros(LATENT_DIM),
    }

# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(params, x):
    """
    Hebbian gram encoder.  x: (B, 784) float32 raw pixels → embedding: (B, LATENT_DIM).

    Each pixel p is treated as a data point with feature vector f[p] (derived from
    its row/col position) and activation a[p] = x[p] / 255.

    The Hebbian matrix accumulates co-activations across all 784 pixels:
      H = Fᵀ F   where F[p] = a[p] * f[p]   (784, DEC_RANK)

    H[i,j] = Σ_p a[p]² · f[p,i] · f[p,j]  — how much feature i and j fire together.
    H is then flattened and passed through an MLP to produce the embedding.
    """
    x_norm = x / 255.0   # (B, 784)

    # Per-pixel positional features: (784, D) → (784, DEC_RANK)
    re       = params["enc_row_emb"][ROWS]    # (784, D_R)
    ce       = params["enc_col_emb"][COLS]    # (784, D_C)
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)  # (784, D_RC)
    pos_feat = pos_rc @ params["W_pos_enc"]                                              # (784, D)
    f        = pos_feat @ params["W_enc"]                                                # (784, DEC_RANK)

    # Weighted pixel features: F[b,p] = x_norm[b,p] * f[p]
    F = x_norm[:, :, None] * f[None, :, :]   # (B, 784, DEC_RANK)

    # Hebbian matrix: H = Fᵀ F  (sum of per-pixel outer products)
    H = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)  # (B, DEC_RANK²)

    # MLP to embedding
    h = jax.nn.gelu(H @ params["W_enc1"] + params["b_enc1"])   # (B, MLP_HIDDEN)
    return h @ params["W_enc2"] + params["b_enc2"]              # (B, LATENT_DIM)

# ── Predictor ─────────────────────────────────────────────────────────────────

def predict(params, e):
    """e: (B, LATENT_DIM) → predicted embedding: (B, LATENT_DIM)."""
    h = jax.nn.gelu(e @ params["W_pred1"] + params["b_pred1"])   # (B, MLP_HIDDEN)
    return h @ params["W_pred2"] + params["b_pred2"]              # (B, LATENT_DIM)

# ── SIGReg ────────────────────────────────────────────────────────────────────

def sigreg(emb, key):
    """
    Epps-Pulley test statistic (Maes et al. 2026).
    Tests whether emb ~ N(0, I).  emb: (B, LATENT_DIM).

    Uses trapezoidal quadrature with a Gaussian window over t ∈ [0, 3].
    """
    B = emb.shape[0]
    A   = jax.random.normal(key, (LATENT_DIM, N_PROJ))
    A   = A / jnp.linalg.norm(A, axis=0, keepdims=True)   # unit-norm columns  (D, N_PROJ)

    proj = emb @ A                                          # (B, N_PROJ)
    x_t  = proj[:, :, None] * _t_vec[None, None, :]        # (B, N_PROJ, K)

    cos_mean = jnp.cos(x_t).mean(0)                        # (N_PROJ, K)
    sin_mean = jnp.sin(x_t).mean(0)                        # (N_PROJ, K)

    err = (cos_mean - SIGREG_PHI[None, :]) ** 2 + sin_mean ** 2  # (N_PROJ, K)
    return (err @ SIGREG_WEIGHTS * B).mean()                      # scalar

# ── Loss ──────────────────────────────────────────────────────────────────────

def total_loss(params, x, key):
    """x: (B, 784) raw pixels.  Returns (loss, (pred_loss, sreg_loss))."""
    k_noise, k_sigreg = jax.random.split(key)
    noise    = jax.random.normal(k_noise, x.shape) * NOISE_STD      # (B, 784)
    x_masked = jnp.clip(x + noise, 0.0, 255.0)

    e_full   = encode(params, x)           # (B, LATENT_DIM) — target
    e_masked = encode(params, x_masked)    # (B, LATENT_DIM) — context
    e_pred   = predict(params, e_masked)   # (B, LATENT_DIM) — prediction

    pred_loss = ((e_pred - e_full) ** 2).mean()
    sreg_loss = sigreg(e_full, k_sigreg)

    return pred_loss + LAMBDA * sreg_loss, (pred_loss, sreg_loss)

# ── Epoch function (lax.scan over batches) ────────────────────────────────────

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

# ── Linear-probe evaluation ───────────────────────────────────────────────────

def linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te):
    """
    Train a linear classifier on frozen complete-image embeddings.
    Returns test accuracy.
    """
    def encode_all(X):
        parts = []
        for i in range(0, len(X), 512):
            parts.append(np.array(encode(params, jnp.array(np.array(X[i:i+512])))))
        return np.concatenate(parts, 0)

    E_tr = encode_all(X_tr)    # (N_tr, LATENT_DIM)
    E_te = encode_all(X_te)    # (N_te, LATENT_DIM)

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
        Xs   = jnp.array(
            np.array(X_tr)[perm][: num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784)
        )
        params, opt_state, loss, pred_loss, sreg_loss = epoch_fn(
            params, opt_state, Xs, jax.random.fold_in(key, epoch)
        )
        loss, pred_loss, sreg_loss = float(loss), float(pred_loss), float(sreg_loss)
        history["loss"].append(loss)
        history["pred_loss"].append(pred_loss)
        history["sreg_loss"].append(sreg_loss)

        probe_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te)
            history["probe_acc"].append((epoch, acc))
            probe_str = f"  probe_acc={acc:.1%}"

        logging.info(
            f"epoch {epoch:3d}  loss={loss:.4f}  pred={pred_loss:.4f}"
            f"  sigreg={sreg_loss:.4f}{probe_str}  {time.perf_counter()-t0:.1f}s"
        )

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
    logging.info(f"Parameters: {n_params(params):,}  (encoder + predictor)")
    logging.info(
        f"LATENT_DIM={LATENT_DIM}  DEC_RANK={DEC_RANK}  MLP_HIDDEN={MLP_HIDDEN}"
        f"  LAMBDA={LAMBDA}  N_PROJ={N_PROJ}"
    )

    params, history, elapsed = train(params, X_train, Y_train, X_test, Y_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_acc = linear_probe_accuracy(params, X_train, Y_train, X_test, Y_test)
    append_result({
        "experiment"      : EXP_NAME,
        "name"            : f"jepa-hebbian noise_std={NOISE_STD} LATENT={LATENT_DIM} DEC_RANK={DEC_RANK} MLP_HIDDEN={MLP_HIDDEN}",
        "n_params"        : n_params(params),
        "epochs"          : MAX_EPOCHS,
        "time_s"          : round(elapsed, 1),
        "final_probe_acc" : round(final_acc, 6),
        "loss_curve"      : [round(v, 6) for v in history["loss"]],
        "pred_loss_curve" : [round(v, 6) for v in history["pred_loss"]],
        "sreg_loss_curve" : [round(v, 6) for v in history["sreg_loss"]],
        "probe_acc_curve" : history["probe_acc"],
        "latent_dim"      : LATENT_DIM,
        "dec_rank"        : DEC_RANK,
        "mlp_hidden"      : MLP_HIDDEN,
        "lambda_sigreg"   : LAMBDA,
        "noise_std"       : NOISE_STD,
        "n_proj"          : N_PROJ,
        "sigreg_knots"    : SIGREG_KNOTS,
        "adam_lr"         : ADAM_LR,
        "batch_size"      : BATCH_SIZE,
    })
    logging.info(f"Done. probe_acc={final_acc:.1%}  {elapsed:.1f}s")
