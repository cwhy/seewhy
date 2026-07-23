"""
Phase 0.4 baseline — dense-fire 11-class MNIST classifier (no ROSA).

Validates that the "fire-at-every-position with NO_OUTPUT class + λ_digit_boost"
mechanism is not lossy on its own. If 0.4 << 0.3 (the vanilla VQ-MLP), the
dense-fire setup is broken and exp1 has no chance.

Architecture (vs 0.3):
  + label encoder (10-way one-hot → 4 label tokens via shared recurrent block)
  + stream construction:
      [<SAMPLE>=0,  s_1..s_16,  <LABEL>=1,  l_1..l_4]      length T=22
  + output module fires at every position p in 0..21:
      pool last W=8 token embeddings (pad with <PAD>) → 11-way logits
  + supervised target per position:
      DIGIT(y) at p=17 (the <LABEL>-indicator slot), NO_OUTPUT (=10) elsewhere
  + loss: weighted CE — `mean(no_output) + λ_digit_boost · mean(digit)`

Eval: 10-way argmax over the digit slice of the 11-way logits at position 17.

Usage:
    uv run python projects/rosa/scripts/tmp/dense_baseline.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.random_utils import infinite_safe_keys_from_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Hyperparameters ───────────────────────────────────────────────────────────

DATASET             = "fashion_mnist"   # "mnist" or "fashion_mnist"
VOCAB_SIZE          = 1024
TOKEN_DIM           = 64        # D
HIDDEN              = 256
K_SAMPLE            = 16
K_LABEL             = 4
SIGMA               = 0.01      # DiVeQ noise
LAMBDA_CB           = 1.0
LAMBDA_DIGIT_BOOST  = 20.0      # counters NO_OUTPUT class imbalance
WINDOW              = 8         # output-module recent-window size W
BATCH_SIZE          = 128
N_EPOCHS            = 50
LR                  = 3e-4
SEED                = 42

# Reserved indicator IDs
ID_SAMPLE = 0
ID_LABEL  = 1
N_RESERVED = 2
N_CONTENT  = VOCAB_SIZE - N_RESERVED

# Stream geometry
T_STREAM       = 1 + K_SAMPLE + 1 + K_LABEL          # 22
LABEL_POS      = 1 + K_SAMPLE                         # 17 (where <LABEL> sits)
N_CLASSES_OUT  = 11                                   # 0..9 + NO_OUTPUT
NO_OUTPUT      = 10


# ── Params ────────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    p = {
        # Input projections
        "proj_image_W":   n((784, HIDDEN), 784 ** -0.5),
        "proj_image_b":   jnp.zeros(HIDDEN),
        "proj_label_W":   n((10, HIDDEN), 10 ** -0.5),
        "proj_label_b":   jnp.zeros(HIDDEN),

        # Shared recurrent block
        "rms_pre_gamma":  jnp.ones(HIDDEN),
        "block_W1":       n((HIDDEN, HIDDEN), HIDDEN ** -0.5),
        "block_b1":       jnp.zeros(HIDDEN),
        "block_W2":       n((HIDDEN, HIDDEN), HIDDEN ** -0.5),
        "block_b2":       jnp.zeros(HIDDEN),

        # Emit head
        "emit_rms_gamma": jnp.ones(HIDDEN),
        "emit_W":         n((HIDDEN, TOKEN_DIM), HIDDEN ** -0.5),
        "emit_b":         jnp.zeros(TOKEN_DIM),

        # Codebook (V × D). Indicator IDs 0,1 are masked out of argmin so they
        # never get assigned by the input module; their codebook rows still
        # serve as the indicator embeddings inserted into the stream.
        "codebook":       n((VOCAB_SIZE, TOKEN_DIM), TOKEN_DIM ** -0.5),

        # Pad embedding for output-module window padding at the start of a burst
        "pad_emb":        n((TOKEN_DIM,), TOKEN_DIM ** -0.5),

        # Output module: (W * D) → HIDDEN → 11
        "out_W1":         n((WINDOW * TOKEN_DIM, HIDDEN), (WINDOW * TOKEN_DIM) ** -0.5),
        "out_b1":         jnp.zeros(HIDDEN),
        "out_W2":         n((HIDDEN, N_CLASSES_OUT), HIDDEN ** -0.5),
        "out_b2":         jnp.zeros(N_CLASSES_OUT),
    }
    return p


def n_params(p) -> int:
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


# ── Forward ───────────────────────────────────────────────────────────────────

def rmsnorm(x, gamma, eps=1e-6):
    rms = jnp.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def _block_step(p, h):
    h_norm = rmsnorm(h, p["rms_pre_gamma"])
    h2 = jax.nn.gelu(h_norm @ p["block_W1"] + p["block_b1"])
    return h2 @ p["block_W2"] + p["block_b2"] + h


def _emit(p, h):
    return rmsnorm(h, p["emit_rms_gamma"]) @ p["emit_W"] + p["emit_b"]


def run_input_image(p, x):
    """x: (B, 784) → z: (B, K_SAMPLE, D)."""
    h = jax.nn.gelu(x @ p["proj_image_W"] + p["proj_image_b"])
    zs = []
    for _ in range(K_SAMPLE):
        h = _block_step(p, h)
        zs.append(_emit(p, h))
    return jnp.stack(zs, axis=1)


def run_input_label(p, y_onehot):
    """y_onehot: (B, 10) → z: (B, K_LABEL, D)."""
    h = jax.nn.gelu(y_onehot @ p["proj_label_W"] + p["proj_label_b"])
    zs = []
    for _ in range(K_LABEL):
        h = _block_step(p, h)
        zs.append(_emit(p, h))
    return jnp.stack(zs, axis=1)


def diveq_quantize(z, codebook, key):
    """z: (..., D), codebook: (V, D). Returns (z_q, ids) of shapes
    (..., D) and (...,)."""
    z_flat = z.reshape(-1, TOKEN_DIM)
    dists = ((z_flat[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    mask = jnp.zeros(VOCAB_SIZE).at[:N_RESERVED].set(jnp.inf)
    dists_masked = dists + mask[None, :]
    ids_flat = jnp.argmin(dists_masked, axis=-1)
    c_star = codebook[ids_flat]

    d_tilde = jax.lax.stop_gradient(c_star - z_flat)
    v       = jax.random.normal(key, z_flat.shape) * SIGMA
    vd      = v + d_tilde
    direction = jax.lax.stop_gradient(
        vd / (jnp.linalg.norm(vd, axis=-1, keepdims=True) + 1e-8)
    )
    magnitude = jnp.linalg.norm(c_star - z_flat, axis=-1, keepdims=True)
    z_q_flat = z_flat + magnitude * direction

    return z_q_flat.reshape(z.shape), ids_flat.reshape(z.shape[:-1])


def build_stream_emb(p, c_q_sample, c_q_label):
    """c_q_sample: (B, K_SAMPLE, D), c_q_label: (B, K_LABEL, D).
    Returns (B, T=22, D) sequence of token embeddings, with indicator slots
    filled from codebook[0] and codebook[1]."""
    B = c_q_sample.shape[0]
    sample_emb = jnp.broadcast_to(p["codebook"][ID_SAMPLE][None, None, :],
                                  (B, 1, TOKEN_DIM))
    label_emb  = jnp.broadcast_to(p["codebook"][ID_LABEL][None, None, :],
                                  (B, 1, TOKEN_DIM))
    return jnp.concatenate([sample_emb, c_q_sample, label_emb, c_q_label], axis=1)


def output_module(p, stream_emb):
    """stream_emb: (B, T, D) → logits: (B, T, 11) via sliding-window pool."""
    B, T, D = stream_emb.shape
    pad = jnp.broadcast_to(p["pad_emb"][None, None, :], (B, WINDOW - 1, D))
    padded = jnp.concatenate([pad, stream_emb], axis=1)        # (B, T+W-1, D)
    # windows[p] = padded[:, p:p+W, :]
    windows = jnp.stack([padded[:, t:t + WINDOW, :] for t in range(T)], axis=1)
    # → (B, T, W, D)
    feats = windows.reshape(B, T, WINDOW * D)
    h = jax.nn.gelu(feats @ p["out_W1"] + p["out_b1"])
    return h @ p["out_W2"] + p["out_b2"]                        # (B, T, 11)


# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(p, x_batch, y_batch, key):
    B = x_batch.shape[0]
    y_oh = jax.nn.one_hot(y_batch, 10)

    # Input module forward
    z_sample = run_input_image(p, x_batch)                      # (B, 16, D)
    z_label  = run_input_label(p, y_oh)                         # (B, 4, D)

    # Quantize
    k1, k2 = jax.random.split(key, 2)
    c_q_sample, ids_sample = diveq_quantize(z_sample, p["codebook"], k1)
    c_q_label,  ids_label  = diveq_quantize(z_label,  p["codebook"], k2)

    # Build stream and run output module
    stream_emb = build_stream_emb(p, c_q_sample, c_q_label)     # (B, T, D)
    logits = output_module(p, stream_emb)                       # (B, T, 11)

    # Targets: NO_OUTPUT everywhere except position LABEL_POS -> digit
    targets = jnp.full((B, T_STREAM), NO_OUTPUT, dtype=jnp.int32)
    targets = targets.at[:, LABEL_POS].set(y_batch)

    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)  # (B, T)

    # Weighted: mean(no_output) + λ * mean(digit)
    digit_mask = jnp.zeros(T_STREAM).at[LABEL_POS].set(1.0)
    no_mask    = 1.0 - digit_mask
    loss_no    = (ce * no_mask).sum(axis=1) / no_mask.sum()
    loss_dg    = (ce * digit_mask).sum(axis=1) / digit_mask.sum()
    cls = loss_no.mean() + LAMBDA_DIGIT_BOOST * loss_dg.mean()

    # Codebook loss (sample + label combined)
    z_all  = jnp.concatenate([z_sample, z_label], axis=1)
    ids_all = jnp.concatenate([ids_sample, ids_label], axis=1)
    c_star = p["codebook"][ids_all]
    cb = ((jax.lax.stop_gradient(z_all) - c_star) ** 2).mean()

    aux = (cls, loss_no.mean(), loss_dg.mean(), cb, ids_sample, ids_label, logits)
    return cls + LAMBDA_CB * cb, aux


# ── Step ──────────────────────────────────────────────────────────────────────

def make_step_fn(optimizer):
    @jax.jit
    def step(p, opt_state, x_batch, y_batch, key):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, x_batch, y_batch, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        cls, l_no, l_dg, cb, ids_s, ids_l, logits = aux
        return p, opt_state, loss, cls, l_no, l_dg, cb
    return step


# ── Eval ──────────────────────────────────────────────────────────────────────

@jax.jit
def _eval_batch(p, x_batch, y_batch):
    B = x_batch.shape[0]
    y_oh = jax.nn.one_hot(y_batch, 10)
    z_sample = run_input_image(p, x_batch)
    z_label  = run_input_label(p, y_oh)
    c_q_sample, ids_s = diveq_quantize(z_sample, p["codebook"], jnp.zeros(2, dtype=jnp.uint32))
    c_q_label,  ids_l = diveq_quantize(z_label,  p["codebook"], jnp.zeros(2, dtype=jnp.uint32))
    stream_emb = build_stream_emb(p, c_q_sample, c_q_label)
    logits = output_module(p, stream_emb)                       # (B, T, 11)

    digit_logits = logits[:, LABEL_POS, :10]                    # 10-way at LABEL_POS
    preds = jnp.argmax(digit_logits, axis=-1)                   # (B,)

    # Diagnostic: fire_rate and fire_at_label_rate
    pred_classes = jnp.argmax(logits, axis=-1)                  # (B, T) values in 0..10
    fire_mask = pred_classes != NO_OUTPUT                       # (B, T)
    fire_rate = fire_mask.mean()
    pos_mask = jnp.zeros(T_STREAM, dtype=jnp.bool_).at[LABEL_POS].set(True)
    fire_at_label = (fire_mask & pos_mask[None, :]).sum() / fire_mask.sum().clip(min=1)

    return preds, ids_s, ids_l, fire_rate, fire_at_label


def eval_metrics(p, X, y):
    preds_all = []
    ids_s_all, ids_l_all = [], []
    fr_sum, fr_lab_sum, n_batches = 0.0, 0.0, 0
    for i in range(0, len(X), 256):
        bx = jnp.array(X[i:i + 256])
        by = jnp.array(y[i:i + 256])
        preds, ids_s, ids_l, fr, fr_lab = _eval_batch(p, bx, by)
        preds_all.append(np.array(preds))
        ids_s_all.append(np.array(ids_s)); ids_l_all.append(np.array(ids_l))
        fr_sum += float(fr); fr_lab_sum += float(fr_lab); n_batches += 1
    preds = np.concatenate(preds_all)
    ids_s = np.concatenate(ids_s_all)
    ids_l = np.concatenate(ids_l_all)
    acc = float((preds == np.array(y)).mean())
    cb_used = int(len(np.unique(np.concatenate([ids_s.flatten(), ids_l.flatten()]))))
    return acc, cb_used, fr_sum / n_batches, fr_lab_sum / n_batches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.info(f"JAX devices: {jax.devices()}")
    data = load_supervised_image(DATASET)
    X_train = np.array(data.X.reshape(data.n_samples,           784)) / 255.0
    X_test  = np.array(data.X_test.reshape(data.n_test_samples, 784)) / 255.0
    y_train = np.array(data.y, dtype=np.int32)
    y_test  = np.array(data.y_test, dtype=np.int32)
    logging.info(f"MNIST: train={X_train.shape}  test={X_test.shape}")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = init_params(key_gen)
    logging.info(
        f"params: {n_params(p):,}  V={VOCAB_SIZE}  D={TOKEN_DIM}  "
        f"K_s={K_SAMPLE} K_l={K_LABEL}  T={T_STREAM}  λ_digit={LAMBDA_DIGIT_BOOST}"
    )

    num_batches = len(X_train) // BATCH_SIZE
    schedule = optax.cosine_decay_schedule(LR, N_EPOCHS * num_batches, alpha=0.05)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(p)
    step_fn = make_step_fn(optimizer)

    rng = np.random.default_rng(SEED + 1)
    t0 = time.perf_counter()
    history = {"loss": [], "cls": [], "l_no": [], "l_dg": [], "cb": [],
               "test_acc": [], "cb_used": [], "fire_rate": [], "fire_at_label": []}

    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs = X_train[perm]; ys = y_train[perm]
        ep_loss = ep_cls = ep_no = ep_dg = ep_cb = 0.0
        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx = jnp.array(Xs[s:e]); by = jnp.array(ys[s:e])
            key = next(key_gen).get()
            p, opt_state, loss, cls, l_no, l_dg, cb = step_fn(p, opt_state, bx, by, key)
            ep_loss += float(loss); ep_cls += float(cls); ep_no += float(l_no)
            ep_dg += float(l_dg);   ep_cb += float(cb)

        for k, v in [("loss", ep_loss), ("cls", ep_cls), ("l_no", ep_no),
                     ("l_dg", ep_dg), ("cb", ep_cb)]:
            history[k].append(v / num_batches)

        test_acc, cb_used, fr, fr_lab = eval_metrics(p, X_test, y_test)
        history["test_acc"].append(test_acc)
        history["cb_used"].append(cb_used)
        history["fire_rate"].append(fr)
        history["fire_at_label"].append(fr_lab)

        logging.info(
            f"epoch {epoch:2d}  cls={history['cls'][-1]:.4f} "
            f"(no={history['l_no'][-1]:.4f} dg={history['l_dg'][-1]:.4f}) "
            f"cb={history['cb'][-1]:.4f}  "
            f"test={test_acc:.4f}  cb_used={cb_used}/{N_CONTENT}  "
            f"fire={fr:.3f} fire_at_lbl={fr_lab:.3f}  "
            f"{time.perf_counter() - t0:.1f}s"
        )

    elapsed = time.perf_counter() - t0
    final_acc, cb_used, fr, fr_lab = eval_metrics(p, X_test, y_test)
    train_acc, _, _, _ = eval_metrics(p, X_train[:10_000], y_train[:10_000])

    logging.info(
        f"\nFINAL  test={final_acc:.4f}  train(10k)={train_acc:.4f}  "
        f"cb_used={cb_used}/{N_CONTENT}  fire_rate={fr:.3f}  "
        f"fire_at_lbl_rate={fr_lab:.3f}  time={elapsed:.0f}s"
    )

    out = Path(__file__).parent / f"dense_baseline_result.{DATASET}.json"
    with open(out, "w") as f:
        json.dump({
            "final_test_acc": final_acc,
            "final_train_acc": train_acc,
            "cb_used": cb_used,
            "final_fire_rate": fr,
            "final_fire_at_label": fr_lab,
            "n_params": n_params(p),
            "epochs": N_EPOCHS,
            "time_s": elapsed,
            "history": history,
            "hp": {
                "VOCAB_SIZE": VOCAB_SIZE, "TOKEN_DIM": TOKEN_DIM,
                "HIDDEN": HIDDEN, "K_SAMPLE": K_SAMPLE, "K_LABEL": K_LABEL,
                "T_STREAM": T_STREAM, "LABEL_POS": LABEL_POS,
                "WINDOW": WINDOW, "SIGMA": SIGMA, "LAMBDA_CB": LAMBDA_CB,
                "LAMBDA_DIGIT_BOOST": LAMBDA_DIGIT_BOOST,
                "BATCH_SIZE": BATCH_SIZE, "LR": LR, "N_EPOCHS": N_EPOCHS,
            },
        }, f, indent=2)
    logging.info(f"Result saved to {out}")


if __name__ == "__main__":
    main()
