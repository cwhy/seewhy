"""
Phase A — decoupled VQ-AE → ROSA → zero-param head on Fashion-MNIST.

The architecture is structured so that ROSA IS the classifier:

  Stage 1: train recurrent input MLP + codebook + simple decoder as a VQ-AE
           on images only (no labels, pure self-supervised tokenization).
  Stage 2: freeze the encoder. Build per-sample integer streams of the form
              [<SAMPLE>=0, s_1..s_16, <LABEL>=1, y_token=y+2]
           feed all 60K streams to ROSA (SAM grows). Build Witten-Bell LM.
  Stage 3: for each test sample, encode → 16 tokens → walk SAM to the state
           after `[<SAMPLE>, s_1..s_16, <LABEL>]`. Read ROSA's distribution
           there, sum probability mass over digit IDs (2..11), argmax →
           predicted digit. Zero-parameter output head.

Reserved IDs (V=1024):
    0     <SAMPLE>
    1     <LABEL>
    2..11 digit-token slots (digit y → ID y+2)
    12..  content image-token slots

Usage: uv run python projects/rosa/scripts/tmp/decoupled_rosa.py
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

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))
sys.path.insert(0, str(PROJ_DIR))

from shared_lib.datasets import load_supervised_image                  # noqa: E402
from shared_lib.random_utils import infinite_safe_keys_from_key        # noqa: E402
from lib.rosa_core import ROSACore, _advance_state, _predict_deterministic   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Hyperparameters ───────────────────────────────────────────────────────────

DATASET     = "fashion_mnist"

VOCAB_SIZE  = 1024
TOKEN_DIM   = 64
HIDDEN      = 256
K_SAMPLE    = 16
SIGMA       = 0.01
LAMBDA_CB   = 1.0

BATCH_SIZE  = 128
N_EPOCHS    = 50
LR          = 3e-4
SEED        = 42

# Reserved ID layout
ID_SAMPLE     = 0
ID_LABEL      = 1
N_DIGITS      = 10
DIGIT_OFFSET  = 2                                 # digit y → ID y+2  → IDs 2..11
N_RESERVED    = 2 + N_DIGITS                       # 12 total reserved slots
N_CONTENT     = VOCAB_SIZE - N_RESERVED            # 1012 image-token slots

OUT = Path(__file__).parent / f"decoupled_rosa.{DATASET}.json"


# ── Stage 1: VQ-AE ────────────────────────────────────────────────────────────

def init_vqae_params(key_gen):
    def n(shape, scale):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    p = {
        # Encoder (recurrent, identical to experiments1.py's run_input_image)
        "proj_image_W":   n((784, HIDDEN), 784 ** -0.5),
        "proj_image_b":   jnp.zeros(HIDDEN),
        "rms_pre_gamma":  jnp.ones(HIDDEN),
        "block_W1":       n((HIDDEN, HIDDEN), HIDDEN ** -0.5),
        "block_b1":       jnp.zeros(HIDDEN),
        "block_W2":       n((HIDDEN, HIDDEN), HIDDEN ** -0.5),
        "block_b2":       jnp.zeros(HIDDEN),
        "emit_rms_gamma": jnp.ones(HIDDEN),
        "emit_W":         n((HIDDEN, TOKEN_DIM), HIDDEN ** -0.5),
        "emit_b":         jnp.zeros(TOKEN_DIM),

        # Codebook
        "codebook":       n((VOCAB_SIZE, TOKEN_DIM), TOKEN_DIM ** -0.5),

        # Decoder: K_SAMPLE * D → HIDDEN → 784
        "dec_W1":         n((K_SAMPLE * TOKEN_DIM, HIDDEN), (K_SAMPLE * TOKEN_DIM) ** -0.5),
        "dec_b1":         jnp.zeros(HIDDEN),
        "dec_W2":         n((HIDDEN, 784), HIDDEN ** -0.5),
        "dec_b2":         jnp.zeros(784),
    }
    return p


def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


def rmsnorm(x, gamma, eps=1e-6):
    rms = jnp.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def encode_image(p, x):
    """x: (B, 784) → z: (B, K_SAMPLE, D)."""
    h = jax.nn.gelu(x @ p["proj_image_W"] + p["proj_image_b"])
    zs = []
    for _ in range(K_SAMPLE):
        h_norm = rmsnorm(h, p["rms_pre_gamma"])
        h2 = jax.nn.gelu(h_norm @ p["block_W1"] + p["block_b1"])
        h  = h2 @ p["block_W2"] + p["block_b2"] + h
        zs.append(rmsnorm(h, p["emit_rms_gamma"]) @ p["emit_W"] + p["emit_b"])
    return jnp.stack(zs, axis=1)


def diveq_quantize(z, codebook, key):
    """z: (..., D), codebook: (V, D). Returns (z_q, ids)."""
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


def decode(p, z_q):
    """z_q: (B, K_SAMPLE, D) → x_recon: (B, 784) ∈ [0,1]."""
    h = z_q.reshape(z_q.shape[0], -1)                                  # (B, K*D)
    h = jax.nn.gelu(h @ p["dec_W1"] + p["dec_b1"])
    return jax.nn.sigmoid(h @ p["dec_W2"] + p["dec_b2"])


def vqae_loss(p, x_batch, key):
    z = encode_image(p, x_batch)
    z_q, ids = diveq_quantize(z, p["codebook"], key)
    x_recon = decode(p, z_q)
    recon = ((x_recon - x_batch) ** 2).mean()
    c_star = p["codebook"][ids]
    cb = ((jax.lax.stop_gradient(z) - c_star) ** 2).mean()
    return recon + LAMBDA_CB * cb, (recon, cb, ids)


def train_vqae(X_train, X_test):
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = init_vqae_params(key_gen)
    logging.info(f"VQ-AE params: {n_params(p):,}")

    num_batches = len(X_train) // BATCH_SIZE
    schedule = optax.cosine_decay_schedule(LR, N_EPOCHS * num_batches, alpha=0.05)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(p)

    @jax.jit
    def step(p, opt_state, x_batch, key):
        (loss, aux), grads = jax.value_and_grad(vqae_loss, has_aux=True)(p, x_batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss, *aux

    @jax.jit
    def eval_recon(p, x_batch):
        z = encode_image(p, x_batch)
        z_q, ids = diveq_quantize(z, p["codebook"], jnp.zeros(2, dtype=jnp.uint32))
        x_recon = decode(p, z_q)
        return ((x_recon - x_batch) ** 2).mean(), ids

    rng = np.random.default_rng(SEED + 1)
    t0 = time.perf_counter()
    history = {"loss": [], "recon": [], "cb": [], "test_recon": [], "cb_used": []}

    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs = X_train[perm]
        ep_loss = ep_recon = ep_cb = 0.0
        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx = jnp.array(Xs[s:e])
            key = next(key_gen).get()
            p, opt_state, loss, recon, cb, _ = step(p, opt_state, bx, key)
            ep_loss += float(loss); ep_recon += float(recon); ep_cb += float(cb)
        ep_loss /= num_batches; ep_recon /= num_batches; ep_cb /= num_batches

        # Eval on test
        test_recons = []
        all_ids = []
        for i in range(0, len(X_test), 256):
            r, ids = eval_recon(p, jnp.array(X_test[i:i + 256]))
            test_recons.append(float(r))
            all_ids.append(np.array(ids))
        test_recon = float(np.mean(test_recons))
        all_ids = np.concatenate(all_ids)
        cb_used = int(len(np.unique(all_ids)))

        history["loss"].append(ep_loss); history["recon"].append(ep_recon)
        history["cb"].append(ep_cb); history["test_recon"].append(test_recon)
        history["cb_used"].append(cb_used)

        logging.info(
            f"  vqae ep {epoch:2d}  loss={ep_loss:.4f}  recon={ep_recon:.4f}  "
            f"cb={ep_cb:.4f}  test_recon={test_recon:.4f}  cb_used={cb_used}/{N_CONTENT}  "
            f"{time.perf_counter() - t0:.1f}s"
        )

    return p, history, time.perf_counter() - t0


# ── Stage 2: build ROSA ───────────────────────────────────────────────────────

@jax.jit
def encode_to_ids(p, x_batch):
    z = encode_image(p, x_batch)
    _, ids = diveq_quantize(z, p["codebook"], jnp.zeros(2, dtype=jnp.uint32))
    return ids


def build_rosa_streams(p, X, y, batch_size: int = 256):
    """Return list[list[int]] of length len(X), each = [<SAMPLE>, s_1..s_16, <LABEL>, y_token]."""
    streams = []
    for i in range(0, len(X), batch_size):
        bx = jnp.array(X[i:i + batch_size])
        ids = np.array(encode_to_ids(p, bx))                 # (B, K_SAMPLE)
        for j in range(ids.shape[0]):
            s = [ID_SAMPLE]
            s.extend(int(t) for t in ids[j])
            s.append(ID_LABEL)
            s.append(int(y[i + j]) + DIGIT_OFFSET)
            streams.append(s)
    return streams


def build_rosa_from_data(p, X_train, y_train):
    rosa = ROSACore(vocab_size=VOCAB_SIZE, max_order=64)

    logging.info("Stage 2: encoding training set to streams...")
    t0 = time.perf_counter()
    streams = build_rosa_streams(p, X_train, y_train)
    logging.info(f"  built {len(streams)} streams in {time.perf_counter()-t0:.1f}s")

    logging.info("Stage 2: feeding ROSA SAM (commit_burst per stream)...")
    t0 = time.perf_counter()
    for s in streams:
        rosa.commit_burst(s)
    logging.info(f"  SAM has {rosa.n_states():,} states ({time.perf_counter()-t0:.1f}s)")

    logging.info("Stage 2: building Witten-Bell fallback LM...")
    t0 = time.perf_counter()
    rosa.build_lm(streams)
    logging.info(f"  LM built ({time.perf_counter()-t0:.1f}s)")

    return rosa, streams


# ── Stage 3: zero-parameter eval ──────────────────────────────────────────────

USE_DETERMINISTIC_SHORTCUT = False  # if False, always use LM distribution


def predict_digit_from_rosa(rosa: ROSACore, sample_ids):
    """Walk the ROSA SAM along [<SAMPLE>, s_1..s_K, <LABEL>], read the LM
    distribution at the resulting state, sum probability mass over the
    digit-token IDs (2..11), and return argmax over those 10."""
    sam = rosa.sam
    b, c = sam.b, sam.c
    v = 0
    for tok in [ID_SAMPLE] + [int(t) for t in sample_ids] + [ID_LABEL]:
        v = _advance_state(b, c, v, tok)

    if USE_DETERMINISTIC_SHORTCUT:
        det = _predict_deterministic(sam.b, sam.c, sam.d, sam.e,
                                     sam.text, v, sam.boundary_after)
        if det is not None and DIGIT_OFFSET <= det < DIGIT_OFFSET + N_DIGITS:
            return int(det) - DIGIT_OFFSET, "sam"

    if rosa.lm is None:
        return -1, "no_lm"
    dist = rosa.lm.probs_for_state(v)
    digit_probs = [dist.get(DIGIT_OFFSET + d, 0.0) for d in range(N_DIGITS)]
    if sum(digit_probs) == 0:
        return -1, "no_signal"
    return max(range(N_DIGITS), key=lambda d: digit_probs[d]), "lm"


def eval_decoupled(p, rosa: ROSACore, X_test, y_test, batch_size: int = 256):
    correct = 0
    src_counts = {"sam": 0, "lm": 0, "no_lm": 0, "no_signal": 0}
    t0 = time.perf_counter()

    for i in range(0, len(X_test), batch_size):
        bx = jnp.array(X_test[i:i + batch_size])
        ids = np.array(encode_to_ids(p, bx))
        for j in range(ids.shape[0]):
            pred, src = predict_digit_from_rosa(rosa, ids[j])
            if pred == int(y_test[i + j]):
                correct += 1
            src_counts[src] = src_counts.get(src, 0) + 1

    acc = correct / len(X_test)
    logging.info(
        f"  eval done in {time.perf_counter()-t0:.1f}s  "
        f"sources: {src_counts}"
    )
    return acc, src_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.info(f"JAX devices: {jax.devices()}")
    data = load_supervised_image(DATASET)
    X_train = np.array(data.X.reshape(data.n_samples,           784)) / 255.0
    X_test  = np.array(data.X_test.reshape(data.n_test_samples, 784)) / 255.0
    y_train = np.array(data.y, dtype=np.int32)
    y_test  = np.array(data.y_test, dtype=np.int32)
    logging.info(f"{DATASET}: train={X_train.shape}  test={X_test.shape}")

    # ── Stage 1
    logging.info("\n=== Stage 1: VQ-AE training ===")
    t1 = time.perf_counter()
    p, vqae_history, vqae_time = train_vqae(X_train, X_test)
    logging.info(f"Stage 1 done in {vqae_time:.1f}s. Final test_recon={vqae_history['test_recon'][-1]:.4f}")

    # ── Stage 2
    logging.info("\n=== Stage 2: build ROSA over training streams ===")
    t2 = time.perf_counter()
    rosa, streams = build_rosa_from_data(p, X_train, y_train)
    s2_time = time.perf_counter() - t2
    logging.info(f"Stage 2 done in {s2_time:.1f}s. SAM states={rosa.n_states():,}")

    # ── Stage 3
    logging.info("\n=== Stage 3: zero-param decoupled eval ===")
    t3 = time.perf_counter()
    test_acc, src_counts = eval_decoupled(p, rosa, X_test, y_test)
    s3_time = time.perf_counter() - t3
    logging.info(f"Stage 3 done in {s3_time:.1f}s")

    # Also do train-set eval (decoupled)
    t4 = time.perf_counter()
    train_acc, train_src = eval_decoupled(p, rosa, X_train[:10_000], y_train[:10_000])
    s4_time = time.perf_counter() - t4

    total = time.perf_counter() - t1
    logging.info(
        f"\n=== Summary (decoupled, Fashion-MNIST) ===\n"
        f"  Stage 1 (VQ-AE):     {vqae_time:.0f}s  test_recon={vqae_history['test_recon'][-1]:.4f}\n"
        f"  Stage 2 (ROSA):      {s2_time:.0f}s    sam_states={rosa.n_states():,}\n"
        f"  Stage 3 (eval test): {s3_time:.0f}s    test_acc={test_acc:.4f}  src={src_counts}\n"
        f"  Stage 3 (eval train):{s4_time:.0f}s    train_acc={train_acc:.4f}  src={train_src}\n"
        f"  TOTAL:               {total:.0f}s\n"
    )

    with open(OUT, "w") as f:
        json.dump({
            "test_acc":    test_acc,
            "train_acc":   train_acc,
            "src_test":    src_counts,
            "src_train":   train_src,
            "sam_states":  rosa.n_states(),
            "vqae_test_recon": vqae_history["test_recon"][-1],
            "vqae_history": vqae_history,
            "stage_times":  {"vqae": vqae_time, "rosa_build": s2_time,
                             "eval_test": s3_time, "eval_train": s4_time},
            "hp": {
                "DATASET": DATASET, "VOCAB_SIZE": VOCAB_SIZE,
                "TOKEN_DIM": TOKEN_DIM, "HIDDEN": HIDDEN,
                "K_SAMPLE": K_SAMPLE, "SIGMA": SIGMA,
                "BATCH_SIZE": BATCH_SIZE, "N_EPOCHS": N_EPOCHS, "LR": LR,
                "N_RESERVED": N_RESERVED, "DIGIT_OFFSET": DIGIT_OFFSET,
            },
        }, f, indent=2)
    logging.info(f"Result saved to {OUT}")


if __name__ == "__main__":
    main()
