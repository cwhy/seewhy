"""
Universal AR — exp1: in-context MNIST completion via an auto-associative
role⊗filler memory.

Hypothesis
==========
Treat each pixel as a record {value, pos, ref}. Bind role⊗filler with HRR
(circular convolution) and superpose a sample's observed pixels into one pattern
    P_s = Σ_{observed pos}  r_pos[pos] ⊛ val_emb[bin(pixel)]        (+ label bind)
Store patterns auto-associatively (symmetric); complete a *masked field* of a
query sample by reading the OTHER samples' patterns and unbinding the queried
role. The class label is just a field at r_label — so classification is
completion of a masked label field.

Why leave-one-out
=================
Predicting a withheld label is only possible by *cross-sample* generalization
(the random `ref` deliberately isolates a sample's own stored content, so it
cannot supply an unseen label). So each sample's label is completed from the
*other* S-1 samples in the episode via pixel-content similarity — an
auto-associative read that reduces to a learnable kernel / few-shot classifier.

Both readouts are computed (concepts.md §5a):
    linear   (Hebbian):    p̂ = Σ_{s≠q} (cue·key_s)            · value_s
    attention(Hopfield):   p̂ = Σ_{s≠q} softmax(β cue·key_s)   · value_s
Training uses one (READ_MODE); eval reports both.

Usage:
    uv run python projects/universal-ar/scripts/run_experiments.py --bg exp1
"""

import json
import time
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import Array

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp1"
JSONL = Path(__file__).parent / "results.jsonl"

# ── hyperparameters ───────────────────────────────────────────────────────────
DATASET        = "mnist"
D              = 256      # binding / embedding dim (HRR vectors live in R^D)
N_VAL_BINS     = 32       # pixel intensity quantisation (0-255 → K bins)
N_CLASSES      = 10
POS_DIM        = 784      # 28×28 pixel positions

EP_SAMPLES     = 32       # S: samples per episode (the in-context set)
EP_PIXELS      = 160      # P_obs: observed pixels per sample (partial sample)
BATCH_EPISODES = 8        # B: episodes per gradient step

READ_MODE      = "attn"   # "attn" | "linear" — read used for the training loss
NUM_STEPS      = 4000
EVAL_EVERY     = 200
EVAL_EPISODES  = 64       # episodes averaged for a test metric
LEARNING_RATE  = 1e-3
RANDOM_SEED    = 0


# ── HRR primitives (circular convolution via FFT) ─────────────────────────────
def fft(x: Array) -> Array:
    return jnp.fft.fft(x, axis=-1)


def bind(role: Array, filler: Array) -> Array:
    """HRR bind: circular convolution. role, filler: (..., D) → (..., D)."""
    return jnp.fft.ifft(fft(role) * fft(filler), axis=-1).real


def unbind(bound: Array, role: Array) -> Array:
    """HRR unbind: circular correlation with role (recovers the filler)."""
    return jnp.fft.ifft(fft(bound) * jnp.conj(fft(role)), axis=-1).real


# ── params ────────────────────────────────────────────────────────────────────
def init() -> Tuple[Dict[str, Any], Dict[str, Array]]:
    config = {
        "dataset": DATASET, "D": D, "n_val_bins": N_VAL_BINS,
        "ep_samples": EP_SAMPLES, "ep_pixels": EP_PIXELS,
        "batch_episodes": BATCH_EPISODES, "read_mode": READ_MODE,
        "num_steps": NUM_STEPS, "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
    }
    keys = jax.random.split(jax.random.PRNGKey(RANDOM_SEED), 5)
    s = 1.0 / D ** 0.5
    params = {
        "r_pos":     jax.random.normal(keys[0], (POS_DIM, D)) * s,   # position roles
        "val_emb":   jax.random.normal(keys[1], (N_VAL_BINS, D)) * s,  # value fillers
        "r_label":   jax.random.normal(keys[2], (D,)) * s,            # label role
        "label_emb": jax.random.normal(keys[3], (N_CLASSES, D)) * s,  # label fillers
        "log_beta":  jnp.array(np.log(20.0), dtype=jnp.float32),      # attn temperature
    }
    return config, params


def n_params(params: Dict[str, Array]) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))


# ── episode sampling (host side) ──────────────────────────────────────────────
def sample_episodes(Xflat: np.ndarray, y: np.ndarray, B: int, S: int, P: int,
                    rng: np.random.Generator):
    """Return bins (B,S,P) int32, pos (B,S,P) int32, labels (B,S) int32."""
    N = Xflat.shape[0]
    idx = rng.integers(0, N, size=(B, S))
    imgs = Xflat[idx]                                   # (B,S,784)
    labels = y[idx].astype(np.int32)                    # (B,S)
    pos = rng.random((B, S, POS_DIM)).argsort(-1)[..., :P].astype(np.int32)  # (B,S,P)
    vals = np.take_along_axis(imgs, pos, axis=-1)       # (B,S,P) 0-255
    bins = np.floor(vals / 255.0 * (N_VAL_BINS - 1)).astype(np.int32)
    return jnp.asarray(bins), jnp.asarray(pos), jnp.asarray(labels)


# ── model ─────────────────────────────────────────────────────────────────────
def build_patterns(params, bins, pos):
    """Superpose observed pixels into a per-sample pattern.  → pixel_pattern (B,S,D)."""
    F_pos = fft(params["r_pos"])[pos]        # (B,S,P,D) complex — role of each position
    F_val = fft(params["val_emb"])[bins]     # (B,S,P,D) complex — filler of each value
    freq = jnp.sum(F_pos * F_val, axis=2)    # Σ_p bind(r_pos, val)  in freq domain (B,S,D)
    return jnp.fft.ifft(freq, axis=-1).real  # (B,S,D)


def stored_value(params, pixel_pattern, labels):
    """Value written to memory = pixels + bound label (support content)."""
    lab_bound = bind(params["r_label"], params["label_emb"][labels])   # (B,S,D)
    return pixel_pattern + lab_bound


def classify(params, pixel_pattern, value, mode: str) -> Array:
    """Leave-one-out read + unbind label → class logits (B,S,N_CLASSES)."""
    B, S, _ = pixel_pattern.shape
    key = pixel_pattern / (jnp.linalg.norm(pixel_pattern, axis=-1, keepdims=True) + 1e-6)
    sim = jnp.einsum("bqd,bkd->bqk", key, key)     # cosine(query pixels, memory pixels)
    off = 1.0 - jnp.eye(S)[None]                    # leave-one-out mask (drop self)

    if mode == "attn":
        beta = jnp.exp(params["log_beta"])
        scores = beta * sim - 1e9 * jnp.eye(S)[None]
        w = jax.nn.softmax(scores, axis=-1)
    else:  # linear (Hebbian) read
        w = sim * off

    p_hat = jnp.einsum("bqk,bkd->bqd", w, value)   # retrieved full pattern
    f_label = unbind(p_hat, params["r_label"])     # recover label filler
    return jnp.einsum("bqd,cd->bqc", f_label, params["label_emb"])


def loss_fn(params, bins, pos, labels, mode: str):
    pix = build_patterns(params, bins, pos)
    val = stored_value(params, pix, labels)
    logits = classify(params, pix, val, mode)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, acc


@partial(jax.jit, static_argnums=(0, 5))
def train_step(optimizer, params, opt_state, bins, pos, labels, mode):
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, bins, pos, labels, mode)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, acc


@jax.jit
def eval_both(params, bins, pos, labels):
    pix = build_patterns(params, bins, pos)
    val = stored_value(params, pix, labels)
    acc_attn = jnp.mean(jnp.argmax(classify(params, pix, val, "attn"), -1) == labels)
    acc_lin = jnp.mean(jnp.argmax(classify(params, pix, val, "linear"), -1) == labels)
    return acc_attn, acc_lin


def evaluate(params, Xflat, y, rng) -> Tuple[float, float]:
    accs_a, accs_l = [], []
    for _ in range(EVAL_EPISODES // BATCH_EPISODES):
        b, p, l = sample_episodes(Xflat, y, BATCH_EPISODES, EP_SAMPLES, EP_PIXELS, rng)
        a, ll = eval_both(params, b, p, l)
        accs_a.append(float(a)); accs_l.append(float(ll))
    return float(np.mean(accs_a)), float(np.mean(accs_l))


# ── train ─────────────────────────────────────────────────────────────────────
def train(config, params, Xtr, ytr, Xte, yte):
    schedule = optax.cosine_decay_schedule(LEARNING_RATE, NUM_STEPS)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)

    rng = np.random.default_rng(RANDOM_SEED)
    eval_rng = np.random.default_rng(RANDOM_SEED + 1)
    hist = {"step": [], "loss": [], "train_acc": [], "test_acc_attn": [], "test_acc_lin": []}
    t0 = time.perf_counter()

    for step in range(1, NUM_STEPS + 1):
        b, p, l = sample_episodes(Xtr, ytr, BATCH_EPISODES, EP_SAMPLES, EP_PIXELS, rng)
        params, opt_state, loss, acc = train_step(
            optimizer, params, opt_state, b, p, l, READ_MODE)

        if step % EVAL_EVERY == 0 or step == 1:
            ta, tl = evaluate(params, Xte, yte, eval_rng)
            hist["step"].append(step); hist["loss"].append(float(loss))
            hist["train_acc"].append(float(acc))
            hist["test_acc_attn"].append(ta); hist["test_acc_lin"].append(tl)
            logging.info(
                f"step {step:5d}  loss {float(loss):.4f}  train_acc {float(acc):.4f}  "
                f"test_acc[attn] {ta:.4f}  test_acc[lin] {tl:.4f}  "
                f"elapsed {time.perf_counter()-t0:.0f}s")

    return params, hist, time.perf_counter() - t0


if __name__ == "__main__":
    # skip-if-done
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping"); raise SystemExit(0)

    config, params = init()
    logging.info(f"Loading {DATASET}…")
    data = load_supervised_image(DATASET)
    Xtr = np.asarray(data.X.reshape(data.n_samples, -1), dtype=np.float32)
    Xte = np.asarray(data.X_test.reshape(data.n_test_samples, -1), dtype=np.float32)
    ytr = np.asarray(data.y); yte = np.asarray(data.y_test)
    logging.info(f"train {Xtr.shape}  test {Xte.shape}  n_params {n_params(params)}")
    logging.info(f"episode: S={EP_SAMPLES} samples × P={EP_PIXELS} pixels, "
                 f"B={BATCH_EPISODES} episodes/step, read_mode={READ_MODE}")

    params, hist, elapsed = train(config, params, Xtr, ytr, Xte, yte)

    final_attn = hist["test_acc_attn"][-1]
    final_lin = hist["test_acc_lin"][-1]
    logging.info(f"DONE  test_acc[attn]={final_attn:.4f}  test_acc[lin]={final_lin:.4f}  "
                 f"{elapsed:.0f}s")

    row = {
        "experiment": EXP_NAME,
        "name": "in-context MNIST label completion, auto-associative role⊗filler HRR memory",
        "time_s": elapsed,
        "n_params": n_params(params),
        "test_acc_attn": final_attn,
        "test_acc_lin": final_lin,
        **{k: config[k] for k in config},
        "history": hist,
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended result → {JSONL}")
