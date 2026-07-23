"""
exp1 — ROSA + DiVeQ + dense-fire headline (fixed K, no ACT).

Hypothesis: a continual ROSA SAM, fed all training tokens, provides a
predicted-next-token feature that lets the dense-fire output module exceed
the no-ROSA dense baseline (0.4 = 98.10%) by ≥0.5pp on MNIST.

Architecture: same as scripts/tmp/dense_baseline.py except the output module
also consumes ROSA's argmax-predicted-token embedding at every position
(stop_grad). Stream is `<SAMPLE> s_1..s_16 <LABEL> l_1..l_4` (T=22 tokens).
ROSA grows monotonically through training; frozen at eval.

Usage: uv run python projects/rosa/scripts/run_experiments.py exp1
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJ_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))
sys.path.insert(0, str(PROJ_DIR))

from shared_lib.datasets import load_supervised_image                 # noqa: E402
from shared_lib.random_utils import infinite_safe_keys_from_key       # noqa: E402
from lib.rosa_core import ROSACore                                    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Hyperparameters ───────────────────────────────────────────────────────────

DATASET             = "fashion_mnist"   # "mnist" or "fashion_mnist"
USE_INDICATORS      = True               # if False, drop <SAMPLE>/<LABEL> from stream
STOP_GRAD_ROSA      = False              # if False, gradient flows through codebook[predicted_ids]
_BASE               = "exp1" if STOP_GRAD_ROSA else "exp6"
_DS_TAG             = "" if DATASET == "mnist" else f"_{DATASET.replace('fashion_mnist', 'fmnist')}"
_IND_TAG            = "" if USE_INDICATORS else "n"
EXP_NAME            = f"{_BASE}{_IND_TAG}{_DS_TAG}"

VOCAB_SIZE          = 1024
TOKEN_DIM           = 64
HIDDEN              = 256
K_SAMPLE            = 16
K_LABEL             = 4
SIGMA               = 0.01
LAMBDA_CB           = 1.0
LAMBDA_DIGIT_BOOST  = 20.0
WINDOW              = 8
BATCH_SIZE          = 128
N_EPOCHS            = 50
LR                  = 3e-4
SEED                = 42

ID_SAMPLE = 0
ID_LABEL  = 1
N_RESERVED = 2
N_CONTENT  = VOCAB_SIZE - N_RESERVED

# Stream geometry depends on whether indicators are inserted into the stream.
# With indicators: [<SAMPLE>, s_1..s_K, <LABEL>, l_1..l_K'], length = K_S + K_L + 2.
# Without:        [s_1..s_K, l_1..l_K'],                     length = K_S + K_L.
# In both cases LABEL_POS is the position whose target is DIGIT(y); for the
# no-indicator variant this is the last sample-token slot (model must decide
# to fire there based on content alone).
if USE_INDICATORS:
    T_STREAM  = 1 + K_SAMPLE + 1 + K_LABEL          # 22
    LABEL_POS = 1 + K_SAMPLE                         # 17
else:
    T_STREAM  = K_SAMPLE + K_LABEL                  # 20
    LABEL_POS = K_SAMPLE - 1                        # 15 (just observed s_16)

N_CLASSES_OUT  = 11
NO_OUTPUT      = 10

JSONL = PROJ_DIR / "results.jsonl"


# ── Params ────────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    p = {
        "proj_image_W":   n((784, HIDDEN), 784 ** -0.5),
        "proj_image_b":   jnp.zeros(HIDDEN),
        "proj_label_W":   n((10, HIDDEN), 10 ** -0.5),
        "proj_label_b":   jnp.zeros(HIDDEN),

        "rms_pre_gamma":  jnp.ones(HIDDEN),
        "block_W1":       n((HIDDEN, HIDDEN), HIDDEN ** -0.5),
        "block_b1":       jnp.zeros(HIDDEN),
        "block_W2":       n((HIDDEN, HIDDEN), HIDDEN ** -0.5),
        "block_b2":       jnp.zeros(HIDDEN),

        "emit_rms_gamma": jnp.ones(HIDDEN),
        "emit_W":         n((HIDDEN, TOKEN_DIM), HIDDEN ** -0.5),
        "emit_b":         jnp.zeros(TOKEN_DIM),

        "codebook":       n((VOCAB_SIZE, TOKEN_DIM), TOKEN_DIM ** -0.5),
        "pad_emb":        n((TOKEN_DIM,), TOKEN_DIM ** -0.5),

        # Output: features dim = (W + 1) * D    [W context tokens + 1 ROSA token]
        "out_W1":         n(((WINDOW + 1) * TOKEN_DIM, HIDDEN),
                            ((WINDOW + 1) * TOKEN_DIM) ** -0.5),
        "out_b1":         jnp.zeros(HIDDEN),
        "out_W2":         n((HIDDEN, N_CLASSES_OUT), HIDDEN ** -0.5),
        "out_b2":         jnp.zeros(N_CLASSES_OUT),
    }
    return p


def n_params(p) -> int:
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


# ── Forward primitives ────────────────────────────────────────────────────────

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
    h = jax.nn.gelu(x @ p["proj_image_W"] + p["proj_image_b"])
    zs = []
    for _ in range(K_SAMPLE):
        h = _block_step(p, h)
        zs.append(_emit(p, h))
    return jnp.stack(zs, axis=1)                              # (B, K_SAMPLE, D)


def run_input_label(p, y_oh):
    h = jax.nn.gelu(y_oh @ p["proj_label_W"] + p["proj_label_b"])
    zs = []
    for _ in range(K_LABEL):
        h = _block_step(p, h)
        zs.append(_emit(p, h))
    return jnp.stack(zs, axis=1)


def diveq_quantize(z, codebook, key):
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
    if USE_INDICATORS:
        B = c_q_sample.shape[0]
        sample_emb = jnp.broadcast_to(p["codebook"][ID_SAMPLE][None, None, :],
                                      (B, 1, TOKEN_DIM))
        label_emb  = jnp.broadcast_to(p["codebook"][ID_LABEL][None, None, :],
                                      (B, 1, TOKEN_DIM))
        return jnp.concatenate([sample_emb, c_q_sample, label_emb, c_q_label],
                               axis=1)
    return jnp.concatenate([c_q_sample, c_q_label], axis=1)


def output_module(p, stream_emb, rosa_pred_emb):
    """stream_emb: (B, T, D), rosa_pred_emb: (B, T, D) [stop_grad'd by caller]
    Returns logits: (B, T, 11)."""
    B, T, D = stream_emb.shape
    pad = jnp.broadcast_to(p["pad_emb"][None, None, :], (B, WINDOW - 1, D))
    padded = jnp.concatenate([pad, stream_emb], axis=1)
    windows = jnp.stack([padded[:, t:t + WINDOW, :] for t in range(T)], axis=1)
    # windows: (B, T, W, D); concat ROSA feature along channel
    feats = jnp.concatenate([
        windows.reshape(B, T, WINDOW * D),
        rosa_pred_emb,                                         # (B, T, D)
    ], axis=-1)                                                # (B, T, (W+1)*D)
    h = jax.nn.gelu(feats @ p["out_W1"] + p["out_b1"])
    return h @ p["out_W2"] + p["out_b2"]


# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(p, x_batch, y_batch, predicted_ids, key):
    B = x_batch.shape[0]
    y_oh = jax.nn.one_hot(y_batch, 10)

    z_sample = run_input_image(p, x_batch)
    z_label  = run_input_label(p, y_oh)
    k1, k2 = jax.random.split(key, 2)
    c_q_sample, ids_sample = diveq_quantize(z_sample, p["codebook"], k1)
    c_q_label,  ids_label  = diveq_quantize(z_label,  p["codebook"], k2)

    stream_emb = build_stream_emb(p, c_q_sample, c_q_label)
    rosa_pred_emb = p["codebook"][predicted_ids]                       # (B, T, D)
    if STOP_GRAD_ROSA:
        rosa_pred_emb = jax.lax.stop_gradient(rosa_pred_emb)

    logits = output_module(p, stream_emb, rosa_pred_emb)               # (B, T, 11)

    targets = jnp.full((B, T_STREAM), NO_OUTPUT, dtype=jnp.int32)
    targets = targets.at[:, LABEL_POS].set(y_batch)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    digit_mask = jnp.zeros(T_STREAM).at[LABEL_POS].set(1.0)
    no_mask    = 1.0 - digit_mask
    loss_no = (ce * no_mask).sum(axis=1) / no_mask.sum()
    loss_dg = (ce * digit_mask).sum(axis=1) / digit_mask.sum()
    cls = loss_no.mean() + LAMBDA_DIGIT_BOOST * loss_dg.mean()

    z_all  = jnp.concatenate([z_sample, z_label], axis=1)
    ids_all = jnp.concatenate([ids_sample, ids_label], axis=1)
    c_star = p["codebook"][ids_all]
    cb = ((jax.lax.stop_gradient(z_all) - c_star) ** 2).mean()

    aux = (cls, loss_no.mean(), loss_dg.mean(), cb, ids_sample, ids_label)
    return cls + LAMBDA_CB * cb, aux


# ── JIT'd helpers ─────────────────────────────────────────────────────────────

@jax.jit
def extract_ids(p, x_batch, y_batch, key):
    """Forward through input + DiVeQ to get token IDs (no gradient)."""
    y_oh = jax.nn.one_hot(y_batch, 10)
    z_sample = run_input_image(p, x_batch)
    z_label  = run_input_label(p, y_oh)
    k1, k2 = jax.random.split(key, 2)
    _, ids_sample = diveq_quantize(z_sample, p["codebook"], k1)
    _, ids_label  = diveq_quantize(z_label,  p["codebook"], k2)
    return ids_sample, ids_label


def make_step_fn(optimizer):
    @jax.jit
    def step(p, opt_state, x_batch, y_batch, predicted_ids, key):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, x_batch, y_batch, predicted_ids, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        cls, l_no, l_dg, cb, ids_s, ids_l = aux
        return p, opt_state, loss, cls, l_no, l_dg, cb, ids_s, ids_l
    return step


# ── ROSA helpers ──────────────────────────────────────────────────────────────

def build_int_streams(ids_sample_np, ids_label_np):
    """ids_sample_np: (B, K_SAMPLE), ids_label_np: (B, K_LABEL).
    Returns list[list[int]] of length B, each of length T_STREAM."""
    streams = []
    for i in range(ids_sample_np.shape[0]):
        if USE_INDICATORS:
            s = [ID_SAMPLE]
            s.extend(int(t) for t in ids_sample_np[i])
            s.append(ID_LABEL)
            s.extend(int(t) for t in ids_label_np[i])
        else:
            s = [int(t) for t in ids_sample_np[i]]
            s.extend(int(t) for t in ids_label_np[i])
        streams.append(s)
    return streams


def rosa_predict_for_streams(rosa: ROSACore, streams):
    """For each stream, call predict_argmax_along_stream. Returns:
       predicted_ids: (B, T_STREAM) int32
       hit_sources:   (B, T_STREAM) of strings
    """
    B = len(streams)
    pred = np.zeros((B, T_STREAM), dtype=np.int32)
    sources = []
    for i, s in enumerate(streams):
        preds = rosa.predict_argmax_along_stream(s)
        srcs_i = []
        for t, (pid, src) in enumerate(preds):
            pred[i, t] = pid
            srcs_i.append(src)
        sources.append(srcs_i)
    return pred, sources


def rosa_commit_streams(rosa: ROSACore, streams):
    for s in streams:
        rosa.commit_burst(s)


def hit_rate_at_label(sources):
    """Fraction of LABEL_POS predictions that came from SAM (vs lm/unigram/empty)."""
    return sum(1 for s in sources if s[LABEL_POS] == "sam") / max(1, len(sources))


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_metrics(p, rosa: ROSACore, X, y, batch_size: int = 256):
    """Frozen-ROSA eval. Walks each test sample's tokens through the SAM
    without committing, computes 10-way digit argmax at LABEL_POS."""
    preds_all, ids_s_all, ids_l_all = [], [], []
    fr_sum, fr_lab_sum, hit_sum, n_batches = 0.0, 0.0, 0.0, 0
    fr_helper = jax.jit(_fire_rate_summary)
    for i in range(0, len(X), batch_size):
        bx = jnp.array(X[i:i + batch_size])
        by = jnp.array(y[i:i + batch_size])
        ids_s, ids_l = extract_ids(p, bx, by, jnp.zeros(2, dtype=jnp.uint32))
        ids_s_np = np.array(ids_s); ids_l_np = np.array(ids_l)
        streams = build_int_streams(ids_s_np, ids_l_np)
        predicted_ids_np, sources = rosa_predict_for_streams(rosa, streams)
        predicted_ids = jnp.array(predicted_ids_np)

        preds, fr, fr_lab = _eval_logits(p, bx, by, predicted_ids,
                                         jnp.zeros(2, dtype=jnp.uint32))
        preds_all.append(np.array(preds))
        ids_s_all.append(ids_s_np); ids_l_all.append(ids_l_np)
        fr_sum += float(fr); fr_lab_sum += float(fr_lab)
        hit_sum += hit_rate_at_label(sources) * len(streams)
        n_batches += 1

    preds = np.concatenate(preds_all)
    acc = float((preds == np.array(y)).mean())
    cb_used = int(len(np.unique(np.concatenate(
        [np.concatenate(ids_s_all).flatten(),
         np.concatenate(ids_l_all).flatten()]))))
    return {
        "test_acc": acc,
        "cb_used": cb_used,
        "fire_rate": fr_sum / n_batches,
        "fire_at_label": fr_lab_sum / n_batches,
        "rosa_hit_rate_at_label": hit_sum / len(X),
    }


def _fire_rate_summary(logits):
    pred_classes = jnp.argmax(logits, axis=-1)
    fire_mask = pred_classes != NO_OUTPUT
    fire_rate = fire_mask.mean()
    pos_mask = jnp.zeros(T_STREAM, dtype=jnp.bool_).at[LABEL_POS].set(True)
    fire_at_lbl = (fire_mask & pos_mask[None, :]).sum() / fire_mask.sum().clip(min=1)
    return fire_rate, fire_at_lbl


@jax.jit
def _eval_logits(p, x_batch, y_batch, predicted_ids, key):
    y_oh = jax.nn.one_hot(y_batch, 10)
    z_sample = run_input_image(p, x_batch)
    z_label  = run_input_label(p, y_oh)
    c_q_sample, _ = diveq_quantize(z_sample, p["codebook"], key)
    c_q_label,  _ = diveq_quantize(z_label,  p["codebook"], key)
    stream_emb = build_stream_emb(p, c_q_sample, c_q_label)
    rosa_pred_emb = p["codebook"][predicted_ids]
    logits = output_module(p, stream_emb, rosa_pred_emb)
    digit_logits = logits[:, LABEL_POS, :10]
    preds = jnp.argmax(digit_logits, axis=-1)
    fr, fr_lab = _fire_rate_summary(logits)
    return preds, fr, fr_lab


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Skip-if-done
    done = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                try: done.add(json.loads(line).get("experiment"))
                except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping")
        return

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
        f"params: {n_params(p):,}  V={VOCAB_SIZE} D={TOKEN_DIM}  "
        f"K_s={K_SAMPLE} K_l={K_LABEL}  T={T_STREAM}  λ_digit={LAMBDA_DIGIT_BOOST}"
    )

    rosa = ROSACore(vocab_size=VOCAB_SIZE)
    logging.info(f"ROSA initialized: vocab={VOCAB_SIZE}, max_order=None")

    num_batches = len(X_train) // BATCH_SIZE
    schedule = optax.cosine_decay_schedule(LR, N_EPOCHS * num_batches, alpha=0.05)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(p)
    step_fn = make_step_fn(optimizer)

    rng = np.random.default_rng(SEED + 1)
    t0 = time.perf_counter()
    history = {"loss": [], "cls": [], "l_no": [], "l_dg": [], "cb": [],
               "test_acc": [], "cb_used": [], "fire_rate": [],
               "fire_at_label": [], "rosa_hit_rate_at_label": [],
               "sam_states": [], "rosa_train_hit_rate": []}

    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs = X_train[perm]; ys = y_train[perm]
        ep_loss = ep_cls = ep_no = ep_dg = ep_cb = 0.0
        ep_train_hits = 0; ep_train_pos = 0

        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx = jnp.array(Xs[s:e]); by = jnp.array(ys[s:e])
            key = next(key_gen).get()

            # 1) Extract ids (no grad)
            ids_s, ids_l = extract_ids(p, bx, by, key)
            ids_s_np = np.array(ids_s); ids_l_np = np.array(ids_l)
            streams = build_int_streams(ids_s_np, ids_l_np)

            # 2) ROSA query (frozen during this batch)
            predicted_ids_np, sources = rosa_predict_for_streams(rosa, streams)
            predicted_ids = jnp.array(predicted_ids_np)

            # 3) Step (forward + grad + update)
            p, opt_state, loss, cls, l_no, l_dg, cb, _, _ = step_fn(
                p, opt_state, bx, by, predicted_ids, key
            )

            # 4) Commit batch's streams to ROSA
            rosa_commit_streams(rosa, streams)

            ep_loss += float(loss); ep_cls += float(cls); ep_no += float(l_no)
            ep_dg += float(l_dg);   ep_cb += float(cb)
            ep_train_hits += sum(s[LABEL_POS] == "sam" for s in sources)
            ep_train_pos  += len(sources)

        for k, v in [("loss", ep_loss), ("cls", ep_cls), ("l_no", ep_no),
                     ("l_dg", ep_dg), ("cb", ep_cb)]:
            history[k].append(v / num_batches)
        history["rosa_train_hit_rate"].append(ep_train_hits / max(1, ep_train_pos))
        history["sam_states"].append(rosa.n_states())

        m = eval_metrics(p, rosa, X_test, y_test)
        history["test_acc"].append(m["test_acc"])
        history["cb_used"].append(m["cb_used"])
        history["fire_rate"].append(m["fire_rate"])
        history["fire_at_label"].append(m["fire_at_label"])
        history["rosa_hit_rate_at_label"].append(m["rosa_hit_rate_at_label"])

        logging.info(
            f"epoch {epoch:2d}  cls={history['cls'][-1]:.4f} "
            f"(no={history['l_no'][-1]:.4f} dg={history['l_dg'][-1]:.4f}) "
            f"cb={history['cb'][-1]:.4f}  "
            f"test={m['test_acc']:.4f}  cb_used={m['cb_used']}/{N_CONTENT}  "
            f"fire={m['fire_rate']:.3f} fr_lbl={m['fire_at_label']:.3f}  "
            f"rosa_hit={m['rosa_hit_rate_at_label']:.3f}  "
            f"sam={rosa.n_states():,}  {time.perf_counter() - t0:.1f}s"
        )

    elapsed = time.perf_counter() - t0
    final = eval_metrics(p, rosa, X_test, y_test)
    train_eval = eval_metrics(p, rosa, X_train[:10_000], y_train[:10_000])

    logging.info(
        f"\nFINAL  test={final['test_acc']:.4f}  "
        f"train(10k)={train_eval['test_acc']:.4f}  "
        f"cb_used={final['cb_used']}/{N_CONTENT}  "
        f"rosa_hit={final['rosa_hit_rate_at_label']:.3f}  "
        f"sam={rosa.n_states():,}  time={elapsed:.0f}s"
    )

    # Save
    with open(PROJ_DIR / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in p.items()}, f)
    rosa.save(str(PROJ_DIR / f"rosa_{EXP_NAME}.json"))
    with open(PROJ_DIR / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    row = {
        "experiment": EXP_NAME,
        "name": f"baseline-fixedK-{DATASET}",
        "dataset": DATASET,
        "test_acc": final["test_acc"],
        "train_acc": train_eval["test_acc"],
        "cb_used": final["cb_used"],
        "fire_rate": final["fire_rate"],
        "fire_at_label_rate": final["fire_at_label"],
        "rosa_hit_rate_at_label": final["rosa_hit_rate_at_label"],
        "sam_state_count": rosa.n_states(),
        "n_params": n_params(p),
        "epochs": N_EPOCHS,
        "time_s": round(elapsed, 1),
        "history": history,
        "hp": {
            "VOCAB_SIZE": VOCAB_SIZE, "TOKEN_DIM": TOKEN_DIM, "HIDDEN": HIDDEN,
            "K_SAMPLE": K_SAMPLE, "K_LABEL": K_LABEL, "T_STREAM": T_STREAM,
            "LABEL_POS": LABEL_POS, "WINDOW": WINDOW,
            "SIGMA": SIGMA, "LAMBDA_CB": LAMBDA_CB,
            "LAMBDA_DIGIT_BOOST": LAMBDA_DIGIT_BOOST,
            "BATCH_SIZE": BATCH_SIZE, "LR": LR, "N_EPOCHS": N_EPOCHS,
        },
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended to {JSONL}")


if __name__ == "__main__":
    main()
