"""
exp10 — EGGROLL antithetic ES with ROSA in the loop (Fashion-MNIST).

Hypothesis: gradient-free optimization (low-rank antithetic ES) removes the
stop_grad structural barrier from exp1/exp6. The optimizer can directly
reward parameter configurations where ROSA's predicted-token embedding helps
classify; ROSA's non-differentiability is implicit and the codebook can
re-shape to make ROSA's discrete predictions class-meaningful.

Architecture: identical to exp1 (input MLP + DiVeQ codebook + dense-fire
11-class output module with rosa_pred_emb feature). No stop_grad needed
because ES doesn't use gradients.

Optimizer: antithetic ES, rank-1 perturbations (EGGROLL pattern from
projects/es/experiments3.py). σ=1e-3, n_pop=32 (antithetic pairs=16),
Adam over ES gradient estimate, cosine LR 1e-3 → 1e-4.

For wall-time, we use a 10K training subset; if it shows promise we can scale.

Usage: uv run python projects/rosa/scripts/run_experiments.py exp10
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

from shared_lib.datasets import load_supervised_image                   # noqa: E402
from shared_lib.random_utils import infinite_safe_keys_from_key         # noqa: E402
from lib.rosa_core import ROSACore                                      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Hyperparameters ───────────────────────────────────────────────────────────

DATASET             = "fashion_mnist"
EXP_NAME            = "exp10b_fmnist"

VOCAB_SIZE          = 1024
TOKEN_DIM           = 64
HIDDEN              = 256
K_SAMPLE            = 16
K_LABEL             = 4
WINDOW              = 8
LAMBDA_DIGIT_BOOST  = 20.0

ES_SIGMA            = 0.001
ES_N_POP            = 32           # n_pop is the total number of (positive-half) members
ES_RANK             = 1
ADAM_LR_INIT        = 1e-3
ADAM_LR_END         = 1e-4

N_TRAIN             = 60_000        # exp10b: full Fashion-MNIST training set
N_TEST              = 10_000
BATCH_SIZE          = 64
N_EPOCHS            = 30
SEED                = 42

ID_SAMPLE = 0
ID_LABEL  = 1
N_RESERVED = 2
N_CONTENT  = VOCAB_SIZE - N_RESERVED

T_STREAM  = 1 + K_SAMPLE + 1 + K_LABEL          # 22
LABEL_POS = 1 + K_SAMPLE                         # 17

N_CLASSES_OUT = 11
NO_OUTPUT     = 10

JSONL = PROJ_DIR / "results.jsonl"

EVAL_EVERY = 2     # eval test set every N epochs (eval is slow due to ROSA queries)


# ── Params ────────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
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

        "out_W1":         n(((WINDOW + 1) * TOKEN_DIM, HIDDEN),
                            ((WINDOW + 1) * TOKEN_DIM) ** -0.5),
        "out_b1":         jnp.zeros(HIDDEN),
        "out_W2":         n((HIDDEN, N_CLASSES_OUT), HIDDEN ** -0.5),
        "out_b2":         jnp.zeros(N_CLASSES_OUT),
    }


def n_params(p) -> int:
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


# ── Forward primitives (vectorizable) ─────────────────────────────────────────

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
    return jnp.stack(zs, axis=1)


def run_input_label(p, y_oh):
    h = jax.nn.gelu(y_oh @ p["proj_label_W"] + p["proj_label_b"])
    zs = []
    for _ in range(K_LABEL):
        h = _block_step(p, h)
        zs.append(_emit(p, h))
    return jnp.stack(zs, axis=1)


def hard_quantize(z, codebook):
    """ES doesn't need DiVeQ's gradient trick. Just argmin → c_star. Fast."""
    z_flat = z.reshape(-1, TOKEN_DIM)
    dists = ((z_flat[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    mask = jnp.zeros(VOCAB_SIZE).at[:N_RESERVED].set(jnp.inf)
    ids_flat = jnp.argmin(dists + mask[None, :], axis=-1)
    c_star = codebook[ids_flat]
    return c_star.reshape(z.shape), ids_flat.reshape(z.shape[:-1])


def build_stream_emb(p, c_q_sample, c_q_label):
    B = c_q_sample.shape[0]
    sample_emb = jnp.broadcast_to(p["codebook"][ID_SAMPLE][None, None, :],
                                  (B, 1, TOKEN_DIM))
    label_emb  = jnp.broadcast_to(p["codebook"][ID_LABEL][None, None, :],
                                  (B, 1, TOKEN_DIM))
    return jnp.concatenate([sample_emb, c_q_sample, label_emb, c_q_label], axis=1)


def output_module(p, stream_emb, rosa_pred_emb):
    B, T, D = stream_emb.shape
    pad = jnp.broadcast_to(p["pad_emb"][None, None, :], (B, WINDOW - 1, D))
    padded = jnp.concatenate([pad, stream_emb], axis=1)
    windows = jnp.stack([padded[:, t:t + WINDOW, :] for t in range(T)], axis=1)
    feats = jnp.concatenate([
        windows.reshape(B, T, WINDOW * D),
        rosa_pred_emb,
    ], axis=-1)
    h = jax.nn.gelu(feats @ p["out_W1"] + p["out_b1"])
    return h @ p["out_W2"] + p["out_b2"]


def classify_loss(p, x_batch, y_batch, predicted_ids):
    B = x_batch.shape[0]
    y_oh = jax.nn.one_hot(y_batch, 10)
    z_sample = run_input_image(p, x_batch)
    z_label  = run_input_label(p, y_oh)
    c_q_sample, _ = hard_quantize(z_sample, p["codebook"])
    c_q_label,  _ = hard_quantize(z_label,  p["codebook"])
    stream_emb = build_stream_emb(p, c_q_sample, c_q_label)
    rosa_pred_emb = p["codebook"][predicted_ids]
    logits = output_module(p, stream_emb, rosa_pred_emb)

    targets = jnp.full((B, T_STREAM), NO_OUTPUT, dtype=jnp.int32)
    targets = targets.at[:, LABEL_POS].set(y_batch)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    digit_mask = jnp.zeros(T_STREAM).at[LABEL_POS].set(1.0)
    no_mask    = 1.0 - digit_mask
    loss_no = (ce * no_mask).sum(axis=1) / no_mask.sum()
    loss_dg = (ce * digit_mask).sum(axis=1) / digit_mask.sum()
    return loss_no.mean() + LAMBDA_DIGIT_BOOST * loss_dg.mean()


# ── Perturbation: rank-1 EGGROLL ──────────────────────────────────────────────

def make_eps(pop_key, params):
    """Generate rank-1 (matrices) / full (vectors) perturbation for one pop member."""
    eps = {}
    for idx, (k, v) in enumerate(params.items()):
        subkey = jax.random.fold_in(pop_key, idx)
        if v.ndim == 2:
            m, n = v.shape
            ab = jax.random.normal(subkey, (m + n, ES_RANK))
            a, b = ab[:m], ab[m:]
            eps[k] = (a @ b.T) * ES_SIGMA
        else:
            eps[k] = jax.random.normal(subkey, v.shape) * ES_SIGMA
    return eps


def apply_perturbation(params, eps, sign):
    return {k: v + sign * eps[k] for k, v in params.items()}


# ── JIT helpers ───────────────────────────────────────────────────────────────

@jax.jit
def extract_ids_pop(params, pop_keys, x_batch, y_batch):
    """For each pop member, compute IDs at ±eps. Returns:
       ids_pos_s: (n_pop, B, K_SAMPLE)
       ids_pos_l: (n_pop, B, K_LABEL)
       ids_neg_s, ids_neg_l: same shapes."""
    y_oh = jax.nn.one_hot(y_batch, 10)

    def per_pop(pop_key):
        eps = make_eps(pop_key, params)
        p_pos = apply_perturbation(params, eps, +1)
        p_neg = apply_perturbation(params, eps, -1)
        z_pos_s = run_input_image(p_pos, x_batch)
        z_pos_l = run_input_label(p_pos, y_oh)
        z_neg_s = run_input_image(p_neg, x_batch)
        z_neg_l = run_input_label(p_neg, y_oh)
        _, ids_pos_s = hard_quantize(z_pos_s, p_pos["codebook"])
        _, ids_pos_l = hard_quantize(z_pos_l, p_pos["codebook"])
        _, ids_neg_s = hard_quantize(z_neg_s, p_neg["codebook"])
        _, ids_neg_l = hard_quantize(z_neg_l, p_neg["codebook"])
        return ids_pos_s, ids_pos_l, ids_neg_s, ids_neg_l

    return jax.vmap(per_pop)(pop_keys)


@jax.jit
def compute_fitnesses(params, pop_keys, x_batch, y_batch,
                      pred_ids_pos, pred_ids_neg):
    """Return (fitness[n_pop],) = (loss_pos - loss_neg) per pop member."""
    def per_pop(pop_key, pred_pos, pred_neg):
        eps = make_eps(pop_key, params)
        p_pos = apply_perturbation(params, eps, +1)
        p_neg = apply_perturbation(params, eps, -1)
        loss_pos = classify_loss(p_pos, x_batch, y_batch, pred_pos)
        loss_neg = classify_loss(p_neg, x_batch, y_batch, pred_neg)
        return loss_pos - loss_neg
    return jax.vmap(per_pop)(pop_keys, pred_ids_pos, pred_ids_neg)


def make_apply_es_update(optimizer):
    @jax.jit
    def apply_es_update(params, opt_state, pop_keys, fitnesses):
        """Aggregate eps × fitness, apply Adam update."""
        def per_pop_eps(pop_key):
            return make_eps(pop_key, params)
        epses = jax.vmap(per_pop_eps)(pop_keys)        # dict of (n_pop, *v.shape)
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (ES_N_POP * ES_SIGMA ** 2),
            epses,
        )
        updates, opt_state = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state
    return apply_es_update


@jax.jit
def extract_ids_one(p, x_batch, y_batch):
    """Single forward pass (used for committing tokens to ROSA after step)."""
    y_oh = jax.nn.one_hot(y_batch, 10)
    z_sample = run_input_image(p, x_batch)
    z_label  = run_input_label(p, y_oh)
    _, ids_s = hard_quantize(z_sample, p["codebook"])
    _, ids_l = hard_quantize(z_label,  p["codebook"])
    return ids_s, ids_l


# ── ROSA stream helpers ───────────────────────────────────────────────────────

def build_int_streams(ids_sample_np, ids_label_np):
    streams = []
    for i in range(ids_sample_np.shape[0]):
        s = [ID_SAMPLE]
        s.extend(int(t) for t in ids_sample_np[i])
        s.append(ID_LABEL)
        s.extend(int(t) for t in ids_label_np[i])
        streams.append(s)
    return streams


def rosa_predict_for_streams(rosa: ROSACore, streams):
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


# ── Eval ──────────────────────────────────────────────────────────────────────

@jax.jit
def _eval_logits(p, x_batch, y_batch, predicted_ids):
    y_oh = jax.nn.one_hot(y_batch, 10)
    z_sample = run_input_image(p, x_batch)
    z_label  = run_input_label(p, y_oh)
    c_q_sample, _ = hard_quantize(z_sample, p["codebook"])
    c_q_label,  _ = hard_quantize(z_label,  p["codebook"])
    stream_emb = build_stream_emb(p, c_q_sample, c_q_label)
    rosa_pred_emb = p["codebook"][predicted_ids]
    return output_module(p, stream_emb, rosa_pred_emb)


def eval_test(p, rosa, X, y, batch_size: int = 128):
    preds_all = []
    for i in range(0, len(X), batch_size):
        bx = jnp.array(X[i:i + batch_size])
        by = jnp.array(y[i:i + batch_size])
        ids_s, ids_l = extract_ids_one(p, bx, by)
        streams = build_int_streams(np.array(ids_s), np.array(ids_l))
        predicted_ids_np, _ = rosa_predict_for_streams(rosa, streams)
        predicted_ids = jnp.array(predicted_ids_np)
        logits = _eval_logits(p, bx, by, predicted_ids)
        digit_logits = logits[:, LABEL_POS, :10]
        preds_all.append(np.array(jnp.argmax(digit_logits, axis=-1)))
    preds = np.concatenate(preds_all)
    return float((preds == np.array(y)).mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
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
    X_train, y_train = X_train[:N_TRAIN], y_train[:N_TRAIN]
    X_test,  y_test  = X_test[:N_TEST],  y_test[:N_TEST]
    logging.info(f"{DATASET}: train={X_train.shape}  test={X_test.shape}")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = init_params(key_gen)
    logging.info(
        f"params: {n_params(p):,}  V={VOCAB_SIZE} D={TOKEN_DIM}  "
        f"K_s={K_SAMPLE} K_l={K_LABEL}  T={T_STREAM}  "
        f"n_pop={ES_N_POP} σ={ES_SIGMA} rank={ES_RANK}"
    )

    rosa = ROSACore(vocab_size=VOCAB_SIZE)

    num_batches = len(X_train) // BATCH_SIZE
    total_steps = N_EPOCHS * num_batches
    lr_sched = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                            alpha=ADAM_LR_END / ADAM_LR_INIT)
    optimizer = optax.adam(lr_sched)
    opt_state = optimizer.init(p)
    apply_es_update = make_apply_es_update(optimizer)

    rng = np.random.default_rng(SEED + 1)
    t0 = time.perf_counter()
    history = {"loss_signed_mag": [], "test_acc": [], "sam_states": [],
               "rosa_hit_train": []}

    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs = X_train[perm]; ys = y_train[perm]
        ep_fit_abs = 0.0
        ep_hits = ep_pos = 0

        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx = jnp.array(Xs[s:e]); by = jnp.array(ys[s:e])
            key = next(key_gen).get()
            pop_keys = jax.random.split(key, ES_N_POP)

            # 1) Extract ids for all pop members (vmapped over n_pop)
            ids_pos_s, ids_pos_l, ids_neg_s, ids_neg_l = extract_ids_pop(
                p, pop_keys, bx, by
            )
            ids_pos_s_np = np.array(ids_pos_s); ids_pos_l_np = np.array(ids_pos_l)
            ids_neg_s_np = np.array(ids_neg_s); ids_neg_l_np = np.array(ids_neg_l)

            # 2) ROSA query per pop member, per sign
            pred_pos = np.zeros((ES_N_POP, BATCH_SIZE, T_STREAM), dtype=np.int32)
            pred_neg = np.zeros((ES_N_POP, BATCH_SIZE, T_STREAM), dtype=np.int32)
            for k in range(ES_N_POP):
                streams_pos = build_int_streams(ids_pos_s_np[k], ids_pos_l_np[k])
                streams_neg = build_int_streams(ids_neg_s_np[k], ids_neg_l_np[k])
                pp, _ = rosa_predict_for_streams(rosa, streams_pos)
                pn, _ = rosa_predict_for_streams(rosa, streams_neg)
                pred_pos[k] = pp; pred_neg[k] = pn

            # 3) Compute fitnesses (loss_pos - loss_neg) per pop member
            fitnesses = compute_fitnesses(
                p, pop_keys, bx, by,
                jnp.array(pred_pos), jnp.array(pred_neg),
            )

            # 4) Apply ES gradient + Adam update
            p, opt_state = apply_es_update(p, opt_state, pop_keys, fitnesses)

            # 5) Commit current-batch streams to ROSA (using updated params)
            ids_s, ids_l = extract_ids_one(p, bx, by)
            streams = build_int_streams(np.array(ids_s), np.array(ids_l))
            _, train_sources = rosa_predict_for_streams(rosa, streams)
            ep_hits += sum(s[LABEL_POS] == "sam" for s in train_sources)
            ep_pos  += len(train_sources)
            rosa_commit_streams(rosa, streams)

            ep_fit_abs += float(jnp.mean(jnp.abs(fitnesses)))

        history["loss_signed_mag"].append(ep_fit_abs / num_batches)
        history["rosa_hit_train"].append(ep_hits / max(1, ep_pos))
        history["sam_states"].append(rosa.n_states())

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == N_EPOCHS - 1:
            test_acc = eval_test(p, rosa, X_test, y_test)
            history["test_acc"].append((epoch, test_acc))
            tag = f"  test={test_acc:.4f}"
        else:
            tag = ""

        logging.info(
            f"epoch {epoch:2d}  |fit|={history['loss_signed_mag'][-1]:.4f}  "
            f"rosa_hit_tr={history['rosa_hit_train'][-1]:.3f}  "
            f"sam={rosa.n_states():,}{tag}  "
            f"{time.perf_counter() - t0:.1f}s"
        )

    elapsed = time.perf_counter() - t0
    final_test = eval_test(p, rosa, X_test, y_test)
    # Eval on first 10K of training set for direct comparability with exp10
    final_train = eval_test(p, rosa, X_train[:10_000], y_train[:10_000])

    logging.info(
        f"\nFINAL  test={final_test:.4f}  train={final_train:.4f}  "
        f"sam={rosa.n_states():,}  time={elapsed:.0f}s"
    )

    with open(PROJ_DIR / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in p.items()}, f)
    rosa.save(str(PROJ_DIR / f"rosa_{EXP_NAME}.json"))
    with open(PROJ_DIR / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    row = {
        "experiment": EXP_NAME,
        "name": f"eggroll-rosa-{DATASET}",
        "dataset": DATASET,
        "test_acc": final_test,
        "train_acc": final_train,
        "sam_state_count": rosa.n_states(),
        "n_params": n_params(p),
        "epochs": N_EPOCHS,
        "n_train": N_TRAIN,
        "time_s": round(elapsed, 1),
        "history": history,
        "hp": {
            "VOCAB_SIZE": VOCAB_SIZE, "TOKEN_DIM": TOKEN_DIM, "HIDDEN": HIDDEN,
            "K_SAMPLE": K_SAMPLE, "K_LABEL": K_LABEL,
            "ES_SIGMA": ES_SIGMA, "ES_N_POP": ES_N_POP, "ES_RANK": ES_RANK,
            "ADAM_LR_INIT": ADAM_LR_INIT, "ADAM_LR_END": ADAM_LR_END,
            "BATCH_SIZE": BATCH_SIZE, "N_EPOCHS": N_EPOCHS, "N_TRAIN": N_TRAIN,
            "LAMBDA_DIGIT_BOOST": LAMBDA_DIGIT_BOOST, "WINDOW": WINDOW,
        },
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended to {JSONL}")


if __name__ == "__main__":
    main()
