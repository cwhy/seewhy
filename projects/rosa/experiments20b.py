"""
exp20 — ES + ROSA, minimal codebook-as-classifier head (MNIST first).

Architecture: no indicators, no separate label encoder, no output MLP. The
output classifier is the dot product of ROSA's predicted-token embedding
against the first 11 codebook entries (10 digit slots + 1 null), i.e. the
codebook itself serves as the linear head.

Stream per training sample (length 17, no indicators):
   [ s_1, s_2, ..., s_16, y_token ]
where s_i are content tokens (IDs >= 11) from input ACT-MLP + DiVeQ, and
y_token = y (digit 0..9). At inference, ROSA is queried at position 15 after
walking [s_1..s_16]; the predicted-token embedding is projected to 11
logits via the codebook, and argmax over the digit slice gives the digit.

Optimizer: antithetic ES, rank=4, n_pop=32, σ=1e-3, Adam over the ES gradient
estimate with cosine LR 1e-3→1e-4. Identical to exp10c except for the
minimal output head and no label encoder.

Usage: uv run python projects/rosa/scripts/run_experiments.py exp20
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

DATASET             = "mnist"
EXP_NAME            = f"exp20b_{DATASET.replace('fashion_mnist', 'fmnist')}"

VOCAB_SIZE          = 1024
TOKEN_DIM           = 64
HIDDEN              = 256
K_SAMPLE            = 16
LAMBDA_DIGIT_BOOST  = 16.0   # 16 null positions : 1 digit position

ES_SIGMA            = 0.001
ES_N_POP            = 128       # exp20b: 4× SNR vs exp20
ES_RANK             = 4
ADAM_LR_INIT        = 1e-3
ADAM_LR_END         = 1e-4

N_TRAIN             = 60_000
N_TEST              = 10_000
BATCH_SIZE          = 64
N_EPOCHS            = 30
SEED                = 42

# Reserved-ID layout (no indicators):
#   IDs 0..9   → digit-class slots         (10 reserved)
#   ID  10     → null slot                 (1 reserved)
#   IDs 11..1023 → content tokens          (1013 entries the input MLP can produce)
N_CLASSES_OUT       = 11             # 10 digits + 1 null
DIGIT_OFFSET        = 0              # digit y → ID y
NULL_ID             = 10
N_RESERVED          = 11
N_CONTENT           = VOCAB_SIZE - N_RESERVED

# Stream layout: 16 sample tokens + 1 label token, no indicators
T_STREAM            = K_SAMPLE + 1   # 17
LABEL_POS           = K_SAMPLE - 1   # 15 (just after observing s_16; next token = y)

JSONL = PROJ_DIR / "results.jsonl"
EVAL_EVERY = 2


# ── Params ────────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        # Input MLP (image only — no label encoder)
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
        # Codebook — entries 0..10 also serve as the linear classifier head
        "codebook":       n((VOCAB_SIZE, TOKEN_DIM), TOKEN_DIM ** -0.5),
    }


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
    return jnp.stack(zs, axis=1)


def hard_quantize_content_only(z, codebook):
    """argmin over content slots only (IDs >= N_RESERVED). Returns IDs in 11..1023."""
    z_flat = z.reshape(-1, TOKEN_DIM)
    dists = ((z_flat[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    mask = jnp.zeros(VOCAB_SIZE).at[:N_RESERVED].set(jnp.inf)
    ids_flat = jnp.argmin(dists + mask[None, :], axis=-1)
    return ids_flat.reshape(z.shape[:-1])


def classify_loss(p, x_batch, y_batch, predicted_ids):
    """predicted_ids: (B, T_STREAM) — ROSA's argmax-predicted token IDs along each stream.
    Output module:   logits[b, t, c] = codebook[predicted_ids[b, t]] @ codebook[c].T  for c in 0..10."""
    rosa_pred_emb = p["codebook"][predicted_ids]                       # (B, T, D)
    class_embs    = p["codebook"][:N_CLASSES_OUT]                      # (11, D)
    logits        = rosa_pred_emb @ class_embs.T                        # (B, T, 11)

    B = x_batch.shape[0]
    targets = jnp.full((B, T_STREAM), NULL_ID, dtype=jnp.int32)
    targets = targets.at[:, LABEL_POS].set(y_batch + DIGIT_OFFSET)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)  # (B, T)

    digit_mask = jnp.zeros(T_STREAM).at[LABEL_POS].set(1.0)
    no_mask    = 1.0 - digit_mask
    loss_no    = (ce * no_mask).sum(axis=1) / no_mask.sum()
    loss_dg    = (ce * digit_mask).sum(axis=1) / digit_mask.sum()
    return loss_no.mean() + LAMBDA_DIGIT_BOOST * loss_dg.mean()


# ── Perturbation: rank-k EGGROLL ──────────────────────────────────────────────

def make_eps(pop_key, params):
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
def extract_ids_pop(params, pop_keys, x_batch):
    """For each pop member, get content-token IDs at ±ε. Returns (n_pop, B, K_SAMPLE) twice."""
    def per_pop(pop_key):
        eps = make_eps(pop_key, params)
        p_pos = apply_perturbation(params, eps, +1)
        p_neg = apply_perturbation(params, eps, -1)
        z_pos = run_input_image(p_pos, x_batch)
        z_neg = run_input_image(p_neg, x_batch)
        ids_pos = hard_quantize_content_only(z_pos, p_pos["codebook"])
        ids_neg = hard_quantize_content_only(z_neg, p_neg["codebook"])
        return ids_pos, ids_neg
    return jax.vmap(per_pop)(pop_keys)


@jax.jit
def compute_fitnesses(params, pop_keys, x_batch, y_batch,
                      pred_ids_pos, pred_ids_neg):
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
        def per_pop_eps(pop_key):
            return make_eps(pop_key, params)
        epses = jax.vmap(per_pop_eps)(pop_keys)
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (ES_N_POP * ES_SIGMA ** 2),
            epses,
        )
        updates, opt_state = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state
    return apply_es_update


@jax.jit
def extract_ids_one(p, x_batch):
    z = run_input_image(p, x_batch)
    return hard_quantize_content_only(z, p["codebook"])


# ── Stream / ROSA helpers ─────────────────────────────────────────────────────

def build_int_streams_with_label(ids_sample_np, y_np):
    """Returns list[list[int]], each = [s_1..s_16, y_token]  (length 17)."""
    streams = []
    for i in range(ids_sample_np.shape[0]):
        s = [int(t) for t in ids_sample_np[i]]
        s.append(int(y_np[i]) + DIGIT_OFFSET)
        streams.append(s)
    return streams


def build_int_streams_no_label(ids_sample_np):
    """For eval: just the 16 sample tokens (no label committed)."""
    return [[int(t) for t in row] for row in ids_sample_np]


def rosa_predict_for_streams(rosa: ROSACore, streams):
    B = len(streams)
    T = len(streams[0])
    pred = np.zeros((B, T), dtype=np.int32)
    sources = []
    for i, s in enumerate(streams):
        preds = rosa.predict_argmax_along_stream(s)
        src_i = []
        for t, (pid, src) in enumerate(preds):
            pred[i, t] = pid
            src_i.append(src)
        sources.append(src_i)
    return pred, sources


def rosa_commit_streams(rosa: ROSACore, streams):
    for s in streams:
        rosa.commit_burst(s)


# ── Eval ──────────────────────────────────────────────────────────────────────

@jax.jit
def _eval_digit_logits(p, x_batch, rosa_pred_emb_at_label):
    """rosa_pred_emb_at_label: (B, D) — the predicted-token embedding at position LABEL_POS only.
    Returns (B, 10) digit logits (ignoring null)."""
    class_embs = p["codebook"][:N_CLASSES_OUT]                # (11, D)
    full_logits = rosa_pred_emb_at_label @ class_embs.T       # (B, 11)
    return full_logits[:, :10]                                # 10 digit logits


def eval_test(p, rosa, X, y, batch_size: int = 128):
    preds_all = []
    hit_count, total = 0, 0
    for i in range(0, len(X), batch_size):
        bx = jnp.array(X[i:i + batch_size])
        ids = extract_ids_one(p, bx)
        ids_np = np.array(ids)
        # At eval, we walk ROSA along just the sample tokens [s_1..s_16]
        # (we don't know the label y, so no y_token appended)
        streams = build_int_streams_no_label(ids_np)
        # Query ROSA at each position; we need position LABEL_POS = 15 (the last
        # position of [s_1..s_16] — predicting what comes next, which should be y_token)
        pred_ids_np = np.zeros((len(streams), K_SAMPLE), dtype=np.int32)
        for j, s in enumerate(streams):
            preds = rosa.predict_argmax_along_stream(s)
            for t, (pid, src) in enumerate(preds):
                pred_ids_np[j, t] = pid
                if t == LABEL_POS and src == "sam":
                    hit_count += 1
            total += 1
        rosa_pred_at_label = jnp.array(pred_ids_np[:, LABEL_POS])  # (B,)
        rosa_emb_at_label  = p["codebook"][rosa_pred_at_label]      # (B, D)
        digit_logits = _eval_digit_logits(p, bx, rosa_emb_at_label) # (B, 10)
        preds_all.append(np.array(jnp.argmax(digit_logits, axis=-1)))
    preds = np.concatenate(preds_all)
    return float((preds == np.array(y)).mean()), hit_count / max(1, total)


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
        f"params: {n_params(p):,}  V={VOCAB_SIZE} D={TOKEN_DIM}  K_s={K_SAMPLE}  "
        f"T={T_STREAM} LABEL_POS={LABEL_POS}  "
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
    history = {"fit_abs": [], "test_acc": [], "sam_states": [], "rosa_hit_eval": []}

    for epoch in range(N_EPOCHS):
        perm = rng.permutation(len(X_train))
        Xs = X_train[perm]; ys = y_train[perm]
        ep_fit_abs = 0.0

        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx = jnp.array(Xs[s:e]); by = jnp.array(ys[s:e])
            key = next(key_gen).get()
            pop_keys = jax.random.split(key, ES_N_POP)

            # 1) Per-pop content-token IDs at ±ε (vmapped JAX)
            ids_pos, ids_neg = extract_ids_pop(p, pop_keys, bx)
            ids_pos_np = np.array(ids_pos); ids_neg_np = np.array(ids_neg)
            y_np = np.array(by)

            # 2) Per-pop ROSA queries (Python). Stream = [s_1..s_16, y_token]
            pred_pos = np.zeros((ES_N_POP, BATCH_SIZE, T_STREAM), dtype=np.int32)
            pred_neg = np.zeros((ES_N_POP, BATCH_SIZE, T_STREAM), dtype=np.int32)
            for k in range(ES_N_POP):
                streams_pos = build_int_streams_with_label(ids_pos_np[k], y_np)
                streams_neg = build_int_streams_with_label(ids_neg_np[k], y_np)
                pp, _ = rosa_predict_for_streams(rosa, streams_pos)
                pn, _ = rosa_predict_for_streams(rosa, streams_neg)
                pred_pos[k] = pp; pred_neg[k] = pn

            # 3) Fitnesses (vmapped JAX)
            fitnesses = compute_fitnesses(
                p, pop_keys, bx, by,
                jnp.array(pred_pos), jnp.array(pred_neg),
            )

            # 4) ES gradient + Adam update
            p, opt_state = apply_es_update(p, opt_state, pop_keys, fitnesses)

            # 5) Commit one set of streams (using updated params) to ROSA
            ids_s = extract_ids_one(p, bx)
            streams = build_int_streams_with_label(np.array(ids_s), y_np)
            rosa_commit_streams(rosa, streams)

            ep_fit_abs += float(jnp.mean(jnp.abs(fitnesses)))

        history["fit_abs"].append(ep_fit_abs / num_batches)
        history["sam_states"].append(rosa.n_states())

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == N_EPOCHS - 1:
            test_acc, hit = eval_test(p, rosa, X_test, y_test)
            history["test_acc"].append((epoch, test_acc))
            history["rosa_hit_eval"].append((epoch, hit))
            tag = f"  test={test_acc:.4f}  hit@lbl={hit:.3f}"
        else:
            tag = ""

        logging.info(
            f"epoch {epoch:2d}  |fit|={history['fit_abs'][-1]:.4f}  "
            f"sam={rosa.n_states():,}{tag}  "
            f"{time.perf_counter() - t0:.1f}s"
        )

    elapsed = time.perf_counter() - t0
    final_test, final_hit = eval_test(p, rosa, X_test, y_test)
    final_train, _        = eval_test(p, rosa, X_train[:10_000], y_train[:10_000])

    logging.info(
        f"\nFINAL  test={final_test:.4f}  train(10k)={final_train:.4f}  "
        f"sam={rosa.n_states():,}  hit@lbl={final_hit:.3f}  time={elapsed:.0f}s"
    )

    with open(PROJ_DIR / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in p.items()}, f)
    rosa.save(str(PROJ_DIR / f"rosa_{EXP_NAME}.json"))
    with open(PROJ_DIR / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    row = {
        "experiment": EXP_NAME,
        "name": f"es-rosa-min-head-{DATASET}",
        "dataset": DATASET,
        "test_acc": final_test,
        "train_acc": final_train,
        "rosa_hit_at_label_eval": final_hit,
        "sam_state_count": rosa.n_states(),
        "n_params": n_params(p),
        "epochs": N_EPOCHS,
        "n_train": N_TRAIN,
        "time_s": round(elapsed, 1),
        "history": history,
        "hp": {
            "VOCAB_SIZE": VOCAB_SIZE, "TOKEN_DIM": TOKEN_DIM, "HIDDEN": HIDDEN,
            "K_SAMPLE": K_SAMPLE, "T_STREAM": T_STREAM, "LABEL_POS": LABEL_POS,
            "N_CLASSES_OUT": N_CLASSES_OUT, "DIGIT_OFFSET": DIGIT_OFFSET,
            "NULL_ID": NULL_ID, "N_RESERVED": N_RESERVED,
            "ES_SIGMA": ES_SIGMA, "ES_N_POP": ES_N_POP, "ES_RANK": ES_RANK,
            "ADAM_LR_INIT": ADAM_LR_INIT, "ADAM_LR_END": ADAM_LR_END,
            "BATCH_SIZE": BATCH_SIZE, "N_EPOCHS": N_EPOCHS, "N_TRAIN": N_TRAIN,
            "LAMBDA_DIGIT_BOOST": LAMBDA_DIGIT_BOOST,
        },
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended to {JSONL}")


if __name__ == "__main__":
    main()
