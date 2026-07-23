"""
Universal AR — exp3: version A — pure self-recall training, test if cross-sample
completion EMERGES.

Question
========
Train the memory ONLY to recall its own given data (auto-associative storage,
no other rows in the loss). Then measure whether cross-sample label/content
completion emerges anyway. This is the strict "generalise first" test: the
cross-row kernel is never trained.

Store label INTO the row pattern (so there is something to self-recall):
    P_full = Σ_{obs pos} r_pos[pos] ⊛ val_emb[bin]  +  r_label ⊛ label_emb[y]

Loss (self-recall only, no cross-row term):
    pixel:  unbind r_pos[obs] from the row's OWN P_full → predict its value
    label:  unbind r_label     from the row's OWN P_full → predict its own label

Eval (3 tests; cross reads are NOT trained):
    1. retrieval          — self-recall observed entries      (trained)
    2. label  generalise  — leave-self-out cross read → label (emergent?)
    3. content generalise — leave-self-out cross read → pixel (emergent?)
       reported as raw acc, ink-only acc (true bin>0), and the
       "predict background (bin 0)" baseline, so it is interpretable.

Similarity key = pixels-only pattern; aggregated value = P_full.

Usage:
    uv run python projects/universal-ar/scripts/run_experiments.py --bg exp3
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

EXP_NAME = "exp3"
JSONL = Path(__file__).parent / "results.jsonl"

# ── hyperparameters ───────────────────────────────────────────────────────────
DATASET        = "mnist"
D              = 256
N_VAL_BINS     = 32
N_CLASSES      = 10
POS_DIM        = 784

EP_SAMPLES     = 32
N_OBS          = 384      # observed entries per row (~50%)
N_REC          = 64       # observed cells self-recalled per step (loss target)
N_EVAL         = 64       # cells scored per row for recall / content tests
BATCH_EPISODES = 8

NUM_STEPS      = 3000
EVAL_EVERY     = 200
EVAL_EPISODES  = 64
LEARNING_RATE  = 1e-3
RANDOM_SEED    = 0


# ── HRR primitives ────────────────────────────────────────────────────────────
def fft(x: Array) -> Array:
    return jnp.fft.fft(x, axis=-1)


def bind(role: Array, filler: Array) -> Array:
    return jnp.fft.ifft(fft(role) * fft(filler), axis=-1).real


def unbind(bound: Array, role: Array) -> Array:
    return jnp.fft.ifft(fft(bound) * jnp.conj(fft(role)), axis=-1).real


# ── params ────────────────────────────────────────────────────────────────────
def init() -> Tuple[Dict[str, Any], Dict[str, Array]]:
    config = {
        "dataset": DATASET, "D": D, "n_val_bins": N_VAL_BINS, "ep_samples": EP_SAMPLES,
        "n_obs": N_OBS, "n_rec": N_REC, "batch_episodes": BATCH_EPISODES,
        "num_steps": NUM_STEPS, "learning_rate": LEARNING_RATE, "random_seed": RANDOM_SEED,
        "variant": "A_self_recall_only",
    }
    k = jax.random.split(jax.random.PRNGKey(RANDOM_SEED), 5)
    s = 1.0 / D ** 0.5
    params = {
        "r_pos":     jax.random.normal(k[0], (POS_DIM, D)) * s,
        "val_emb":   jax.random.normal(k[1], (N_VAL_BINS, D)) * s,
        "r_label":   jax.random.normal(k[2], (D,)) * s,
        "label_emb": jax.random.normal(k[3], (N_CLASSES, D)) * s,
        "log_beta":  jnp.array(np.log(20.0), dtype=jnp.float32),
    }
    return config, params


def n_params(params) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))


# ── episode sampling ──────────────────────────────────────────────────────────
def sample_episodes(Xflat, y, B, S, rng):
    N = Xflat.shape[0]
    idx = rng.integers(0, N, size=(B, S))
    imgs = Xflat[idx]
    labels = y[idx].astype(np.int32)
    perm = rng.random((B, S, POS_DIM)).argsort(-1)
    obs = perm[..., :N_OBS]
    ho = perm[..., N_OBS:]
    rec = obs[..., :N_REC]
    hoe = ho[..., :N_EVAL]

    def binof(pos):
        v = np.take_along_axis(imgs, pos, axis=-1)
        return np.floor(v / 255.0 * (N_VAL_BINS - 1)).astype(np.int32)

    return {k: jnp.asarray(v) for k, v in dict(
        obs_pos=obs, obs_bin=binof(obs),
        rec_pos=rec, rec_bin=binof(rec),
        ho_pos=hoe, ho_bin=binof(hoe),
        labels=labels).items()}


# ── model ─────────────────────────────────────────────────────────────────────
def build_pix(params, pos, binv):
    """Pixels-only row pattern (similarity key).  → (B,S,D)."""
    Fp = fft(params["r_pos"])[pos]
    Fv = fft(params["val_emb"])[binv]
    return jnp.fft.ifft(jnp.sum(Fp * Fv, axis=2), axis=-1).real


def build_full(params, pos, binv, labels):
    """Row pattern with the label bound in (stored value)."""
    return build_pix(params, pos, binv) + bind(params["r_label"], params["label_emb"][labels])


def entries_at(params, pat, pos):
    f = unbind(pat[:, :, None, :], params["r_pos"][pos])
    return jnp.einsum("bsnd,kd->bsnk", f, params["val_emb"])


def label_at(params, pat):
    f = unbind(pat, params["r_label"])
    return jnp.einsum("bsd,cd->bsc", f, params["label_emb"])


def cross_read(params, key, value, mode: str):
    """Leave-self-out read: similarity over keys, aggregate values.  → (B,S,D)."""
    S = key.shape[1]
    kn = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + 1e-6)
    sim = jnp.einsum("bqd,bkd->bqk", kn, kn)
    eye = jnp.eye(S)[None]
    if mode == "attn":
        A = jax.nn.softmax(jnp.exp(params["log_beta"]) * sim - 1e9 * eye, axis=-1)
    else:
        A = sim * (1.0 - eye)
    return jnp.einsum("bqk,bkd->bqd", A, value)


# ── loss: SELF-RECALL ONLY (no cross-row term) ────────────────────────────────
def loss_fn(params, obs_pos, obs_bin, rec_pos, rec_bin, labels):
    P_full = build_full(params, obs_pos, obs_bin, labels)   # each row's own pattern
    pix_logits = entries_at(params, P_full, rec_pos)        # recall own observed pixels
    pix_loss = optax.softmax_cross_entropy_with_integer_labels(pix_logits, rec_bin).mean()
    lab_logits = label_at(params, P_full)                   # recall own label
    lab_loss = optax.softmax_cross_entropy_with_integer_labels(lab_logits, labels).mean()
    loss = pix_loss + lab_loss
    aux = (jnp.mean(jnp.argmax(pix_logits, -1) == rec_bin),
           jnp.mean(jnp.argmax(lab_logits, -1) == labels))
    return loss, aux


@partial(jax.jit, static_argnums=(0,))
def train_step(optimizer, params, opt_state, obs_pos, obs_bin, rec_pos, rec_bin, labels):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, obs_pos, obs_bin, rec_pos, rec_bin, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, aux


@jax.jit
def eval_all(params, b):
    key = build_pix(params, b["obs_pos"], b["obs_bin"])          # similarity key (pixels)
    value = build_full(params, b["obs_pos"], b["obs_bin"], b["labels"])  # stored value

    # test 1: self-recall of observed cells (trained)
    recall = jnp.mean(jnp.argmax(entries_at(params, value, b["rec_pos"]), -1) == b["rec_bin"])

    ho_bin = b["ho_bin"]
    ink = ho_bin > 0
    bg = jnp.mean(ho_bin == 0)   # "predict background" baseline

    def gen(mode):
        phat = cross_read(params, key, value, mode)
        lab = jnp.mean(jnp.argmax(label_at(params, phat), -1) == b["labels"])
        pred = jnp.argmax(entries_at(params, phat, b["ho_pos"]), -1)
        content = jnp.mean(pred == ho_bin)
        content_ink = jnp.sum((pred == ho_bin) & ink) / (jnp.sum(ink) + 1e-6)
        return lab, content, content_ink

    la, ca, ia = gen("attn")
    ll, cl, il = gen("linear")
    return dict(recall=recall, bg=bg,
                label_attn=la, content_attn=ca, ink_attn=ia,
                label_lin=ll, content_lin=cl, ink_lin=il)


def evaluate(params, Xflat, y, rng):
    keys = ("recall", "bg", "label_attn", "content_attn", "ink_attn",
            "label_lin", "content_lin", "ink_lin")
    acc = {k: 0.0 for k in keys}
    n = EVAL_EPISODES // BATCH_EPISODES
    for _ in range(n):
        b = sample_episodes(Xflat, y, BATCH_EPISODES, EP_SAMPLES, rng)
        m = eval_all(params, b)
        for k in keys:
            acc[k] += float(m[k])
    return {k: v / n for k, v in acc.items()}


# ── train ─────────────────────────────────────────────────────────────────────
EVAL_KEYS = ("recall", "bg", "label_attn", "content_attn", "ink_attn",
             "label_lin", "content_lin", "ink_lin")


def train(params, Xtr, ytr, Xte, yte):
    sched = optax.cosine_decay_schedule(LEARNING_RATE, NUM_STEPS)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(RANDOM_SEED)
    ev_rng_te = np.random.default_rng(RANDOM_SEED + 1)
    ev_rng_tr = np.random.default_rng(RANDOM_SEED + 2)
    hist = {k: [] for k in ("step", "loss")}
    for k in EVAL_KEYS:
        hist[k] = []; hist["tr_" + k] = []   # test_* (bare) and train_* (tr_) metrics
    t0 = time.perf_counter()

    for step in range(1, NUM_STEPS + 1):
        b = sample_episodes(Xtr, ytr, BATCH_EPISODES, EP_SAMPLES, rng)
        params, opt_state, loss, aux = train_step(
            optimizer, params, opt_state,
            b["obs_pos"], b["obs_bin"], b["rec_pos"], b["rec_bin"], b["labels"])

        if step % EVAL_EVERY == 0 or step == 1:
            m = evaluate(params, Xte, yte, ev_rng_te)     # test-set episodes
            mt = evaluate(params, Xtr, ytr, ev_rng_tr)    # train-set episodes
            hist["step"].append(step); hist["loss"].append(float(loss))
            for k in EVAL_KEYS:
                hist[k].append(m[k]); hist["tr_" + k].append(mt[k])
            logging.info(
                f"step {step:5d}  loss {float(loss):.3f}  "
                f"recall tr/te {mt['recall']:.3f}/{m['recall']:.3f}  "
                f"label[a] tr/te {mt['label_attn']:.3f}/{m['label_attn']:.3f}  "
                f"content_ink[a] tr/te {mt['ink_attn']:.3f}/{m['ink_attn']:.3f}  "
                f"(bg={m['bg']:.3f})  ({time.perf_counter()-t0:.0f}s)")

    return params, hist, time.perf_counter() - t0


if __name__ == "__main__":
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
    logging.info(f"train {Xtr.shape}  test {Xte.shape}  n_params {n_params(params)}  "
                 f"variant=A(self-recall only)")

    params, hist, elapsed = train(params, Xtr, ytr, Xte, yte)

    final = {k: hist[k][-1] for k in EVAL_KEYS}
    final.update({"tr_" + k: hist["tr_" + k][-1] for k in ("recall", "label_attn", "ink_attn")})
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {
        "experiment": EXP_NAME,
        "name": "variant A: self-recall-only training; test emergent cross completion",
        "time_s": elapsed, "n_params": n_params(params),
        **final, **config, "history": hist,
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
