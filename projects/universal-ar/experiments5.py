"""
Universal AR — exp5: variant A (self-recall) + ref self-recall, optimized.

Changes vs exp3
===============
1. EFFICIENCY: episode sampling moved ON-DEVICE (jax.random) and the whole
   train step is jitted — no per-step host→GPU stall. Bigger batch + episodes.
2. REF FIELD: each sample gets a random reference tag from a learned pool of
   size V (random assignment per episode → content-free binding tag). The tag is
   bound into the row pattern and SELF-RECALLED (predict a sample's own ref).
   (Only ref is added; position stays the addressing role.)
3. SCALE: larger S (rows/episode), larger batch, more steps.

Row patterns:
    P_pix  = Σ_{obs} r_pos[pos] ⊛ val_emb[bin]                     (similarity key)
    P_val  = P_pix + r_label ⊛ label_emb[y]                        (cross-read value)
    P_full = P_val + r_ref   ⊛ ref_emb[ref]                        (self-recall target)

Loss (self-recall only, no cross-row term):
    value : unbind r_pos[rec]  from own P_full → predict value
    label : unbind r_label     from own P_full → predict own label
    ref   : unbind r_ref       from own P_full → predict own ref     (NEW)

Eval (cross reads use key=P_pix, value=P_val — ref kept out so label/content stay
comparable to exp3):
    1. retrieval      : value self-recall + ref self-recall
    2. label   gen    : leave-self-out cross read → label
    3. content gen    : leave-self-out cross read → held-out pixel (ink + bg baseline)

Usage:
    uv run python projects/universal-ar/scripts/run_experiments.py --bg exp4
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

EXP_NAME = "exp5"
JSONL = Path(__file__).parent / "results.jsonl"

# ── hyperparameters ───────────────────────────────────────────────────────────
DATASET        = "mnist"
D              = 256
N_VAL_BINS     = 32
N_CLASSES      = 10
POS_DIM        = 784
V_REFS         = 384      # ref pool (>= S)

EP_SAMPLES     = 256      # S: LONG context length (vs 48 in exp4)
N_OBS          = 384      # observed entries per row (~50%)
N_REC          = 64       # self-recall pixel targets
N_EVAL         = 64       # cells scored per row for recall/content
BATCH_EPISODES = 12       # B (smaller to fit the long context)

NUM_STEPS      = 5000
EVAL_EVERY     = 250
EVAL_BATCHES   = 2        # eval episodes = EVAL_BATCHES * B
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
        "dataset": DATASET, "D": D, "n_val_bins": N_VAL_BINS, "v_refs": V_REFS,
        "ep_samples": EP_SAMPLES, "n_obs": N_OBS, "n_rec": N_REC,
        "batch_episodes": BATCH_EPISODES, "num_steps": NUM_STEPS,
        "learning_rate": LEARNING_RATE, "random_seed": RANDOM_SEED,
        "variant": "A_self_recall+ref_longctx",
    }
    k = jax.random.split(jax.random.PRNGKey(RANDOM_SEED), 6)
    s = 1.0 / D ** 0.5
    params = {
        "r_pos":     jax.random.normal(k[0], (POS_DIM, D)) * s,
        "val_emb":   jax.random.normal(k[1], (N_VAL_BINS, D)) * s,
        "r_label":   jax.random.normal(k[2], (D,)) * s,
        "label_emb": jax.random.normal(k[3], (N_CLASSES, D)) * s,
        "r_ref":     jax.random.normal(k[4], (D,)) * s,
        "ref_emb":   jax.random.normal(k[5], (V_REFS, D)) * s,
        "log_beta":  jnp.array(np.log(20.0), dtype=jnp.float32),
    }
    return config, params


def n_params(params) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))


# ── on-device episode sampling ────────────────────────────────────────────────
def sample_batch(key, X, y):
    """All on-device. Returns dict of arrays for one batch of episodes."""
    N = X.shape[0]
    k = jax.random.split(key, 3)
    idx = jax.random.randint(k[0], (BATCH_EPISODES, EP_SAMPLES), 0, N)
    imgs = X[idx]                                              # (B,S,784)
    labels = y[idx].astype(jnp.int32)
    order = jnp.argsort(jax.random.uniform(k[1], (BATCH_EPISODES, EP_SAMPLES, POS_DIM)), axis=-1)
    obs = order[..., :N_OBS]
    ho = order[..., N_OBS:N_OBS + N_EVAL]
    rec = obs[..., :N_REC]
    # distinct ref tags per episode from the pool
    refs = jnp.argsort(jax.random.uniform(k[2], (BATCH_EPISODES, V_REFS)), axis=-1)[:, :EP_SAMPLES]

    def binof(pos):
        v = jnp.take_along_axis(imgs, pos, axis=-1)
        return jnp.floor(v / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)

    return dict(obs_pos=obs, obs_bin=binof(obs), rec_pos=rec, rec_bin=binof(rec),
                ho_pos=ho, ho_bin=binof(ho), labels=labels, refs=refs)


# ── model ─────────────────────────────────────────────────────────────────────
def build_pix(params, pos, binv):
    Fp = fft(params["r_pos"])[pos]
    Fv = fft(params["val_emb"])[binv]
    return jnp.fft.ifft(jnp.sum(Fp * Fv, axis=2), axis=-1).real


def patterns(params, b):
    """Return (P_pix key, P_val cross-value, P_full self-recall)."""
    P_pix = build_pix(params, b["obs_pos"], b["obs_bin"])
    P_val = P_pix + bind(params["r_label"], params["label_emb"][b["labels"]])
    P_full = P_val + bind(params["r_ref"], params["ref_emb"][b["refs"]])
    return P_pix, P_val, P_full


def entries_at(params, pat, pos):
    f = unbind(pat[:, :, None, :], params["r_pos"][pos])
    return jnp.einsum("bsnd,kd->bsnk", f, params["val_emb"])


def label_at(params, pat):
    return jnp.einsum("bsd,cd->bsc", unbind(pat, params["r_label"]), params["label_emb"])


def ref_at(params, pat):
    return jnp.einsum("bsd,vd->bsv", unbind(pat, params["r_ref"]), params["ref_emb"])


def cross_read(params, key, value, mode: str):
    S = key.shape[1]
    kn = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + 1e-6)
    sim = jnp.einsum("bqd,bkd->bqk", kn, kn)
    eye = jnp.eye(S)[None]
    if mode == "attn":
        A = jax.nn.softmax(jnp.exp(params["log_beta"]) * sim - 1e9 * eye, axis=-1)
    else:
        A = sim * (1.0 - eye)
    return jnp.einsum("bqk,bkd->bqd", A, value)


# ── loss: SELF-RECALL of value + label + ref (no cross-row term) ──────────────
def loss_fn(params, b):
    _, _, P_full = patterns(params, b)
    pix_logits = entries_at(params, P_full, b["rec_pos"])
    pix_loss = optax.softmax_cross_entropy_with_integer_labels(pix_logits, b["rec_bin"]).mean()
    lab_logits = label_at(params, P_full)
    lab_loss = optax.softmax_cross_entropy_with_integer_labels(lab_logits, b["labels"]).mean()
    ref_logits = ref_at(params, P_full)
    ref_loss = optax.softmax_cross_entropy_with_integer_labels(ref_logits, b["refs"]).mean()
    loss = pix_loss + lab_loss + ref_loss
    aux = (jnp.mean(jnp.argmax(pix_logits, -1) == b["rec_bin"]),
           jnp.mean(jnp.argmax(lab_logits, -1) == b["labels"]),
           jnp.mean(jnp.argmax(ref_logits, -1) == b["refs"]))
    return loss, aux


@partial(jax.jit, static_argnums=(0,))
def train_step(optimizer, key, params, opt_state, X, y):
    b = sample_batch(key, X, y)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, aux


@jax.jit
def eval_step(key, params, X, y):
    b = sample_batch(key, X, y)
    P_pix, P_val, P_full = patterns(params, b)
    recall = jnp.mean(jnp.argmax(entries_at(params, P_full, b["rec_pos"]), -1) == b["rec_bin"])
    ref_recall = jnp.mean(jnp.argmax(ref_at(params, P_full), -1) == b["refs"])
    ho_bin = b["ho_bin"]; ink = ho_bin > 0
    bg = jnp.mean(ho_bin == 0)

    def gen(mode):
        phat = cross_read(params, P_pix, P_val, mode)
        lab = jnp.mean(jnp.argmax(label_at(params, phat), -1) == b["labels"])
        pred = jnp.argmax(entries_at(params, phat, b["ho_pos"]), -1)
        return lab, jnp.sum((pred == ho_bin) & ink) / (jnp.sum(ink) + 1e-6)

    la, ia = gen("attn")
    ll, il = gen("linear")
    return dict(recall=recall, ref_recall=ref_recall, bg=bg,
                label_attn=la, ink_attn=ia, label_lin=ll, ink_lin=il)


EVAL_KEYS = ("recall", "ref_recall", "bg", "label_attn", "ink_attn", "label_lin", "ink_lin")


def evaluate(key, params, X, y):
    acc = {k: 0.0 for k in EVAL_KEYS}
    for i in range(EVAL_BATCHES):
        m = eval_step(jax.random.fold_in(key, i), params, X, y)
        for k in EVAL_KEYS:
            acc[k] += float(m[k])
    return {k: v / EVAL_BATCHES for k, v in acc.items()}


# ── train ─────────────────────────────────────────────────────────────────────
def train(params, Xtr, ytr, Xte, yte):
    sched = optax.cosine_decay_schedule(LEARNING_RATE, NUM_STEPS)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(RANDOM_SEED + 7)
    hist = {k: [] for k in ("step", "loss")}
    for k in EVAL_KEYS:
        hist[k] = []; hist["tr_" + k] = []
    t0 = time.perf_counter()

    for step in range(1, NUM_STEPS + 1):
        key, sk = jax.random.split(key)
        params, opt_state, loss, aux = train_step(optimizer, sk, params, opt_state, Xtr, ytr)

        if step % EVAL_EVERY == 0 or step == 1:
            m = evaluate(jax.random.PRNGKey(1), params, Xte, yte)
            mt = evaluate(jax.random.PRNGKey(2), params, Xtr, ytr)
            hist["step"].append(step); hist["loss"].append(float(loss))
            for k in EVAL_KEYS:
                hist[k].append(m[k]); hist["tr_" + k].append(mt[k])
            logging.info(
                f"step {step:5d}  loss {float(loss):.3f}  "
                f"recall {m['recall']:.3f}  ref_recall {m['ref_recall']:.3f}  "
                f"label[a] tr/te {mt['label_attn']:.3f}/{m['label_attn']:.3f}  "
                f"ink[a] tr/te {mt['ink_attn']:.3f}/{m['ink_attn']:.3f}  "
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
    Xtr = jnp.asarray(data.X.reshape(data.n_samples, -1), dtype=jnp.float32)
    Xte = jnp.asarray(data.X_test.reshape(data.n_test_samples, -1), dtype=jnp.float32)
    ytr = jnp.asarray(data.y); yte = jnp.asarray(data.y_test)
    logging.info(f"train {Xtr.shape}  test {Xte.shape}  n_params {n_params(params)}  "
                 f"B={BATCH_EPISODES} S={EP_SAMPLES} V_refs={V_REFS}  variant=A+ref")

    params, hist, elapsed = train(params, Xtr, ytr, Xte, yte)

    final = {k: hist[k][-1] for k in EVAL_KEYS}
    final.update({"tr_" + k: hist["tr_" + k][-1] for k in ("recall", "label_attn", "ink_attn")})
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {
        "experiment": EXP_NAME,
        "name": "variant A + ref, long context S=256",
        "time_s": elapsed, "n_params": n_params(params),
        **final, **config, "history": hist,
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
