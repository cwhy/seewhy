"""
Universal AR — exp6: variant B (cross-completion) at LONG context (S=256).

Same architecture, data, and eval as exp5 (variant A, S=256); the ONLY change is
the training loss: instead of self-recall, the pixel and label targets are
predicted from the OTHER rows via a leave-self-out cross read (cross-completion).
This trains the cross-sample kernel — the question is whether a trained kernel
makes long context HELP (vs variant A, where label gen dropped 0.40→0.15 as S grew
32→256 precisely because its kernel is untrained).

Training loss (cross-completion of observed entries + ref self-recall):
    pixel : predict a held-back subset of observed pixels (tgt, excluded from the
            row's cue) from the OTHER rows                → CE
    label : predict each row's label from the OTHER rows (leave-one-out)  → CE
    ref   : self-recall each row's own ref tag                            → CE

Eval is identical to exp5 (build P from all observed): test1 self-recall +
ref-recall; test2 label (cross); test3 held-out pixel (cross, ink + bg).

Usage: uv run python projects/universal-ar/scripts/run_experiments.py --bg exp6
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

EXP_NAME = "exp6"
JSONL = Path(__file__).parent / "results.jsonl"

# ── hyperparameters ───────────────────────────────────────────────────────────
DATASET        = "mnist"
D              = 256
N_VAL_BINS     = 32
N_CLASSES      = 10
POS_DIM        = 784
V_REFS         = 384

EP_SAMPLES     = 256      # S: LONG context
N_OBS          = 384      # observed per row (~50%)
N_TGT          = 64       # cross-completion pixel targets (⊂ observed, excluded from cue)
N_EVAL         = 64
BATCH_EPISODES = 8

NUM_STEPS      = 5000
EVAL_EVERY     = 250
EVAL_BATCHES   = 2
LEARNING_RATE  = 1e-3
RANDOM_SEED    = 0


# ── HRR primitives ────────────────────────────────────────────────────────────
def fft(x): return jnp.fft.fft(x, axis=-1)
def bind(role, filler): return jnp.fft.ifft(fft(role) * fft(filler), axis=-1).real
def unbind(bound, role): return jnp.fft.ifft(fft(bound) * jnp.conj(fft(role)), axis=-1).real


# ── params ────────────────────────────────────────────────────────────────────
def init() -> Tuple[Dict[str, Any], Dict[str, Array]]:
    config = {
        "dataset": DATASET, "D": D, "n_val_bins": N_VAL_BINS, "v_refs": V_REFS,
        "ep_samples": EP_SAMPLES, "n_obs": N_OBS, "n_tgt": N_TGT,
        "batch_episodes": BATCH_EPISODES, "num_steps": NUM_STEPS,
        "learning_rate": LEARNING_RATE, "random_seed": RANDOM_SEED,
        "variant": "B_cross_completion+ref_longctx",
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


# ── host image gather + on-device per-episode sampling ────────────────────────
def draw(Xnp, ynp, rng):
    idx = rng.integers(0, Xnp.shape[0], size=(BATCH_EPISODES, EP_SAMPLES))
    return jnp.asarray(Xnp[idx]), jnp.asarray(ynp[idx].astype(np.int32))


def sample_batch(key, imgs, labels):
    k = jax.random.split(key, 2)
    order = jnp.argsort(jax.random.uniform(k[0], (BATCH_EPISODES, EP_SAMPLES, POS_DIM)), axis=-1)
    obs = order[..., :N_OBS]
    ho = order[..., N_OBS:N_OBS + N_EVAL]
    tgt = obs[..., :N_TGT]          # cross-completion pixel targets
    cue = obs[..., N_TGT:]          # builds the training key (excludes targets)
    rec = obs[..., :N_EVAL]         # eval self-recall subset
    refs = jnp.argsort(jax.random.uniform(k[1], (BATCH_EPISODES, V_REFS)), axis=-1)[:, :EP_SAMPLES]

    def binof(pos):
        v = jnp.take_along_axis(imgs, pos, axis=-1)
        return jnp.floor(v / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)

    return dict(obs_pos=obs, obs_bin=binof(obs), cue_pos=cue, cue_bin=binof(cue),
                tgt_pos=tgt, tgt_bin=binof(tgt), rec_pos=rec, rec_bin=binof(rec),
                ho_pos=ho, ho_bin=binof(ho), labels=labels, refs=refs)


# ── model ─────────────────────────────────────────────────────────────────────
def build_pix(params, pos, binv):
    Fp = fft(params["r_pos"])[pos]
    onehot = jax.nn.one_hot(binv, N_VAL_BINS, dtype=jnp.float32)
    Fv = jnp.einsum("bsnk,kd->bsnd", onehot, fft(params["val_emb"]))
    return jnp.fft.ifft(jnp.sum(Fp * Fv, axis=2), axis=-1).real


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


# ── loss: CROSS-COMPLETION of pixels + label, ref self-recall ─────────────────
def loss_fn(params, b):
    P_pix = build_pix(params, b["cue_pos"], b["cue_bin"])              # key from cue
    P_val = P_pix + bind(params["r_label"], params["label_emb"][b["labels"]])
    P_full = P_val + bind(params["r_ref"], params["ref_emb"][b["refs"]])
    phat = cross_read(params, P_pix, P_val, "attn")                   # leave-self-out
    pix_logits = entries_at(params, phat, b["tgt_pos"])               # pixels from OTHERS
    pix_loss = optax.softmax_cross_entropy_with_integer_labels(pix_logits, b["tgt_bin"]).mean()
    lab_logits = label_at(params, phat)                               # label from OTHERS
    lab_loss = optax.softmax_cross_entropy_with_integer_labels(lab_logits, b["labels"]).mean()
    ref_logits = ref_at(params, P_full)                              # ref self-recall
    ref_loss = optax.softmax_cross_entropy_with_integer_labels(ref_logits, b["refs"]).mean()
    loss = pix_loss + lab_loss + ref_loss
    aux = (jnp.mean(jnp.argmax(pix_logits, -1) == b["tgt_bin"]),
           jnp.mean(jnp.argmax(lab_logits, -1) == b["labels"]),
           jnp.mean(jnp.argmax(ref_logits, -1) == b["refs"]))
    return loss, aux


@partial(jax.jit, static_argnums=(0,))
def train_step(optimizer, key, params, opt_state, imgs, labels):
    b = sample_batch(key, imgs, labels)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, aux


@jax.jit
def eval_step(key, params, imgs, labels):
    b = sample_batch(key, imgs, labels)
    P_pix = build_pix(params, b["obs_pos"], b["obs_bin"])             # eval: full observed
    P_val = P_pix + bind(params["r_label"], params["label_emb"][b["labels"]])
    P_full = P_val + bind(params["r_ref"], params["ref_emb"][b["refs"]])
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


def evaluate(pkey, params, Xnp, ynp, rng):
    acc = {k: 0.0 for k in EVAL_KEYS}
    for i in range(EVAL_BATCHES):
        imgs, labels = draw(Xnp, ynp, rng)
        m = eval_step(jax.random.fold_in(pkey, i), params, imgs, labels)
        for k in EVAL_KEYS:
            acc[k] += float(m[k])
    return {k: v / EVAL_BATCHES for k, v in acc.items()}


# ── train ─────────────────────────────────────────────────────────────────────
def train(params, Xtr, ytr, Xte, yte):
    sched = optax.cosine_decay_schedule(LEARNING_RATE, NUM_STEPS)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(RANDOM_SEED + 7)
    rng = np.random.default_rng(RANDOM_SEED)
    hist = {k: [] for k in ("step", "loss")}
    for k in EVAL_KEYS:
        hist[k] = []; hist["tr_" + k] = []
    t0 = time.perf_counter()

    for step in range(1, NUM_STEPS + 1):
        imgs, labels = draw(Xtr, ytr, rng)
        key, sk = jax.random.split(key)
        params, opt_state, loss, aux = train_step(optimizer, sk, params, opt_state, imgs, labels)

        if step % EVAL_EVERY == 0 or step == 1:
            m = evaluate(jax.random.PRNGKey(1), params, Xte, yte, np.random.default_rng(1))
            mt = evaluate(jax.random.PRNGKey(2), params, Xtr, ytr, np.random.default_rng(2))
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
    Xtr = np.asarray(data.X.reshape(data.n_samples, -1), dtype=np.float32)
    Xte = np.asarray(data.X_test.reshape(data.n_test_samples, -1), dtype=np.float32)
    ytr = np.asarray(data.y); yte = np.asarray(data.y_test)
    logging.info(f"train {Xtr.shape}  test {Xte.shape}  n_params {n_params(params)}  "
                 f"B={BATCH_EPISODES} S={EP_SAMPLES}  variant=B(cross)+ref, long ctx")

    params, hist, elapsed = train(params, Xtr, ytr, Xte, yte)

    final = {k: hist[k][-1] for k in EVAL_KEYS}
    final.update({"tr_" + k: hist["tr_" + k][-1] for k in ("recall", "label_attn", "ink_attn")})
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {
        "experiment": EXP_NAME,
        "name": "variant B (cross-completion) + ref, long context S=256",
        "time_s": elapsed, "n_params": n_params(params),
        **final, **config, "history": hist,
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
