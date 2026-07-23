"""
Universal AR — exp7: variant B (cross-completion), longer training + revised eval.

Training (same as exp6): variant B cross-completion at long context S=256, but
NUM_STEPS is much larger.

Revised EVAL — support vs query, and support comes from TRAIN:
  * The in-context SUPPORT set is drawn from the TRAIN split (partial, labeled).
  * The QUERY (the held-out thing we predict) is drawn from the TEST split.
    (Query is NOT in the support, so the read is a plain cross-attention, no LOO.)
  Label conditions:
    - label_sup      : query = ONE FULL test image (all 784 px) + SMALL support.  ("supervised")
    - label_sup_plus : query = ONE FULL test image      + LARGE support.           ("supervised+"; more context)
    - label_partial  : query = 50%-observed test image  + LARGE support.
  Content:
    - content_ink    : 50%-observed test query; predict held-out pixels from support (ink acc).
  Diagnostics: recall + ref_recall (self, on support).
  Each metric also reported with query drawn from TRAIN (tr_) → generalization gap.

Usage: uv run python projects/universal-ar/scripts/run_experiments.py --bg exp7
"""

import json, time, logging
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

EXP_NAME = "exp7"
JSONL = Path(__file__).parent / "results.jsonl"

# ── hyperparameters ───────────────────────────────────────────────────────────
DATASET = "mnist"
D, N_VAL_BINS, N_CLASSES, POS_DIM, V_REFS = 256, 32, 10, 784, 384
EP_SAMPLES, N_OBS, N_TGT, BATCH_EPISODES = 256, 384, 64, 8   # training episode
NUM_STEPS, EVAL_EVERY, LEARNING_RATE, RANDOM_SEED = 12000, 500, 1e-3, 0
# eval structure
B_EVAL, M_SUPPORT, M_SMALL, Q_QUERY, N_EVAL = 4, 256, 32, 64, 64


# ── HRR ───────────────────────────────────────────────────────────────────────
def fft(x): return jnp.fft.fft(x, axis=-1)
def bind(r, f): return jnp.fft.ifft(fft(r) * fft(f), axis=-1).real
def unbind(b, r): return jnp.fft.ifft(fft(b) * jnp.conj(fft(r)), axis=-1).real


def init() -> Tuple[Dict[str, Any], Dict[str, Array]]:
    config = dict(dataset=DATASET, D=D, n_val_bins=N_VAL_BINS, v_refs=V_REFS,
                  ep_samples=EP_SAMPLES, n_obs=N_OBS, n_tgt=N_TGT, batch_episodes=BATCH_EPISODES,
                  num_steps=NUM_STEPS, m_support=M_SUPPORT, m_small=M_SMALL, q_query=Q_QUERY,
                  learning_rate=LEARNING_RATE, random_seed=RANDOM_SEED,
                  variant="B_cross_completion_longer_supquery")
    k = jax.random.split(jax.random.PRNGKey(RANDOM_SEED), 6); s = 1.0 / D ** 0.5
    params = {
        "r_pos": jax.random.normal(k[0], (POS_DIM, D)) * s,
        "val_emb": jax.random.normal(k[1], (N_VAL_BINS, D)) * s,
        "r_label": jax.random.normal(k[2], (D,)) * s,
        "label_emb": jax.random.normal(k[3], (N_CLASSES, D)) * s,
        "r_ref": jax.random.normal(k[4], (D,)) * s,
        "ref_emb": jax.random.normal(k[5], (V_REFS, D)) * s,
        "log_beta": jnp.array(np.log(20.0), dtype=jnp.float32),
    }
    return config, params


def n_params(params) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))


# ── model ─────────────────────────────────────────────────────────────────────
def binof(imgs, pos):
    v = jnp.take_along_axis(imgs, pos, axis=-1)
    return jnp.floor(v / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)


def build_pix(params, pos, bv):
    Fp = fft(params["r_pos"])[pos]
    oh = jax.nn.one_hot(bv, N_VAL_BINS, dtype=jnp.float32)
    Fv = jnp.einsum("bsnk,kd->bsnd", oh, fft(params["val_emb"]))
    return jnp.fft.ifft(jnp.sum(Fp * Fv, axis=2), axis=-1).real


def entries_at(params, pat, pos):
    f = unbind(pat[:, :, None, :], params["r_pos"][pos])
    return jnp.einsum("bsnd,kd->bsnk", f, params["val_emb"])

def label_at(params, pat):
    return jnp.einsum("bsd,cd->bsc", unbind(pat, params["r_label"]), params["label_emb"])

def ref_at(params, pat):
    return jnp.einsum("bsd,vd->bsv", unbind(pat, params["r_ref"]), params["ref_emb"])


def _norm(x): return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def cross_read_loo(params, key, value, mode):
    """Leave-self-out read over one set (training)."""
    S = key.shape[1]
    sim = jnp.einsum("bqd,bkd->bqk", _norm(key), _norm(key))
    eye = jnp.eye(S)[None]
    A = jax.nn.softmax(jnp.exp(params["log_beta"]) * sim - 1e9 * eye, -1) if mode == "attn" else sim * (1 - eye)
    return jnp.einsum("bqk,bkd->bqd", A, value)


def cross_read_qs(params, qkey, skey, sval, mode):
    """Query set attends over a disjoint support set (eval)."""
    sim = jnp.einsum("bqd,bmd->bqm", _norm(qkey), _norm(skey))
    A = jax.nn.softmax(jnp.exp(params["log_beta"]) * sim, -1) if mode == "attn" else sim
    return jnp.einsum("bqm,bmd->bqd", A, sval)


# ── training loss: cross-completion (same as exp6) ────────────────────────────
def draw(Xnp, ynp, rng, n):
    idx = rng.integers(0, Xnp.shape[0], size=(BATCH_EPISODES if n == EP_SAMPLES else B_EVAL, n))
    return jnp.asarray(Xnp[idx]), jnp.asarray(ynp[idx].astype(np.int32))


def sample_train(key, imgs, labels):
    k = jax.random.split(key, 2)
    order = jnp.argsort(jax.random.uniform(k[0], (BATCH_EPISODES, EP_SAMPLES, POS_DIM)), -1)
    obs = order[..., :N_OBS]; tgt = obs[..., :N_TGT]; cue = obs[..., N_TGT:]
    refs = jnp.argsort(jax.random.uniform(k[1], (BATCH_EPISODES, V_REFS)), -1)[:, :EP_SAMPLES]
    return dict(cue_pos=cue, cue_bin=binof(imgs, cue), tgt_pos=tgt, tgt_bin=binof(imgs, tgt),
                labels=labels, refs=refs)


def loss_fn(params, b):
    P_pix = build_pix(params, b["cue_pos"], b["cue_bin"])
    P_val = P_pix + bind(params["r_label"], params["label_emb"][b["labels"]])
    P_full = P_val + bind(params["r_ref"], params["ref_emb"][b["refs"]])
    phat = cross_read_loo(params, P_pix, P_val, "attn")
    pl = entries_at(params, phat, b["tgt_pos"])
    ll = label_at(params, phat)
    rl = ref_at(params, P_full)
    loss = (optax.softmax_cross_entropy_with_integer_labels(pl, b["tgt_bin"]).mean()
            + optax.softmax_cross_entropy_with_integer_labels(ll, b["labels"]).mean()
            + optax.softmax_cross_entropy_with_integer_labels(rl, b["refs"]).mean())
    return loss, ()


@partial(jax.jit, static_argnums=(0,))
def train_step(optimizer, key, params, opt_state, imgs, labels):
    b = sample_train(key, imgs, labels)
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


# ── revised eval: train support, (test or train) query ────────────────────────
@jax.jit
def eval_step(key, params, sup_imgs, sup_lab, q_imgs, q_lab):
    k = jax.random.split(key, 3)
    # support (partial, labeled, ref-tagged)
    so = jnp.argsort(jax.random.uniform(k[0], (B_EVAL, M_SUPPORT, POS_DIM)), -1)
    sup_obs = so[..., :N_OBS]; sup_rec = sup_obs[..., :N_EVAL]
    refs = jnp.argsort(jax.random.uniform(k[1], (B_EVAL, V_REFS)), -1)[:, :M_SUPPORT]
    P_sp = build_pix(params, sup_obs, binof(sup_imgs, sup_obs))
    P_sv = P_sp + bind(params["r_label"], params["label_emb"][sup_lab])
    P_sf = P_sv + bind(params["r_ref"], params["ref_emb"][refs])
    recall = jnp.mean(jnp.argmax(entries_at(params, P_sf, sup_rec), -1) == binof(sup_imgs, sup_rec))
    ref_recall = jnp.mean(jnp.argmax(ref_at(params, P_sf), -1) == refs)

    # query FULL (all pixels) and PARTIAL
    full_pos = jnp.broadcast_to(jnp.arange(POS_DIM), (B_EVAL, Q_QUERY, POS_DIM))
    P_qfull = build_pix(params, full_pos, binof(q_imgs, full_pos))
    qo = jnp.argsort(jax.random.uniform(k[2], (B_EVAL, Q_QUERY, POS_DIM)), -1)
    q_obs = qo[..., :N_OBS]; q_ho = qo[..., N_OBS:N_OBS + N_EVAL]
    P_qpart = build_pix(params, q_obs, binof(q_imgs, q_obs))

    def lab_acc(qkey, m):
        phat = cross_read_qs(params, qkey, P_sp[:, :m], P_sv[:, :m], "attn")
        return jnp.mean(jnp.argmax(label_at(params, phat), -1) == q_lab)

    ho_bin = binof(q_imgs, q_ho); ink = ho_bin > 0
    phat_p = cross_read_qs(params, P_qpart, P_sp, P_sv, "attn")
    content = jnp.sum((jnp.argmax(entries_at(params, phat_p, q_ho), -1) == ho_bin) & ink) / (jnp.sum(ink) + 1e-6)
    return dict(recall=recall, ref_recall=ref_recall, bg=jnp.mean(ho_bin == 0),
                label_sup=lab_acc(P_qfull, M_SMALL), label_sup_plus=lab_acc(P_qfull, M_SUPPORT),
                label_partial=lab_acc(P_qpart, M_SUPPORT), content_ink=content)


EVAL_KEYS = ("recall", "ref_recall", "bg", "label_sup", "label_sup_plus", "label_partial", "content_ink")


def evaluate(pkey, params, Xsup, ysup, Xq, yq, rng):
    acc = {k: 0.0 for k in EVAL_KEYS}
    for i in range(2):
        si = rng.integers(0, Xsup.shape[0], (B_EVAL, M_SUPPORT))
        qi = rng.integers(0, Xq.shape[0], (B_EVAL, Q_QUERY))
        m = eval_step(jax.random.fold_in(pkey, i), params,
                      jnp.asarray(Xsup[si]), jnp.asarray(ysup[si].astype(np.int32)),
                      jnp.asarray(Xq[qi]), jnp.asarray(yq[qi].astype(np.int32)))
        for k in EVAL_KEYS: acc[k] += float(m[k])
    return {k: v / 2 for k, v in acc.items()}


def train(params, Xtr, ytr, Xte, yte):
    sched = optax.cosine_decay_schedule(LEARNING_RATE, NUM_STEPS)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(RANDOM_SEED + 7); rng = np.random.default_rng(RANDOM_SEED)
    hist = {k: [] for k in ("step", "loss")}
    for k in EVAL_KEYS: hist[k] = []; hist["tr_" + k] = []
    t0 = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        imgs, labels = draw(Xtr, ytr, rng, EP_SAMPLES)
        key, sk = jax.random.split(key)
        params, opt_state, loss = train_step(optimizer, sk, params, opt_state, imgs, labels)
        if step % EVAL_EVERY == 0 or step == 1:
            m = evaluate(jax.random.PRNGKey(1), params, Xtr, ytr, Xte, yte, np.random.default_rng(1))   # test query
            mt = evaluate(jax.random.PRNGKey(2), params, Xtr, ytr, Xtr, ytr, np.random.default_rng(2))  # train query
            hist["step"].append(step); hist["loss"].append(float(loss))
            for k in EVAL_KEYS: hist[k].append(m[k]); hist["tr_" + k].append(mt[k])
            logging.info(
                f"step {step:5d}  loss {float(loss):.3f}  recall {m['recall']:.3f} ref {m['ref_recall']:.3f}  "
                f"label sup/sup+/part {m['label_sup']:.3f}/{m['label_sup_plus']:.3f}/{m['label_partial']:.3f}  "
                f"content_ink {m['content_ink']:.3f} (bg {m['bg']:.3f})  ({time.perf_counter()-t0:.0f}s)")
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
    Xtr = np.asarray(data.X.reshape(data.n_samples, -1), np.float32)
    Xte = np.asarray(data.X_test.reshape(data.n_test_samples, -1), np.float32)
    ytr = np.asarray(data.y); yte = np.asarray(data.y_test)
    logging.info(f"train {Xtr.shape} test {Xte.shape} n_params {n_params(params)}  "
                 f"variant=B longer; support(train) M={M_SUPPORT}/{M_SMALL}, query(test) Q={Q_QUERY}")
    params, hist, elapsed = train(params, Xtr, ytr, Xte, yte)
    final = {k: hist[k][-1] for k in EVAL_KEYS}
    final.update({"tr_" + k: hist["tr_" + k][-1] for k in ("label_sup", "label_sup_plus", "content_ink")})
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {"experiment": EXP_NAME, "name": "variant B, longer; support=train / query=test; supervised & supervised+ label",
           "time_s": elapsed, "n_params": n_params(params), **final, **config, "history": hist}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
