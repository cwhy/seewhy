"""
Universal AR — exp1 (token-level, v1): masked-token completion with a small
transformer over the (pos, value, ref) token bag.

Every datum is a token (pos, value, ref). The label is a token at pos_label
(value = class). The context is a flat bag of tokens; a small L-layer transformer
(set attention, no causal mask) predicts the value of MASKed query tokens.

Value vocab (content ids the head predicts over):
    0 .. K-1        pixel bins
    K .. K+9        class labels
  ( K+10 = MASK, an input-only embedding, never a target )

Episode (training): S samples from train; each sample gets a random ref from a
pool. Per sample: N_CTX observed pixel tokens (value given) + N_QP masked pixel
tokens (predict) + one label token (given for ~half the samples, MASKed→predict
for the rest). Loss = CE on masked tokens. Hold-out is honored: a masked token's
pos/value/ref each appear individually elsewhere (other samples' pixels at that
pos, other tokens of that value, the sample's own other pixels for that ref).

Eval: support samples from TRAIN (labels given), query samples from TEST
(label MASKed + some pixels MASKed) → label + content(ink) generalization.

Usage: uv run python projects/universal-ar/scripts/run_experiments.py --bg exp1
"""

import json, time, logging
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax
from jax import Array

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp1"
JSONL = Path(__file__).parent / "results.jsonl"

# ── hyperparameters ───────────────────────────────────────────────────────────
DATASET = "mnist"
K = 32                       # pixel bins
N_CLASSES = 10
POS_PIX = 784
POS_LABEL = 784              # special position id
N_POS = 785
N_CONTENT = K + N_CLASSES    # 42 predictable value ids
MASK_ID = N_CONTENT          # 42
N_VAL = N_CONTENT + 1        # 43 input value embeddings
V_REFS = 64

D, N_LAYERS, N_HEADS, MLP_MULT = 256, 4, 8, 4
S_SAMPLES, N_CTX, N_QP = 16, 48, 16          # per training episode
LABEL_QUERY_FRAC = 0.5
BATCH = 16
# eval episode
M_SUP, Q_QRY = 16, 8

NUM_STEPS, EVAL_EVERY, LR, SEED = 4000, 250, 3e-4, 0


# ── params ────────────────────────────────────────────────────────────────────
def init():
    config = dict(dataset=DATASET, K=K, D=D, n_layers=N_LAYERS, n_heads=N_HEADS,
                  s_samples=S_SAMPLES, n_ctx=N_CTX, n_qp=N_QP, v_refs=V_REFS,
                  batch=BATCH, num_steps=NUM_STEPS, lr=LR, seed=SEED, variant="token_transformer_v1")
    g = jax.random.split(jax.random.PRNGKey(SEED), 3 + N_LAYERS * 6 + 3)
    i = iter(g)
    lin = lambda k, shp: jax.random.normal(k, shp) * (1.0 / shp[0] ** 0.5)
    p = {"pos_emb": jax.random.normal(next(i), (N_POS, D)) * 0.02,
         "val_emb": jax.random.normal(next(i), (N_VAL, D)) * 0.02,
         "ref_emb": jax.random.normal(next(i), (V_REFS, D)) * 0.02,
         "layers": []}
    for _ in range(N_LAYERS):
        p["layers"].append(dict(
            ln1_g=jnp.ones(D), ln1_b=jnp.zeros(D),
            Wqkv=lin(next(i), (D, 3 * D)), Wo=lin(next(i), (D, D)),
            ln2_g=jnp.ones(D), ln2_b=jnp.zeros(D),
            W1=lin(next(i), (D, MLP_MULT * D)), b1=jnp.zeros(MLP_MULT * D),
            W2=lin(next(i), (MLP_MULT * D, D)), b2=jnp.zeros(D)))
    p["lnf_g"] = jnp.ones(D); p["lnf_b"] = jnp.zeros(D)
    p["head_W"] = lin(next(i), (D, N_CONTENT)); p["head_b"] = jnp.zeros(N_CONTENT)
    return config, p


def n_params(p): return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


# ── transformer ───────────────────────────────────────────────────────────────
def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def mha(x, L):
    B, N, _ = x.shape
    qkv = x @ L["Wqkv"]
    q, k, v = jnp.split(qkv, 3, -1)
    sh = lambda t: t.reshape(B, N, N_HEADS, D // N_HEADS).transpose(0, 2, 1, 3)
    q, k, v = sh(q), sh(k), sh(v)
    a = jax.nn.softmax(jnp.einsum("bhid,bhjd->bhij", q, k) / (D // N_HEADS) ** 0.5, -1)
    o = jnp.einsum("bhij,bhjd->bhid", a, v).transpose(0, 2, 1, 3).reshape(B, N, D)
    return o @ L["Wo"]


def onehot_mm(ids, table, n):
    return jnp.einsum("bnk,kd->bnd", jax.nn.one_hot(ids, n, dtype=jnp.float32), table)


def forward(p, pos, val, ref):
    x = onehot_mm(pos, p["pos_emb"], N_POS) + onehot_mm(val, p["val_emb"], N_VAL) + onehot_mm(ref, p["ref_emb"], V_REFS)
    for L in p["layers"]:
        x = x + mha(ln(x, L["ln1_g"], L["ln1_b"]), L)
        h = jax.nn.gelu(ln(x, L["ln2_g"], L["ln2_b"]) @ L["W1"] + L["b1"]) @ L["W2"] + L["b2"]
        x = x + h
    x = ln(x, p["lnf_g"], p["lnf_b"])
    return x @ p["head_W"] + p["head_b"]                       # (B,N,N_CONTENT)


def loss_fn(p, pos, val, ref, target, isq):
    logits = forward(p, pos, val, ref)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(target, 0, N_CONTENT - 1))
    return (ce * isq).sum() / (isq.sum() + 1e-6), ()


@partial(jax.jit, static_argnums=(0,))
def train_step(opt, p, st, pos, val, ref, target, isq):
    (loss, _), g = jax.value_and_grad(loss_fn, has_aux=True)(p, pos, val, ref, target, isq)
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, loss


# ── episode building (host) ───────────────────────────────────────────────────
def _bin(px): return np.floor(px / 255.0 * (K - 1)).astype(np.int32)


def build_train(Xtr, ytr, rng):
    B, S = BATCH, S_SAMPLES
    T = S * (N_CTX + N_QP + 1)
    pos = np.zeros((B, T), np.int32); val = np.zeros((B, T), np.int32)
    ref = np.zeros((B, T), np.int32); tgt = -np.ones((B, T), np.int32); isq = np.zeros((B, T), np.float32)
    for b in range(B):
        idx = rng.integers(0, Xtr.shape[0], S)
        refs = rng.permutation(V_REFS)[:S]
        lab_q = rng.random(S) < LABEL_QUERY_FRAC
        t = 0
        for s in range(S):
            img = Xtr[idx[s]]; cls = int(ytr[idx[s]]); r = int(refs[s])
            perm = rng.permutation(POS_PIX)[:N_CTX + N_QP]
            cpos, qpos = perm[:N_CTX], perm[N_CTX:]
            # context pixels
            for ppos in cpos:
                pos[b, t], val[b, t], ref[b, t] = ppos, int(_bin(img[ppos])), r; t += 1
            # query pixels (masked)
            for ppos in qpos:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = ppos, MASK_ID, r, int(_bin(img[ppos])), 1.0; t += 1
            # label token
            if lab_q[s]:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = POS_LABEL, MASK_ID, r, K + cls, 1.0
            else:
                pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + cls, r
            t += 1
    return tuple(jnp.asarray(a) for a in (pos, val, ref, tgt, isq))


def build_eval(Xsup, ysup, Xq, yq, rng):
    """Support from Xsup (labels given), query from Xq (label + pixels masked)."""
    B = BATCH
    T = M_SUP * (N_CTX + 1) + Q_QRY * (N_CTX + N_QP + 1)
    pos = np.zeros((B, T), np.int32); val = np.zeros((B, T), np.int32); ref = np.zeros((B, T), np.int32)
    tgt = -np.ones((B, T), np.int32); isq = np.zeros((B, T), np.float32)
    is_lab = np.zeros((B, T), np.float32)   # query token is a label (else pixel)
    for b in range(B):
        refs = rng.permutation(V_REFS)[:M_SUP + Q_QRY]; t = 0
        si = rng.integers(0, Xsup.shape[0], M_SUP)
        for s in range(M_SUP):
            img = Xsup[si[s]]; cls = int(ysup[si[s]]); r = int(refs[s])
            for pp in rng.permutation(POS_PIX)[:N_CTX]:
                pos[b, t], val[b, t], ref[b, t] = pp, int(_bin(img[pp])), r; t += 1
            pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + cls, r; t += 1
        qi = rng.integers(0, Xq.shape[0], Q_QRY)
        for s in range(Q_QRY):
            img = Xq[qi[s]]; cls = int(yq[qi[s]]); r = int(refs[M_SUP + s])
            perm = rng.permutation(POS_PIX)[:N_CTX + N_QP]
            for pp in perm[:N_CTX]:
                pos[b, t], val[b, t], ref[b, t] = pp, int(_bin(img[pp])), r; t += 1
            for pp in perm[N_CTX:]:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, int(_bin(img[pp])), 1.0; t += 1
            pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t], is_lab[b, t] = POS_LABEL, MASK_ID, r, K + cls, 1.0, 1.0; t += 1
    return tuple(jnp.asarray(a) for a in (pos, val, ref, tgt, isq, is_lab))


@jax.jit
def eval_metrics(p, pos, val, ref, tgt, isq, is_lab):
    pred = jnp.argmax(forward(p, pos, val, ref), -1)
    correct = (pred == tgt)
    is_pix = isq * (1 - is_lab)
    lab_acc = (correct * is_lab).sum() / (is_lab.sum() + 1e-6)
    # content ink: pixel query tokens whose true bin > 0
    ink = is_pix * (tgt > 0)
    content_ink = (correct * ink).sum() / (ink.sum() + 1e-6)
    bg = (is_pix * (tgt == 0)).sum() / (is_pix.sum() + 1e-6)
    return lab_acc, content_ink, bg


def evaluate(p, Xsup, ysup, Xq, yq, rng):
    la = ci = bgv = 0.0
    for _ in range(3):
        arrs = build_eval(Xsup, ysup, Xq, yq, rng)
        a, c, g = eval_metrics(p, *arrs)
        la += float(a); ci += float(c); bgv += float(g)
    return la / 3, ci / 3, bgv / 3


# ── train ─────────────────────────────────────────────────────────────────────
def train(p, Xtr, ytr, Xte, yte):
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, 200, NUM_STEPS)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    st = opt.init(p)
    rng = np.random.default_rng(SEED)
    hist = {k: [] for k in ("step", "loss", "label_te", "ink_te", "bg", "label_tr", "ink_tr")}
    t0 = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        arrs = build_train(Xtr, ytr, rng)
        p, st, loss = train_step(opt, p, st, *arrs)
        if step % EVAL_EVERY == 0 or step == 1:
            la, ci, bg = evaluate(p, Xtr, ytr, Xte, yte, np.random.default_rng(1))
            lat, cit, _ = evaluate(p, Xtr, ytr, Xtr, ytr, np.random.default_rng(2))
            for k, v in zip(("step", "loss", "label_te", "ink_te", "bg", "label_tr", "ink_tr"),
                            (step, float(loss), la, ci, bg, lat, cit)):
                hist[k].append(v)
            logging.info(f"step {step:5d}  loss {float(loss):.3f}  "
                         f"label tr/te {lat:.3f}/{la:.3f}  content_ink tr/te {cit:.3f}/{ci:.3f}  "
                         f"(bg {bg:.3f})  ({time.perf_counter()-t0:.0f}s)")
    return p, hist, time.perf_counter() - t0


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping"); raise SystemExit(0)
    config, p = init()
    logging.info(f"Loading {DATASET}…")
    data = load_supervised_image(DATASET)
    Xtr = np.asarray(data.X.reshape(data.n_samples, -1), np.float32)
    Xte = np.asarray(data.X_test.reshape(data.n_test_samples, -1), np.float32)
    ytr = np.asarray(data.y); yte = np.asarray(data.y_test)
    logging.info(f"train {Xtr.shape} test {Xte.shape}  n_params {n_params(p)}  "
                 f"tokens/episode {S_SAMPLES*(N_CTX+N_QP+1)}  L={N_LAYERS} d={D} heads={N_HEADS}")
    p, hist, elapsed = train(p, Xtr, ytr, Xte, yte)
    final = {k: hist[k][-1] for k in ("label_te", "ink_te", "bg", "label_tr", "ink_tr")}
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {"experiment": EXP_NAME, "name": "token-level transformer v1: masked-token completion (label+pixels)",
           "time_s": elapsed, "n_params": n_params(p), **final, **config, "history": hist}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
