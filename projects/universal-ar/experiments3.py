"""
Universal AR — exp3 (token-level): context-size sweep.

Additive token embedding (exp1's winner). Train once on moderate episodes, then
evaluate label + content(ink) as a function of the SUPPORT-set size M (number of
in-context labeled train samples), with query drawn from TEST. Directly measures
"context is the lever".

Sweep M ∈ {2,4,8,16,32}; support from TRAIN, query (8) from TEST.

Usage: uv run python projects/universal-ar/scripts/run_experiments.py --bg exp3
"""

import json, time, logging
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp3"
JSONL = Path(__file__).parent / "results.jsonl"

DATASET = "mnist"
K = 32; N_CLASSES = 10; POS_PIX = 784; POS_LABEL = 784; N_POS = 785
N_CONTENT = K + N_CLASSES; MASK_ID = N_CONTENT; N_VAL = N_CONTENT + 1; V_REFS = 96
D, N_LAYERS, N_HEADS, MLP_MULT = 256, 4, 8, 4
S_SAMPLES, N_CTX, N_QP = 32, 48, 16
LABEL_QUERY_FRAC = 0.5
BATCH = 12
MSUP_SWEEP = [2, 4, 8, 16, 32]; Q_QRY = 8
NUM_STEPS, EVAL_EVERY, LR, SEED = 4000, 500, 3e-4, 0


def init():
    config = dict(dataset=DATASET, K=K, D=D, n_layers=N_LAYERS, n_heads=N_HEADS, s_samples=S_SAMPLES,
                  n_ctx=N_CTX, n_qp=N_QP, v_refs=V_REFS, batch=BATCH, msup_sweep=MSUP_SWEEP, q_qry=Q_QRY,
                  num_steps=NUM_STEPS, lr=LR, seed=SEED, variant="token_additive_ctxsweep")
    g = jax.random.split(jax.random.PRNGKey(SEED), 3 + N_LAYERS * 6 + 3); i = iter(g)
    lin = lambda k, s: jax.random.normal(k, s) * (1.0 / s[0] ** 0.5)
    p = {"pos_emb": jax.random.normal(next(i), (N_POS, D)) * 0.02,
         "val_emb": jax.random.normal(next(i), (N_VAL, D)) * 0.02,
         "ref_emb": jax.random.normal(next(i), (V_REFS, D)) * 0.02, "layers": []}
    for _ in range(N_LAYERS):
        p["layers"].append(dict(ln1_g=jnp.ones(D), ln1_b=jnp.zeros(D), Wqkv=lin(next(i), (D, 3 * D)), Wo=lin(next(i), (D, D)),
                                ln2_g=jnp.ones(D), ln2_b=jnp.zeros(D), W1=lin(next(i), (D, MLP_MULT * D)),
                                b1=jnp.zeros(MLP_MULT * D), W2=lin(next(i), (MLP_MULT * D, D)), b2=jnp.zeros(D)))
    p["lnf_g"] = jnp.ones(D); p["lnf_b"] = jnp.zeros(D)
    p["head_W"] = lin(next(i), (D, N_CONTENT)); p["head_b"] = jnp.zeros(N_CONTENT)
    return config, p


def n_params(p): return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def mha(x, L):
    B, N, _ = x.shape
    q, k, v = jnp.split(x @ L["Wqkv"], 3, -1)
    sh = lambda t: t.reshape(B, N, N_HEADS, D // N_HEADS).transpose(0, 2, 1, 3)
    q, k, v = sh(q), sh(k), sh(v)
    a = jax.nn.softmax(jnp.einsum("bhid,bhjd->bhij", q, k) / (D // N_HEADS) ** 0.5, -1)
    return jnp.einsum("bhij,bhjd->bhid", a, v).transpose(0, 2, 1, 3).reshape(B, N, D) @ L["Wo"]


def onehot_mm(ids, table, n):
    return jnp.einsum("bnk,kd->bnd", jax.nn.one_hot(ids, n, dtype=jnp.float32), table)


def forward(p, pos, val, ref):
    x = onehot_mm(pos, p["pos_emb"], N_POS) + onehot_mm(val, p["val_emb"], N_VAL) + onehot_mm(ref, p["ref_emb"], V_REFS)
    for L in p["layers"]:
        x = x + mha(ln(x, L["ln1_g"], L["ln1_b"]), L)
        x = x + (jax.nn.gelu(ln(x, L["ln2_g"], L["ln2_b"]) @ L["W1"] + L["b1"]) @ L["W2"] + L["b2"])
    return ln(x, p["lnf_g"], p["lnf_b"]) @ p["head_W"] + p["head_b"]


def loss_fn(p, pos, val, ref, target, isq):
    ce = optax.softmax_cross_entropy_with_integer_labels(forward(p, pos, val, ref), jnp.clip(target, 0, N_CONTENT - 1))
    return (ce * isq).sum() / (isq.sum() + 1e-6), ()


@partial(jax.jit, static_argnums=(0,))
def train_step(opt, p, st, pos, val, ref, target, isq):
    (loss, _), g = jax.value_and_grad(loss_fn, has_aux=True)(p, pos, val, ref, target, isq)
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, loss


def _bin(px): return np.floor(px / 255.0 * (K - 1)).astype(np.int32)


def build_train(Xtr, ytr, rng):
    B, S = BATCH, S_SAMPLES; T = S * (N_CTX + N_QP + 1)
    pos = np.zeros((B, T), np.int32); val = np.zeros((B, T), np.int32); ref = np.zeros((B, T), np.int32)
    tgt = -np.ones((B, T), np.int32); isq = np.zeros((B, T), np.float32)
    for b in range(B):
        idx = rng.integers(0, Xtr.shape[0], S); refs = rng.permutation(V_REFS)[:S]; lab_q = rng.random(S) < LABEL_QUERY_FRAC; t = 0
        for s in range(S):
            img = Xtr[idx[s]]; cls = int(ytr[idx[s]]); r = int(refs[s]); perm = rng.permutation(POS_PIX)[:N_CTX + N_QP]
            for pp in perm[:N_CTX]:
                pos[b, t], val[b, t], ref[b, t] = pp, int(_bin(img[pp])), r; t += 1
            for pp in perm[N_CTX:]:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, int(_bin(img[pp])), 1.0; t += 1
            if lab_q[s]:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = POS_LABEL, MASK_ID, r, K + cls, 1.0
            else:
                pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + cls, r
            t += 1
    return tuple(jnp.asarray(a) for a in (pos, val, ref, tgt, isq))


def build_eval(Xsup, ysup, Xq, yq, rng, m_sup):
    B = BATCH; T = m_sup * (N_CTX + 1) + Q_QRY * (N_CTX + N_QP + 1)
    pos = np.zeros((B, T), np.int32); val = np.zeros((B, T), np.int32); ref = np.zeros((B, T), np.int32)
    tgt = -np.ones((B, T), np.int32); isq = np.zeros((B, T), np.float32); is_lab = np.zeros((B, T), np.float32)
    for b in range(B):
        refs = rng.permutation(V_REFS)[:m_sup + Q_QRY]; t = 0
        si = rng.integers(0, Xsup.shape[0], m_sup)
        for s in range(m_sup):
            img = Xsup[si[s]]; cls = int(ysup[si[s]]); r = int(refs[s])
            for pp in rng.permutation(POS_PIX)[:N_CTX]:
                pos[b, t], val[b, t], ref[b, t] = pp, int(_bin(img[pp])), r; t += 1
            pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + cls, r; t += 1
        qi = rng.integers(0, Xq.shape[0], Q_QRY)
        for s in range(Q_QRY):
            img = Xq[qi[s]]; cls = int(yq[qi[s]]); r = int(refs[m_sup + s]); perm = rng.permutation(POS_PIX)[:N_CTX + N_QP]
            for pp in perm[:N_CTX]:
                pos[b, t], val[b, t], ref[b, t] = pp, int(_bin(img[pp])), r; t += 1
            for pp in perm[N_CTX:]:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, int(_bin(img[pp])), 1.0; t += 1
            pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t], is_lab[b, t] = POS_LABEL, MASK_ID, r, K + cls, 1.0, 1.0; t += 1
    return tuple(jnp.asarray(a) for a in (pos, val, ref, tgt, isq, is_lab))


@jax.jit
def eval_metrics(p, pos, val, ref, tgt, isq, is_lab):
    pred = jnp.argmax(forward(p, pos, val, ref), -1); correct = (pred == tgt); is_pix = isq * (1 - is_lab)
    lab = (correct * is_lab).sum() / (is_lab.sum() + 1e-6)
    ink = is_pix * (tgt > 0)
    return lab, (correct * ink).sum() / (ink.sum() + 1e-6), (is_pix * (tgt == 0)).sum() / (is_pix.sum() + 1e-6)


def evaluate_sweep(p, Xsup, ysup, Xq, yq, seed):
    out = {}
    for m in MSUP_SWEEP:
        rng = np.random.default_rng(seed); la = ci = bg = 0.0
        for _ in range(2):
            a, c, g = eval_metrics(p, *build_eval(Xsup, ysup, Xq, yq, rng, m))
            la += float(a); ci += float(c); bg += float(g)
        out[f"label_{m}"] = la / 2; out[f"ink_{m}"] = ci / 2; out["bg"] = bg / 2
    return out


def train(p, Xtr, ytr, Xte, yte):
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, 200, NUM_STEPS)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4)); st = opt.init(p)
    rng = np.random.default_rng(SEED)
    hist = {"step": [], "loss": []}
    for m in MSUP_SWEEP: hist[f"label_{m}"] = []; hist[f"ink_{m}"] = []
    hist["bg"] = []
    t0 = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        p, st, loss = train_step(opt, p, st, *build_train(Xtr, ytr, rng))
        if step % EVAL_EVERY == 0 or step == 1:
            m = evaluate_sweep(p, Xtr, ytr, Xte, yte, 1)
            hist["step"].append(step); hist["loss"].append(float(loss))
            for k in m: hist[k].append(m[k])
            logging.info(f"step {step:5d} loss {float(loss):.3f}  label[M=2/8/32] "
                         f"{m['label_2']:.3f}/{m['label_8']:.3f}/{m['label_32']:.3f}  "
                         f"ink[M=32] {m['ink_32']:.3f} (bg {m['bg']:.3f})  ({time.perf_counter()-t0:.0f}s)")
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
    logging.info(f"train {Xtr.shape} test {Xte.shape}  n_params {n_params(p)}  train-tokens {S_SAMPLES*(N_CTX+N_QP+1)}")
    p, hist, elapsed = train(p, Xtr, ytr, Xte, yte)
    final = {k: hist[k][-1] for k in hist if k not in ("step", "loss")}
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {"experiment": EXP_NAME, "name": "token-level additive: context-size sweep (label/ink vs support M)",
           "time_s": elapsed, "n_params": n_params(p), **final, **config, "history": hist}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
