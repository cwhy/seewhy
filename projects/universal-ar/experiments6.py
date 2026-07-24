"""
Universal AR — exp6 (token-level): push the MNIST accuracy ceiling.

exp1/4/5 saw ~0.55 label with a small model and partial pixels. Here we sweep
toward the ceiling: FULL image (N_CTX up to 784) and bigger models. If label
climbs to acceptable MNIST accuracy, the earlier numbers were an information +
capacity budget, not a modeling failure.

Grid (each a fresh model, additive token embedding, shared position pool):
    (D=256, L=4, N_CTX=784)   current model, full image
    (D=512, L=6, N_CTX=384)   bigger model, half image
    (D=512, L=6, N_CTX=784)   bigger model, full image
Model dims are read from param shapes (head_dim=32 → n_heads=D/32); everything
else = exp4/5. S=4 samples/episode keeps O(N²) attention tractable at N_CTX=784.

Usage: uv run python projects/universal-ar/scripts/run_experiments.py --bg exp6
"""

import json, time, logging
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp6"
JSONL = Path(__file__).parent / "results.jsonl"

DATASET = "mnist"
K = 32; N_CLASSES = 10; POS_PIX = 784; POS_LABEL = 784; N_POS = 785
N_CONTENT = K + N_CLASSES; MASK_ID = N_CONTENT; N_VAL = N_CONTENT + 1; V_REFS = 64
HEAD_DIM = 32
S_SAMPLES, N_QP = 4, 16
M_SUP, Q_QRY = 4, 8
LABEL_QUERY_FRAC = 0.5
LR, SEED = 3e-4, 0

# (D, n_layers, n_ctx, batch, steps)
GRID = [(256, 4, 768, 8, 5000),
        (512, 6, 384, 6, 5000),
        (512, 6, 768, 4, 6000)]


def init(key, D, L):
    g = jax.random.split(key, 3 + L * 6 + 3); i = iter(g)
    lin = lambda k, s: jax.random.normal(k, s) * (1.0 / s[0] ** 0.5)
    p = {"pos_emb": jax.random.normal(next(i), (N_POS, D)) * 0.02,
         "val_emb": jax.random.normal(next(i), (N_VAL, D)) * 0.02,
         "ref_emb": jax.random.normal(next(i), (V_REFS, D)) * 0.02, "layers": []}
    for _ in range(L):
        p["layers"].append(dict(ln1_g=jnp.ones(D), ln1_b=jnp.zeros(D), Wqkv=lin(next(i), (D, 3 * D)), Wo=lin(next(i), (D, D)),
                                ln2_g=jnp.ones(D), ln2_b=jnp.zeros(D), W1=lin(next(i), (D, 4 * D)),
                                b1=jnp.zeros(4 * D), W2=lin(next(i), (4 * D, D)), b2=jnp.zeros(D)))
    p["lnf_g"] = jnp.ones(D); p["lnf_b"] = jnp.zeros(D)
    p["head_W"] = lin(next(i), (D, N_CONTENT)); p["head_b"] = jnp.zeros(N_CONTENT)
    return p


def n_params(p): return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def mha(x, Lp):
    B, N, D = x.shape; H = D // HEAD_DIM
    q, k, v = jnp.split(x @ Lp["Wqkv"], 3, -1)
    sh = lambda t: t.reshape(B, N, H, HEAD_DIM).transpose(0, 2, 1, 3)
    q, k, v = sh(q), sh(k), sh(v)
    a = jax.nn.softmax(jnp.einsum("bhid,bhjd->bhij", q, k) / HEAD_DIM ** 0.5, -1)
    return jnp.einsum("bhij,bhjd->bhid", a, v).transpose(0, 2, 1, 3).reshape(B, N, D) @ Lp["Wo"]


def onehot_mm(ids, table, n):
    return jnp.einsum("bnk,kd->bnd", jax.nn.one_hot(ids, n, dtype=jnp.float32), table)


def forward(p, pos, val, ref):
    x = onehot_mm(pos, p["pos_emb"], N_POS) + onehot_mm(val, p["val_emb"], N_VAL) + onehot_mm(ref, p["ref_emb"], V_REFS)
    for Lp in p["layers"]:
        x = x + mha(ln(x, Lp["ln1_g"], Lp["ln1_b"]), Lp)
        x = x + (jax.nn.gelu(ln(x, Lp["ln2_g"], Lp["ln2_b"]) @ Lp["W1"] + Lp["b1"]) @ Lp["W2"] + Lp["b2"])
    return ln(x, p["lnf_g"], p["lnf_b"]) @ p["head_W"] + p["head_b"]


def loss_fn(p, pos, val, ref, target, isq):
    ce = optax.softmax_cross_entropy_with_integer_labels(forward(p, pos, val, ref), jnp.clip(target, 0, N_CONTENT - 1))
    return (ce * isq).sum() / (isq.sum() + 1e-6), ()


@partial(jax.jit, static_argnums=(0,))
def train_step(opt, p, st, pos, val, ref, target, isq):
    (loss, _), g = jax.value_and_grad(loss_fn, has_aux=True)(p, pos, val, ref, target, isq)
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, loss


def _bin(px): return int(np.floor(px / 255.0 * (K - 1)))


def build_train(Xtr, ytr, rng, n_ctx, B):
    S, n_pool = S_SAMPLES, min(2 * n_ctx, POS_PIX); T = S * (n_ctx + N_QP + 1)
    pos = np.zeros((B, T), np.int32); val = np.zeros((B, T), np.int32); ref = np.zeros((B, T), np.int32)
    tgt = -np.ones((B, T), np.int32); isq = np.zeros((B, T), np.float32)
    for b in range(B):
        pool = rng.permutation(POS_PIX)[:n_pool]
        idx = rng.integers(0, Xtr.shape[0], S); refs = rng.permutation(V_REFS)[:S]; lab_q = rng.random(S) < LABEL_QUERY_FRAC; t = 0
        for s in range(S):
            img = Xtr[idx[s]]; cls = int(ytr[idx[s]]); r = int(refs[s]); sel = rng.permutation(n_pool)
            for pp in pool[sel[:n_ctx]]:
                pos[b, t], val[b, t], ref[b, t] = pp, _bin(img[pp]), r; t += 1
            for pp in pool[sel[n_ctx:n_ctx + N_QP]]:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, _bin(img[pp]), 1.0; t += 1
            if lab_q[s]:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = POS_LABEL, MASK_ID, r, K + cls, 1.0
            else:
                pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + cls, r
            t += 1
    return tuple(jnp.asarray(a) for a in (pos, val, ref, tgt, isq))


def build_eval(Xsup, ysup, Xq, yq, rng, n_ctx, B):
    n_pool = min(2 * n_ctx, POS_PIX); T = M_SUP * (n_ctx + 1) + Q_QRY * (n_ctx + N_QP + 1)
    pos = np.zeros((B, T), np.int32); val = np.zeros((B, T), np.int32); ref = np.zeros((B, T), np.int32)
    tgt = -np.ones((B, T), np.int32); isq = np.zeros((B, T), np.float32); is_lab = np.zeros((B, T), np.float32)
    for b in range(B):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:M_SUP + Q_QRY]; t = 0
        si = rng.integers(0, Xsup.shape[0], M_SUP)
        for s in range(M_SUP):
            img = Xsup[si[s]]; cls = int(ysup[si[s]]); r = int(refs[s])
            for pp in pool[rng.permutation(n_pool)[:n_ctx]]:
                pos[b, t], val[b, t], ref[b, t] = pp, _bin(img[pp]), r; t += 1
            pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + cls, r; t += 1
        qi = rng.integers(0, Xq.shape[0], Q_QRY)
        for s in range(Q_QRY):
            img = Xq[qi[s]]; cls = int(yq[qi[s]]); r = int(refs[M_SUP + s]); sel = rng.permutation(n_pool)
            for pp in pool[sel[:n_ctx]]:
                pos[b, t], val[b, t], ref[b, t] = pp, _bin(img[pp]), r; t += 1
            for pp in pool[sel[n_ctx:n_ctx + N_QP]]:
                pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, _bin(img[pp]), 1.0; t += 1
            pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t], is_lab[b, t] = POS_LABEL, MASK_ID, r, K + cls, 1.0, 1.0; t += 1
    return tuple(jnp.asarray(a) for a in (pos, val, ref, tgt, isq, is_lab))


@jax.jit
def eval_metrics(p, pos, val, ref, tgt, isq, is_lab):
    pred = jnp.argmax(forward(p, pos, val, ref), -1); correct = (pred == tgt); is_pix = isq * (1 - is_lab)
    lab = (correct * is_lab).sum() / (is_lab.sum() + 1e-6)
    ink = is_pix * (tgt > 0)
    return lab, (correct * ink).sum() / (ink.sum() + 1e-6), (is_pix * (tgt == 0)).sum() / (is_pix.sum() + 1e-6)


def evaluate(p, Xsup, ysup, Xq, yq, seed, n_ctx, B):
    rng = np.random.default_rng(seed); la = ci = bg = 0.0
    for _ in range(3):
        a, i, g = eval_metrics(p, *build_eval(Xsup, ysup, Xq, yq, rng, n_ctx, B))
        la += float(a); ci += float(i); bg += float(g)
    return la / 3, ci / 3, bg / 3


def run_one(D, L, n_ctx, B, steps, Xtr, ytr, Xte, yte):
    p = init(jax.random.PRNGKey(SEED), D, L)
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, 200, steps)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4)); st = opt.init(p)
    rng = np.random.default_rng(SEED)
    hist = {k: [] for k in ("step", "loss", "label_te", "ink_te", "bg")}
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        p, st, loss = train_step(opt, p, st, *build_train(Xtr, ytr, rng, n_ctx, B))
        if step % 500 == 0 or step == 1:
            la, ci, bg = evaluate(p, Xtr, ytr, Xte, yte, 1, n_ctx, B)
            for k, v in zip(("step", "loss", "label_te", "ink_te", "bg"), (step, float(loss), la, ci, bg)):
                hist[k].append(v)
            logging.info(f"  D{D} L{L} N_CTX{n_ctx} step {step:5d}  loss {float(loss):.3f}  "
                         f"label_te {la:.3f}  ink_te {ci:.3f}  ({time.perf_counter()-t0:.0f}s)")
    return hist, n_params(p)


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping"); raise SystemExit(0)
    logging.info(f"Loading {DATASET}…")
    data = load_supervised_image(DATASET)
    Xtr = np.asarray(data.X.reshape(data.n_samples, -1), np.float32)
    Xte = np.asarray(data.X_test.reshape(data.n_test_samples, -1), np.float32)
    ytr = np.asarray(data.y); yte = np.asarray(data.y_test)
    logging.info(f"train {Xtr.shape} test {Xte.shape}  grid {GRID}")

    t0 = time.perf_counter(); results = {}
    for (D, L, n_ctx, B, steps) in GRID:
        tag = f"D{D}_L{L}_ctx{n_ctx}"
        logging.info(f"=== {tag}  ({100*n_ctx/POS_PIX:.0f}% of image, {S_SAMPLES*(n_ctx+N_QP+1)} tok/ep, B={B}, {steps} steps) ===")
        h, npar = run_one(D, L, n_ctx, B, steps, Xtr, ytr, Xte, yte)
        results[tag] = {"label_te": h["label_te"][-1], "ink_te": h["ink_te"][-1], "bg": h["bg"][-1],
                        "n_params": npar, "D": D, "L": L, "n_ctx": n_ctx, "history": h}
        logging.info(f"=== {tag} final: label_te {results[tag]['label_te']:.3f}  ink_te {results[tag]['ink_te']:.3f}  ({npar/1e6:.1f}M params) ===")
    elapsed = time.perf_counter() - t0

    final = {tag: {"label_te": results[tag]["label_te"], "ink_te": results[tag]["ink_te"]} for tag in results}
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {"experiment": EXP_NAME, "name": "token-level additive: full-image + capacity sweep toward MNIST ceiling",
           "time_s": elapsed, "grid": GRID, **{f"{tag}_label": results[tag]["label_te"] for tag in results},
           **{f"{tag}_ink": results[tag]["ink_te"] for tag in results}, "results": results}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
