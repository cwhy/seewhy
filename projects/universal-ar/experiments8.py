"""
Universal AR — exp8 (token-level): train w/ answerable-only masking, eval on a
class-balanced support.

TRAIN (= exp7): random N_SUP given-label support + N_QRY label-query samples; a
label query is scored/trained only if its class appears in the support
(answerable-only loss). No forced balance in training.

EVAL: class-balanced support — one labeled TRAIN sample of EACH class (all 10
present) + Q_EVAL query TEST samples. Now every query is answerable, so the label
metric is a clean 10-way in-context classification accuracy (chance 0.10), not
confounded by coverage. Sweeps N_CTX. Additive embedding, gradient checkpointing.

Usage: uv run python projects/universal-ar/scripts/run_experiments.py --bg exp8
"""

import json, time, logging
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp8"
JSONL = Path(__file__).parent / "results.jsonl"

DATASET = "mnist"
K = 32; N_CLASSES = 10; POS_PIX = 784; POS_LABEL = 784; N_POS = 785
N_CONTENT = K + N_CLASSES; MASK_ID = N_CONTENT; N_VAL = N_CONTENT + 1; V_REFS = 64
HEAD_DIM = 32
D, N_LAYERS = 256, 4
N_SUP, N_QRY, N_QP = 10, 6, 16          # training: random support + query
Q_EVAL = 8                              # eval: balanced 10-class support + Q_EVAL queries
BATCH = 4
NCTX_SWEEP = [96, 192, 384]
LR, SEED = 3e-4, 0
NUM_STEPS, EVAL_EVERY = 4000, 500


def init(key, Dm, L):
    g = jax.random.split(key, 3 + L * 6 + 3); i = iter(g)
    lin = lambda k, s: jax.random.normal(k, s) * (1.0 / s[0] ** 0.5)
    p = {"pos_emb": jax.random.normal(next(i), (N_POS, Dm)) * 0.02,
         "val_emb": jax.random.normal(next(i), (N_VAL, Dm)) * 0.02,
         "ref_emb": jax.random.normal(next(i), (V_REFS, Dm)) * 0.02, "layers": []}
    for _ in range(L):
        p["layers"].append(dict(ln1_g=jnp.ones(Dm), ln1_b=jnp.zeros(Dm), Wqkv=lin(next(i), (Dm, 3 * Dm)), Wo=lin(next(i), (Dm, Dm)),
                                ln2_g=jnp.ones(Dm), ln2_b=jnp.zeros(Dm), W1=lin(next(i), (Dm, 4 * Dm)),
                                b1=jnp.zeros(4 * Dm), W2=lin(next(i), (4 * Dm, Dm)), b2=jnp.zeros(Dm)))
    p["lnf_g"] = jnp.ones(Dm); p["lnf_b"] = jnp.zeros(Dm)
    p["head_W"] = lin(next(i), (Dm, N_CONTENT)); p["head_b"] = jnp.zeros(N_CONTENT)
    return p


def n_params(p): return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def mha(x, Lp):
    B, N, Dm = x.shape; H = Dm // HEAD_DIM
    q, k, v = jnp.split(x @ Lp["Wqkv"], 3, -1)
    sh = lambda t: t.reshape(B, N, H, HEAD_DIM).transpose(0, 2, 1, 3)
    q, k, v = sh(q), sh(k), sh(v)
    a = jax.nn.softmax(jnp.einsum("bhid,bhjd->bhij", q, k) / HEAD_DIM ** 0.5, -1)
    return jnp.einsum("bhij,bhjd->bhid", a, v).transpose(0, 2, 1, 3).reshape(B, N, Dm) @ Lp["Wo"]


def onehot_mm(ids, table, n):
    return jnp.einsum("bnk,kd->bnd", jax.nn.one_hot(ids, n, dtype=jnp.float32), table)


@jax.checkpoint
def _layer(Lp, x):
    x = x + mha(ln(x, Lp["ln1_g"], Lp["ln1_b"]), Lp)
    return x + (jax.nn.gelu(ln(x, Lp["ln2_g"], Lp["ln2_b"]) @ Lp["W1"] + Lp["b1"]) @ Lp["W2"] + Lp["b2"])


def forward(p, pos, val, ref):
    x = onehot_mm(pos, p["pos_emb"], N_POS) + onehot_mm(val, p["val_emb"], N_VAL) + onehot_mm(ref, p["ref_emb"], V_REFS)
    for Lp in p["layers"]:
        x = _layer(Lp, x)
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


def _emit(arrs, b, t, img, cls, r, given, pool, sel, n_ctx):
    pos, val, ref, tgt, isq, is_lab = arrs
    for pp in pool[sel[:n_ctx]]:
        pos[b, t], val[b, t], ref[b, t] = pp, _bin(img[pp]), r; t += 1
    for pp in pool[sel[n_ctx:n_ctx + N_QP]]:
        pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, _bin(img[pp]), 1.0; t += 1
    if given:
        pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + cls, r
    else:
        pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t], is_lab[b, t] = POS_LABEL, MASK_ID, r, K + cls, 1.0, 1.0
    return t + 1


def _alloc(S, n_ctx):
    T = S * (n_ctx + N_QP + 1)
    return [np.zeros((BATCH, T), np.int32), np.zeros((BATCH, T), np.int32), np.zeros((BATCH, T), np.int32),
            -np.ones((BATCH, T), np.int32), np.zeros((BATCH, T), np.float32), np.zeros((BATCH, T), np.float32)]


def build_train(Xs, ys, rng, n_ctx):
    """Random support + answerable-only label queries (exp7 training strategy)."""
    S, n_pool = N_SUP + N_QRY, min(2 * n_ctx, POS_PIX); arrs = _alloc(S, n_ctx)
    for b in range(BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        si = rng.integers(0, Xs.shape[0], N_SUP); qi = rng.integers(0, Xs.shape[0], N_QRY)
        sup_classes = set(int(ys[j]) for j in si)
        for j in si:
            t = _emit(arrs, b, t, Xs[j], int(ys[j]), int(refs[t // (n_ctx + N_QP + 1)]), True, pool, rng.permutation(n_pool), n_ctx)
        for j in qi:
            cls = int(ys[j]); ok = cls in sup_classes; r = int(refs[t // (n_ctx + N_QP + 1)])
            sel = rng.permutation(n_pool)
            for pp in pool[sel[:n_ctx]]:
                arrs[0][b, t], arrs[1][b, t], arrs[2][b, t] = pp, _bin(Xs[j][pp]), r; t += 1
            for pp in pool[sel[n_ctx:n_ctx + N_QP]]:
                arrs[0][b, t], arrs[1][b, t], arrs[2][b, t], arrs[3][b, t], arrs[4][b, t] = pp, MASK_ID, r, _bin(Xs[j][pp]), 1.0; t += 1
            arrs[0][b, t], arrs[1][b, t], arrs[2][b, t] = POS_LABEL, MASK_ID, r
            if ok:
                arrs[3][b, t], arrs[4][b, t], arrs[5][b, t] = K + cls, 1.0, 1.0
            t += 1
    return tuple(jnp.asarray(a) for a in arrs)


def build_eval_bal(Xtr, ytr, cls_idx, Xq, yq, rng, n_ctx):
    """Class-balanced support: one labeled train sample per class + Q_EVAL test queries (all answerable)."""
    S, n_pool = N_CLASSES + Q_EVAL, min(2 * n_ctx, POS_PIX); arrs = _alloc(S, n_ctx)
    for b in range(BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        reps = [(Xtr[cls_idx[c][rng.integers(len(cls_idx[c]))]], c, True) for c in range(N_CLASSES)]
        qi = rng.integers(0, Xq.shape[0], Q_EVAL)
        qry = [(Xq[j], int(yq[j]), False) for j in qi]
        for k, (img, cls, given) in enumerate(reps + qry):
            t = _emit(arrs, b, t, img, cls, int(refs[k]), given, pool, rng.permutation(n_pool), n_ctx)
    return tuple(jnp.asarray(a) for a in arrs)


@jax.jit
def eval_metrics(p, pos, val, ref, tgt, isq, is_lab):
    pred = jnp.argmax(forward(p, pos, val, ref), -1); correct = (pred == tgt); is_pix = isq * (1 - is_lab)
    lab = (correct * is_lab).sum() / (is_lab.sum() + 1e-6)
    ink = is_pix * (tgt > 0)
    return lab, (correct * ink).sum() / (ink.sum() + 1e-6), (is_pix * (tgt == 0)).sum() / (is_pix.sum() + 1e-6)


def evaluate(p, Xtr, ytr, cls_idx, Xte, yte, seed, n_ctx):
    rng = np.random.default_rng(seed); la = ci = bg = 0.0
    for _ in range(4):
        lab, ink, g = eval_metrics(p, *build_eval_bal(Xtr, ytr, cls_idx, Xte, yte, rng, n_ctx))
        la += float(lab); ci += float(ink); bg += float(g)
    return la / 4, ci / 4, bg / 4


def run_one(n_ctx, Xtr, ytr, cls_idx, Xte, yte):
    p = init(jax.random.PRNGKey(SEED), D, N_LAYERS)
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, 200, NUM_STEPS)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4)); st = opt.init(p)
    rng = np.random.default_rng(SEED)
    hist = {k: [] for k in ("step", "loss", "label_te", "ink_te", "bg")}
    t0 = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        p, st, loss = train_step(opt, p, st, *build_train(Xtr, ytr, rng, n_ctx)[:5])
        if step % EVAL_EVERY == 0 or step == 1:
            la, ci, bg = evaluate(p, Xtr, ytr, cls_idx, Xte, yte, 1, n_ctx)
            for k, v in zip(("step", "loss", "label_te", "ink_te", "bg"), (step, float(loss), la, ci, bg)):
                hist[k].append(v)
            logging.info(f"  N_CTX={n_ctx:3d} step {step:5d}  loss {float(loss):.3f}  "
                         f"label_te(balanced) {la:.3f}  ink_te {ci:.3f}  ({time.perf_counter()-t0:.0f}s)")
    return hist


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
    cls_idx = [np.where(ytr == c)[0] for c in range(N_CLASSES)]
    logging.info(f"train {Xtr.shape} test {Xte.shape}  train=random+answerable, eval=class-balanced  sweep N_CTX={NCTX_SWEEP}")

    t0 = time.perf_counter(); sweep = {}
    for n_ctx in NCTX_SWEEP:
        logging.info(f"=== N_CTX={n_ctx} ===")
        h = run_one(n_ctx, Xtr, ytr, cls_idx, Xte, yte)
        sweep[n_ctx] = {"label_te": h["label_te"][-1], "ink_te": h["ink_te"][-1], "bg": h["bg"][-1], "history": h}
        logging.info(f"=== N_CTX={n_ctx} final: label_te(balanced) {sweep[n_ctx]['label_te']:.3f}  ink_te {sweep[n_ctx]['ink_te']:.3f} ===")
    elapsed = time.perf_counter() - t0

    final = {f"label_{n}": sweep[n]["label_te"] for n in NCTX_SWEEP}
    final.update({f"ink_{n}": sweep[n]["ink_te"] for n in NCTX_SWEEP})
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {"experiment": EXP_NAME, "name": "token-level: train random+answerable, eval class-balanced (clean 10-way) + N_CTX sweep",
           "time_s": elapsed, "nctx_sweep": NCTX_SWEEP, "n_sup": N_SUP, "n_qry": N_QRY, "q_eval": Q_EVAL, "batch": BATCH,
           **final, "bg": sweep[NCTX_SWEEP[0]]["bg"], "sweep": {str(n): sweep[n] for n in NCTX_SWEEP}}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
