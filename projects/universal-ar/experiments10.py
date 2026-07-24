"""
Universal AR — exp10 (longer run of exp9) (token-level): half-observed (OBS_FRAC=0.5), grad accumulation.

Cleanups requested:
  * OBS_FRAC = 0.5 as the knob (→ N_CTX = 392 = half of the 784-pixel image);
    shared pool = all 784 positions, so every sample observes a random half.
  * single config (no N_CTX sweep) — training uniformly at half.
  * gradient accumulation (MICRO_BATCH × ACCUM = effective batch) so the ~6.5k-token
    sequence trains at a proper effective batch despite the O(N²) memory limit —
    fixes the B=4 undertraining seen at large N_CTX.

TRAIN (= exp7 strategy): random support + answerable-only label loss.
EVAL: class-balanced 1-shot ×10 support (clean 10-way, chance 0.10).
Additive embedding, gradient checkpointing.

Usage: uv run python projects/universal-ar/scripts/run_experiments.py --bg exp9
"""

import json, time, logging
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp10"
JSONL = Path(__file__).parent / "results.jsonl"

DATASET = "mnist"
K = 32; N_CLASSES = 10; POS_PIX = 784; POS_LABEL = 784; N_POS = 785
N_CONTENT = K + N_CLASSES; MASK_ID = N_CONTENT; N_VAL = N_CONTENT + 1; V_REFS = 64
HEAD_DIM = 32
D, N_LAYERS = 256, 4
OBS_FRAC = 0.5                          # fraction of the image each sample observes
N_CTX = int(round(OBS_FRAC * POS_PIX))  # 392
N_QP = 16
N_SUP, N_QRY = 10, 6                    # training: random support + query
Q_EVAL = 8
MICRO_BATCH, ACCUM = 4, 2              # effective batch = 8, via gradient accumulation
LR, SEED = 3e-4, 0
NUM_STEPS, EVAL_EVERY = 8000, 1000


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
def train_step(opt, p, st, pos, val, ref, tgt, isq):
    """Gradient accumulation: leading axis of each arg is ACCUM micro-batches."""
    def micro(_, xs):
        (loss, _), g = jax.value_and_grad(loss_fn, has_aux=True)(p, *xs)
        return None, (loss, g)
    _, (losses, grads) = jax.lax.scan(micro, None, (pos, val, ref, tgt, isq))
    g = jax.tree_util.tree_map(lambda x: x.mean(0), grads)     # average grads over micro-batches
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, losses.mean()


def _bin(px): return int(np.floor(px / 255.0 * (K - 1)))


def _alloc(S):
    T = S * (N_CTX + N_QP + 1)
    return [np.zeros((MICRO_BATCH, T), np.int32), np.zeros((MICRO_BATCH, T), np.int32), np.zeros((MICRO_BATCH, T), np.int32),
            -np.ones((MICRO_BATCH, T), np.int32), np.zeros((MICRO_BATCH, T), np.float32), np.zeros((MICRO_BATCH, T), np.float32)]


def _emit(arrs, b, t, img, cls, r, given, ok, pool, sel):
    pos, val, ref, tgt, isq, is_lab = arrs
    for pp in pool[sel[:N_CTX]]:
        pos[b, t], val[b, t], ref[b, t] = pp, _bin(img[pp]), r; t += 1
    for pp in pool[sel[N_CTX:N_CTX + N_QP]]:
        pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, _bin(img[pp]), 1.0; t += 1
    if given:
        pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + cls, r
    else:
        pos[b, t], val[b, t], ref[b, t] = POS_LABEL, MASK_ID, r
        if ok:
            tgt[b, t], isq[b, t], is_lab[b, t] = K + cls, 1.0, 1.0
    return t + 1


def build_train(Xs, ys, rng):
    S, n_pool = N_SUP + N_QRY, min(2 * N_CTX, POS_PIX); arrs = _alloc(S)
    for b in range(MICRO_BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        si = rng.integers(0, Xs.shape[0], N_SUP); qi = rng.integers(0, Xs.shape[0], N_QRY)
        sup_classes = set(int(ys[j]) for j in si)
        for k, j in enumerate(si):
            t = _emit(arrs, b, t, Xs[j], int(ys[j]), int(refs[k]), True, True, pool, rng.permutation(n_pool))
        for k, j in enumerate(qi):
            cls = int(ys[j])
            t = _emit(arrs, b, t, Xs[j], cls, int(refs[N_SUP + k]), False, cls in sup_classes, pool, rng.permutation(n_pool))
    return tuple(np.asarray(a) for a in arrs)


def build_eval_bal(Xtr, ytr, cls_idx, Xq, yq, rng):
    S, n_pool = N_CLASSES + Q_EVAL, min(2 * N_CTX, POS_PIX); arrs = _alloc(S)
    for b in range(MICRO_BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        reps = [(Xtr[cls_idx[c][rng.integers(len(cls_idx[c]))]], c, True) for c in range(N_CLASSES)]
        qi = rng.integers(0, Xq.shape[0], Q_EVAL)
        qry = [(Xq[j], int(yq[j]), False) for j in qi]
        for k, (img, cls, given) in enumerate(reps + qry):
            t = _emit(arrs, b, t, img, cls, int(refs[k]), given, True, pool, rng.permutation(n_pool))
    return tuple(jnp.asarray(a) for a in arrs)


@jax.jit
def eval_metrics(p, pos, val, ref, tgt, isq, is_lab):
    pred = jnp.argmax(forward(p, pos, val, ref), -1); correct = (pred == tgt); is_pix = isq * (1 - is_lab)
    lab = (correct * is_lab).sum() / (is_lab.sum() + 1e-6)
    ink = is_pix * (tgt > 0)
    return lab, (correct * ink).sum() / (ink.sum() + 1e-6), (is_pix * (tgt == 0)).sum() / (is_pix.sum() + 1e-6)


def evaluate(p, Xtr, ytr, cls_idx, Xte, yte, seed):
    rng = np.random.default_rng(seed); la = ci = bg = 0.0
    for _ in range(4):
        lab, ink, g = eval_metrics(p, *build_eval_bal(Xtr, ytr, cls_idx, Xte, yte, rng))
        la += float(lab); ci += float(ink); bg += float(g)
    return la / 4, ci / 4, bg / 4


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
    ctx_len = (N_SUP + N_QRY) * (N_CTX + N_QP + 1)
    logging.info(f"train {Xtr.shape}  OBS_FRAC={OBS_FRAC} → N_CTX={N_CTX}  eff_batch={MICRO_BATCH*ACCUM} "
                 f"(micro {MICRO_BATCH}×{ACCUM})  context_len(train)={ctx_len} tokens")

    p = init(jax.random.PRNGKey(SEED), D, N_LAYERS)
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, 200, NUM_STEPS)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4)); st = opt.init(p)
    rng = np.random.default_rng(SEED)
    hist = {k: [] for k in ("step", "loss", "label_te", "ink_te", "bg")}
    t0 = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        mics = [build_train(Xtr, ytr, rng) for _ in range(ACCUM)]
        stacked = tuple(jnp.asarray(np.stack([m[i] for m in mics])) for i in range(5))  # (ACCUM,B,T)
        p, st, loss = train_step(opt, p, st, *stacked)
        if step % EVAL_EVERY == 0 or step == 1:
            la, ci, bg = evaluate(p, Xtr, ytr, cls_idx, Xte, yte, 1)
            for k, v in zip(("step", "loss", "label_te", "ink_te", "bg"), (step, float(loss), la, ci, bg)):
                hist[k].append(v)
            logging.info(f"step {step:5d}  loss {float(loss):.3f}  label_te(balanced) {la:.3f}  "
                         f"ink_te {ci:.3f}  (bg {bg:.3f})  ({time.perf_counter()-t0:.0f}s)")
    elapsed = time.perf_counter() - t0
    final = {"label_te": hist["label_te"][-1], "ink_te": hist["ink_te"][-1], "bg": hist["bg"][-1]}
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {"experiment": EXP_NAME, "name": "token-level: half-observed (OBS_FRAC=0.5) + grad-accum eff_batch=8, balanced eval",
           "time_s": elapsed, "n_params": n_params(p), "obs_frac": OBS_FRAC, "n_ctx": N_CTX,
           "eff_batch": MICRO_BATCH * ACCUM, "context_len": ctx_len, **final, "history": hist}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
