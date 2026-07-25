"""
Universal AR — exp18: 4-vs-9 with a DEEPER model (8 layers).

Companion to exp17. exp15 showed the 4v9 label loss pinned at exactly ln(2) with
train accuracy also at chance — a fitting failure specific to the matching task.
One hypothesis is simply that the match-and-copy computation (locate same-position
tokens across samples, compare values, aggregate, copy the winner's label) needs
more sequential hops than 4 layers provide.

This doubles depth to 8 layers, everything else identical to exp15 (4v9,
anonymised labels, OBS_FRAC=0.5, eff batch 8, full diagnostics).

  learns  → the circuit needs more depth; 4 layers was the limit
  fails   → depth is not the issue (see exp17 for the alignment hypothesis)
"""

import json, time, logging
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp18"
JSONL = Path(__file__).parent / "results.jsonl"

DATASET = "mnist"
K = 32; N_CLASSES = 10; POS_PIX = 784; POS_LABEL = 784; N_POS = 785
N_CONTENT = K + N_CLASSES; MASK_ID = N_CONTENT; N_VAL = N_CONTENT + 1; V_REFS = 64
HEAD_DIM = 32
D, N_LAYERS = 256, 8      # DEPTH test: 8 layers (exp15 used 4)
OBS_FRAC = 0.5                          # fraction of the image each sample observes
N_CTX = int(round(OBS_FRAC * POS_PIX))  # 392
N_QP = 16
N_SUP, N_QRY = 10, 6                    # training: random support + query
Q_EVAL = 8
MICRO_BATCH, ACCUM = 4, 2              # effective batch = 8, via gradient accumulation
ANON_LABELS = True                     # per-episode random class→label-token permutation
TASK_DIGITS = (4, 9)                   # 2-way HARD pair (classic MNIST confusion)
N_TASK = len(TASK_DIGITS)              # → chance = 1/N_TASK = 0.50
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


def loss_fn(p, pos, val, ref, target, isq, is_lab):
    """Same objective as before; aux splits it into the PIXEL and LABEL components.

    Labels are only ~1/17 of the scored tokens (16 masked pixels + 1 label per
    sample), so the combined loss is dominated by pixels and hides whether the
    label task is being learned at all. Report the two separately.
    """
    logits = forward(p, pos, val, ref)                    # single forward pass
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(target, 0, N_CONTENT - 1))
    loss = (ce * isq).sum() / (isq.sum() + 1e-6)          # unchanged training objective
    is_pix = isq * (1 - is_lab)
    pix_loss = (ce * is_pix).sum() / (is_pix.sum() + 1e-6)
    lab_loss = (ce * is_lab).sum() / (is_lab.sum() + 1e-6)
    lab_acc = ((jnp.argmax(logits, -1) == target) * is_lab).sum() / (is_lab.sum() + 1e-6)
    return loss, (pix_loss, lab_loss, lab_acc)


@partial(jax.jit, static_argnums=(0,))
def train_step(opt, p, st, pos, val, ref, tgt, isq, is_lab):
    """Gradient accumulation: leading axis of each arg is ACCUM micro-batches."""
    def micro(_, xs):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, *xs)
        return None, (loss, aux, g)
    _, (losses, auxes, grads) = jax.lax.scan(micro, None, (pos, val, ref, tgt, isq, is_lab))
    g = jax.tree_util.tree_map(lambda x: x.mean(0), grads)     # average grads over micro-batches
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, losses.mean(), tuple(a.mean() for a in auxes)


def _bin(px): return int(np.floor(px / 255.0 * (K - 1)))


def _alloc(S):
    T = S * (N_CTX + N_QP + 1)
    return [np.zeros((MICRO_BATCH, T), np.int32), np.zeros((MICRO_BATCH, T), np.int32), np.zeros((MICRO_BATCH, T), np.int32),
            -np.ones((MICRO_BATCH, T), np.int32), np.zeros((MICRO_BATCH, T), np.float32), np.zeros((MICRO_BATCH, T), np.float32)]


def _emit(arrs, b, t, img, lab_slot, r, given, ok, pool, sel):
    """lab_slot = the (possibly permuted) label-token index for this sample's class."""
    pos, val, ref, tgt, isq, is_lab = arrs
    for pp in pool[sel[:N_CTX]]:
        pos[b, t], val[b, t], ref[b, t] = pp, _bin(img[pp]), r; t += 1
    for pp in pool[sel[N_CTX:N_CTX + N_QP]]:
        pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, _bin(img[pp]), 1.0; t += 1
    if given:
        pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + lab_slot, r
    else:
        pos[b, t], val[b, t], ref[b, t] = POS_LABEL, MASK_ID, r
        if ok:
            tgt[b, t], isq[b, t], is_lab[b, t] = K + lab_slot, 1.0, 1.0
    return t + 1


def _perm(rng):
    """Per-episode class→label-token permutation over the N_TASK label slots."""
    return rng.permutation(N_TASK) if ANON_LABELS else np.arange(N_TASK)


def build_train(Xs, ys, rng):
    S, n_pool = N_SUP + N_QRY, min(2 * N_CTX, POS_PIX); arrs = _alloc(S)
    for b in range(MICRO_BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        perm = _perm(rng)                        # fresh label semantics for THIS episode
        si = rng.integers(0, Xs.shape[0], N_SUP); qi = rng.integers(0, Xs.shape[0], N_QRY)
        sup_classes = set(int(ys[j]) for j in si)
        for k, j in enumerate(si):
            t = _emit(arrs, b, t, Xs[j], int(perm[int(ys[j])]), int(refs[k]), True, True, pool, rng.permutation(n_pool))
        for k, j in enumerate(qi):
            cls = int(ys[j])
            t = _emit(arrs, b, t, Xs[j], int(perm[cls]), int(refs[N_SUP + k]), False,
                      cls in sup_classes, pool, rng.permutation(n_pool))
    return tuple(np.asarray(a) for a in arrs)


def build_eval_bal(Xtr, ytr, cls_idx, Xq, yq, rng):
    S, n_pool = N_TASK + Q_EVAL, min(2 * N_CTX, POS_PIX); arrs = _alloc(S)
    for b in range(MICRO_BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        perm = _perm(rng)                        # fresh label semantics for THIS episode
        reps = [(Xtr[cls_idx[c][rng.integers(len(cls_idx[c]))]], c, True) for c in range(N_TASK)]
        qi = rng.integers(0, Xq.shape[0], Q_EVAL)
        qry = [(Xq[j], int(yq[j]), False) for j in qi]
        for k, (img, cls, given) in enumerate(reps + qry):
            t = _emit(arrs, b, t, img, int(perm[cls]), int(refs[k]), given, True, pool, rng.permutation(n_pool))
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

    # ── restrict to the 2-way task (easy digit pair), remap labels to 0..N_TASK-1 ──
    def _restrict(X, y):
        keep = np.isin(y, TASK_DIGITS)
        remap = {d: i for i, d in enumerate(TASK_DIGITS)}
        return X[keep], np.array([remap[int(v)] for v in y[keep]], np.int32)
    Xtr, ytr = _restrict(Xtr, ytr); Xte, yte = _restrict(Xte, yte)

    cls_idx = [np.where(ytr == c)[0] for c in range(N_TASK)]
    ctx_len = (N_SUP + N_QRY) * (N_CTX + N_QP + 1)
    logging.info(f"train {Xtr.shape}  OBS_FRAC={OBS_FRAC} → N_CTX={N_CTX}  eff_batch={MICRO_BATCH*ACCUM} "
                 f"(micro {MICRO_BATCH}×{ACCUM})  context_len(train)={ctx_len} tokens")
    logging.info(f"2-WAY task digits={TASK_DIGITS} (n_train per class: "
                 f"{[len(c) for c in cls_idx]});  ANON_LABELS={ANON_LABELS};  chance = {1/N_TASK:.2f}")

    p = init(jax.random.PRNGKey(SEED), D, N_LAYERS)
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, 200, NUM_STEPS)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4)); st = opt.init(p)
    rng = np.random.default_rng(SEED)
    hist = {k: [] for k in ("step", "loss", "pix_loss", "lab_loss", "lab_acc_train_batch",
                            "label_te", "label_tr", "ink_te", "ink_tr", "bg")}
    t0 = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        mics = [build_train(Xtr, ytr, rng) for _ in range(ACCUM)]
        stacked = tuple(jnp.asarray(np.stack([m[i] for m in mics])) for i in range(6))  # incl is_lab
        p, st, loss, (pix_l, lab_l, lab_a) = train_step(opt, p, st, *stacked)
        if step % EVAL_EVERY == 0 or step == 1:
            la, ci, bg = evaluate(p, Xtr, ytr, cls_idx, Xte, yte, 1)          # query from TEST
            lat, cit, _ = evaluate(p, Xtr, ytr, cls_idx, Xtr, ytr, 2)         # query from TRAIN
            for k, v in zip(("step", "loss", "pix_loss", "lab_loss", "lab_acc_train_batch",
                             "label_te", "label_tr", "ink_te", "ink_tr", "bg"),
                            (step, float(loss), float(pix_l), float(lab_l), float(lab_a),
                             la, lat, ci, cit, bg)):
                hist[k].append(v)
            logging.info(f"step {step:5d}  loss {float(loss):.3f} [pix {float(pix_l):.3f} | LAB {float(lab_l):.3f}]  "
                         f"label tr/te {lat:.3f}/{la:.3f}  (train-batch lab_acc {float(lab_a):.3f})  "
                         f"ink tr/te {cit:.3f}/{ci:.3f}  ({time.perf_counter()-t0:.0f}s)")
    elapsed = time.perf_counter() - t0
    final = {"label_te": hist["label_te"][-1], "label_tr": hist["label_tr"][-1],
             "lab_loss": hist["lab_loss"][-1], "pix_loss": hist["pix_loss"][-1],
             "lab_acc_train_batch": hist["lab_acc_train_batch"][-1],
             "ink_te": hist["ink_te"][-1], "bg": hist["bg"][-1]}
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {"experiment": EXP_NAME, "name": f"4v9 with 8 layers (depth test)",
           "time_s": elapsed, "n_params": n_params(p), "obs_frac": OBS_FRAC, "n_ctx": N_CTX,
           "task_digits": list(TASK_DIGITS), "n_task": N_TASK, "chance": 1 / N_TASK,
           "anon_labels": ANON_LABELS, "control_exp10_label": 0.828,
           "eff_batch": MICRO_BATCH * ACCUM, "context_len": ctx_len, **final, "history": hist}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
