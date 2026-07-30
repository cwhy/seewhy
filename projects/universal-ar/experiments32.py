"""
Universal AR — exp31: strip the task to its bones with PCA features.

Everything measured so far says the problem is not representation:
  * exp28 — deterministic labels, same model, 4v9 -> 0.875. The encoder sees the shape.
  * exp29/30 — giving the label task 25%/73% of the gradient instead of 1.1% changed
    nothing (and 73% broke retrieval). Not a signal-starvation problem.

And a baseline on these very features shows what the real gap is:

    supervised linear probe on 32 binned PCs : 0.946
    1-shot Euclidean NN on the same features : 0.526      (chance 0.500)

That is the whole story. 0-vs-1 works everywhere because plain Euclidean matching
already gets 0.87 on it. 4-vs-9 needs a LEARNED metric — the discriminative direction
between the clusters — with the support used only to decide which side is which slot.
Seven architectures all sat at 0.50, i.e. none of them learned any metric at all.

This removes every remaining excuse: PCA to 32 components, quantile-binned, and ALL 32
present for every sample. No pooling loss, no partial observation, no position
sparsity, ~660 tokens per episode. The discriminative direction is linearly available.

  learns  -> binding is formable; the earlier failures were about metric learning being
             hard in a sparse, high-dimensional, partially observed token space
  fails   -> the architecture cannot learn a task-adapted metric even when handed clean
             dense features, which is a statement about the objective, not the data
"""

import json, time, logging
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp32"
JSONL = Path(__file__).parent / "results.jsonl"

DATASET = "mnist"
K = 32; N_CLASSES = 10
N_PC = 32                              # PCA components — each is one "position"
POS_PIX = N_PC; POS_LABEL = N_PC; N_POS = N_PC + 1
N_CONTENT = K + N_CLASSES; MASK_ID = N_CONTENT; N_VAL = N_CONTENT + 1; V_REFS = 64
HEAD_DIM = 32
D, N_LAYERS = 256, 4
N_CTX = N_PC                            # ALL 32 components observed — nothing withheld
N_QP = 0                                # nothing held out: full observation by design
N_RETR = 8                             # retrieval canary (address IS in context)
N_SUP, N_QRY = 10, 6                    # training: random support + query
Q_EVAL = 8
MICRO_BATCH, ACCUM = 4, 2              # effective batch = 8, via gradient accumulation
ANON_LABELS = True                     # per-episode random class→label-token permutation
TASK_DIGITS = (0, 1)                   # EASY pair — control on the same PCA features
N_TASK = len(TASK_DIGITS)              # → chance = 1/N_TASK = 0.50
LR, SEED = 3e-4, 0
NUM_STEPS, EVAL_EVERY = 12000, 1000   # episode is ~660 tokens, so steps are cheap


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


def loss_fn(p, pos, val, ref, target, isq, is_lab, is_retr):
    """Same objective as before; aux splits it into the PIXEL and LABEL components.

    Labels are only ~1/17 of the scored tokens (16 masked pixels + 1 label per
    sample), so the combined loss is dominated by pixels and hides whether the
    label task is being learned at all. Report the two separately.
    """
    logits = forward(p, pos, val, ref)                    # single forward pass
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(target, 0, N_CONTENT - 1))
    correct = (jnp.argmax(logits, -1) == target)
    gen = 1 - is_retr
    m = dict(pix_gen=isq * (1 - is_lab) * gen, pix_retr=isq * (1 - is_lab) * is_retr,
             lab_gen=is_lab * gen,             lab_retr=is_lab * is_retr)
    acc = {k: (correct * v).sum() / (v.sum() + 1e-6) for k, v in m.items()}
    mean = lambda k: (ce * m[k]).sum() / (m[k].sum() + 1e-6)
    pix_loss, lab_loss = mean("pix_gen"), mean("lab_gen")
    loss = mean("pix_retr") + mean("lab_retr") + lab_loss      # per-task means, not one pooled mean
    return loss, (pix_loss, lab_loss, acc["lab_gen"], acc["lab_retr"], acc["pix_retr"])


@partial(jax.jit, static_argnums=(0,))
def train_step(opt, p, st, pos, val, ref, tgt, isq, is_lab, is_retr):
    """Gradient accumulation: leading axis of each arg is ACCUM micro-batches."""
    def micro(_, xs):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, *xs)
        return None, (loss, aux, g)
    _, (losses, auxes, grads) = jax.lax.scan(micro, None, (pos, val, ref, tgt, isq, is_lab, is_retr))
    g = jax.tree_util.tree_map(lambda x: x.mean(0), grads)     # average grads over micro-batches
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, losses.mean(), tuple(a.mean() for a in auxes)


def _bin(px): return int(min(max(px, 0), K - 1))   # values are already quantile bins


def _alloc(S, n_given):
    # per sample: context + retrieval queries + generalisation queries + label; given samples add
    # one extra label-RETRIEVAL query (its label is present in context, so it can be copied)
    T = S * (N_CTX + N_RETR + N_QP + 1) + n_given
    z = lambda d: np.zeros((MICRO_BATCH, T), d)
    return [z(np.int32), z(np.int32), z(np.int32), -np.ones((MICRO_BATCH, T), np.int32),
            z(np.float32), z(np.float32), z(np.float32)]   # pos,val,ref,tgt,isq,is_lab,is_retr


def _emit(arrs, b, t, img, lab_slot, r, given, ok, pool, sel, rng):
    """lab_slot = the (possibly permuted) label-token index for this sample's class.

    Emits, per sample:
      * N_CTX  context pixel tokens                       (value GIVEN)
      * N_RETR RETRIEVAL queries  — positions drawn from THIS sample's own context,
        so the (pos, value, ref) triple is present; the answer can be copied.
      * N_QP   generalisation queries — held-out positions, joint absent (as before)
      * the label token: given, or masked-and-predicted
      * if the label is given, one extra label-RETRIEVAL query for the same ref
    """
    pos, val, ref, tgt, isq, is_lab, is_retr = arrs
    ctx_pos = pool[sel[:N_CTX]]
    for pp in ctx_pos:
        pos[b, t], val[b, t], ref[b, t] = pp, _bin(img[pp]), r; t += 1
    for pp in ctx_pos[rng.permutation(len(ctx_pos))[:N_RETR]]:          # RETRIEVAL (address in context)
        pos[b, t], val[b, t], ref[b, t] = pp, MASK_ID, r
        tgt[b, t], isq[b, t], is_retr[b, t] = _bin(img[pp]), 1.0, 1.0; t += 1
    for pp in pool[sel[N_CTX:N_CTX + N_QP]]:                            # generalisation (held out)
        pos[b, t], val[b, t], ref[b, t], tgt[b, t], isq[b, t] = pp, MASK_ID, r, _bin(img[pp]), 1.0; t += 1
    if given:
        pos[b, t], val[b, t], ref[b, t] = POS_LABEL, K + lab_slot, r; t += 1
        pos[b, t], val[b, t], ref[b, t] = POS_LABEL, MASK_ID, r        # label RETRIEVAL
        tgt[b, t], isq[b, t], is_lab[b, t], is_retr[b, t] = K + lab_slot, 1.0, 1.0, 1.0; t += 1
    else:
        pos[b, t], val[b, t], ref[b, t] = POS_LABEL, MASK_ID, r
        if ok:
            tgt[b, t], isq[b, t], is_lab[b, t] = K + lab_slot, 1.0, 1.0
        t += 1
    return t


def _perm(rng):
    """Per-episode class→label-token permutation over the N_TASK label slots."""
    return rng.permutation(N_TASK) if ANON_LABELS else np.arange(N_TASK)


def build_train(Xs, ys, rng):
    S, n_pool = N_SUP + N_QRY, min(2 * N_CTX, POS_PIX); arrs = _alloc(S, N_SUP)
    for b in range(MICRO_BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        perm = _perm(rng)                        # fresh label semantics for THIS episode
        si = rng.integers(0, Xs.shape[0], N_SUP); qi = rng.integers(0, Xs.shape[0], N_QRY)
        sup_classes = set(int(ys[j]) for j in si)
        for k, j in enumerate(si):
            t = _emit(arrs, b, t, Xs[j], int(perm[int(ys[j])]), int(refs[k]), True, True, pool, rng.permutation(n_pool), rng)
        for k, j in enumerate(qi):
            cls = int(ys[j])
            t = _emit(arrs, b, t, Xs[j], int(perm[cls]), int(refs[N_SUP + k]), False,
                      cls in sup_classes, pool, rng.permutation(n_pool), rng)
    return tuple(np.asarray(a) for a in arrs)


def build_eval_bal(Xtr, ytr, cls_idx, Xq, yq, rng):
    S, n_pool = N_TASK + Q_EVAL, min(2 * N_CTX, POS_PIX); arrs = _alloc(S, N_TASK)
    for b in range(MICRO_BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        perm = _perm(rng)                        # fresh label semantics for THIS episode
        reps = [(Xtr[cls_idx[c][rng.integers(len(cls_idx[c]))]], c, True) for c in range(N_TASK)]
        qi = rng.integers(0, Xq.shape[0], Q_EVAL)
        qry = [(Xq[j], int(yq[j]), False) for j in qi]
        for k, (img, cls, given) in enumerate(reps + qry):
            t = _emit(arrs, b, t, img, int(perm[cls]), int(refs[k]), given, True, pool, rng.permutation(n_pool), rng)
    return tuple(jnp.asarray(a) for a in arrs)


@jax.jit
def eval_metrics(p, pos, val, ref, tgt, isq, is_lab, is_retr):
    """Four separately-reported tasks: {label, pixel} x {retrieval, generalisation}."""
    correct = (jnp.argmax(forward(p, pos, val, ref), -1) == tgt)
    gen = 1 - is_retr; is_pix = isq * (1 - is_lab)
    lab_gen, lab_retr = is_lab * gen, is_lab * is_retr
    ink_gen = is_pix * gen * (tgt > 0)                       # ink-only, held-out pixels
    ink_retr = is_pix * is_retr * (tgt > 0)                  # ink-only, in-context pixels
    f = lambda m: (correct * m).sum() / (m.sum() + 1e-6)
    return f(lab_gen), f(lab_retr), f(ink_gen), f(ink_retr)


def evaluate(p, Xtr, ytr, cls_idx, Xte, yte, seed):
    rng = np.random.default_rng(seed); acc = np.zeros(4)
    for _ in range(4):
        acc += np.array([float(x) for x in eval_metrics(p, *build_eval_bal(Xtr, ytr, cls_idx, Xte, yte, rng))])
    return tuple(acc / 4)      # (label_gen, label_retr, ink_gen, ink_retr)


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

    # ── PCA to N_PC components, then quantile-bin each component into K levels ──
    # Every sample now has ALL N_PC components observed. No pooling loss, no partial
    # observation, no position sparsity — and the 4v9 discriminative direction is
    # linearly available (measured: linear probe 0.946 on exactly these binned features,
    # while 1-shot Euclidean NN gets only 0.526 — so success requires a LEARNED metric).
    kk = np.isin(ytr, TASK_DIGITS)
    mu = Xtr[kk].mean(0)
    _, _, Vt = np.linalg.svd(Xtr[kk] - mu, full_matrices=False)
    Pc = Vt[:N_PC]
    proj = lambda A: (A - mu) @ Pc.T
    Atr = proj(Xtr)
    edges = [np.quantile(Atr[kk][:, j], np.linspace(0, 1, K + 1)[1:-1]) for j in range(N_PC)]
    binit = lambda A: np.stack([np.digitize(A[:, j], edges[j]) for j in range(N_PC)], 1).astype(np.float32)
    Xtr = binit(Atr); Xte = binit(proj(Xte))
    logging.info(f"PCA-{N_PC} binned to {K} levels; each sample is {N_PC} tokens, all observed")

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
    hist = {k: [] for k in ("step", "loss", "pix_loss", "lab_loss",
                            "tr_lab_gen", "tr_lab_retr", "tr_pix_retr",
                            "label_gen_te", "label_retr_te", "ink_gen_te", "ink_retr_te",
                            "label_gen_tr", "label_retr_tr", "ink_gen_tr", "ink_retr_tr")}
    t0 = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        mics = [build_train(Xtr, ytr, rng) for _ in range(ACCUM)]
        stacked = tuple(jnp.asarray(np.stack([m[i] for m in mics])) for i in range(7))  # incl is_lab,is_retr
        p, st, loss, (pix_l, lab_l, tlg, tlr, tpr) = train_step(opt, p, st, *stacked)
        if step % EVAL_EVERY == 0 or step == 1:
            lg_te, lr_te, ig_te, ir_te = evaluate(p, Xtr, ytr, cls_idx, Xte, yte, 1)   # query from TEST
            lg_tr, lr_tr, ig_tr, ir_tr = evaluate(p, Xtr, ytr, cls_idx, Xtr, ytr, 2)   # query from TRAIN
            for k, v in zip(("step", "loss", "pix_loss", "lab_loss",
                             "tr_lab_gen", "tr_lab_retr", "tr_pix_retr",
                             "label_gen_te", "label_retr_te", "ink_gen_te", "ink_retr_te",
                             "label_gen_tr", "label_retr_tr", "ink_gen_tr", "ink_retr_tr"),
                            (step, float(loss), float(pix_l), float(lab_l),
                             float(tlg), float(tlr), float(tpr),
                             lg_te, lr_te, ig_te, ir_te, lg_tr, lr_tr, ig_tr, ir_tr)):
                hist[k].append(v)
            logging.info(f"step {step:5d}  loss {float(loss):.3f} [pix {float(pix_l):.3f} | LAB {float(lab_l):.3f}]  "
                         f"RETRIEVAL lab {lr_te:.3f} pix {ir_te:.3f}  |  GENERALISE lab {lg_te:.3f} pix {ig_te:.3f}  "
                         f"(train-q: retr-lab {lr_tr:.3f} gen-lab {lg_tr:.3f})  ({time.perf_counter()-t0:.0f}s)")
    elapsed = time.perf_counter() - t0
    final = {k: hist[k][-1] for k in ("label_gen_te", "label_retr_te", "ink_gen_te", "ink_retr_te",
                                      "label_gen_tr", "label_retr_tr", "lab_loss", "pix_loss")}
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {"experiment": EXP_NAME, "name": f"4v9 anonymised on PCA-{N_PC} features, all components observed",
           "time_s": elapsed, "n_params": n_params(p), "obs_frac": OBS_FRAC, "n_ctx": N_CTX,
           "task_digits": list(TASK_DIGITS), "n_task": N_TASK, "chance": 1 / N_TASK,
           "anon_labels": ANON_LABELS, "control_exp10_label": 0.828,
           "eff_batch": MICRO_BATCH * ACCUM, "context_len": ctx_len, **final, "history": hist}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
