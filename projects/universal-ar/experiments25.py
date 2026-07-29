"""
Universal AR — exp20: add RETRIEVAL-ONLY data (and measure it separately).

Every task so far asked the model to predict a value that was ABSENT from the
context (the hold-out principle). Nothing ever trained or measured the more basic
primitive: given an address (pos, ref) whose value IS present in the context, find
that token and copy its value.

That primitive is the first hop of match-and-copy, so it is worth training directly
and measuring on its own. Per sample this run now emits:

    N_CTX  context pixel tokens                       (value given)
    N_RETR RETRIEVAL pixel queries  — positions taken from THIS sample's own
           context, so the (pos, value, ref) triple is present and copyable
    N_QP   generalisation pixel queries — held-out positions (joint absent)
    label token: given, or masked-and-predicted
    + for every given-label sample, a label-RETRIEVAL query on the same ref

Four tasks are now reported separately: {label, pixel} x {retrieval, generalisation}.

Diagnostic value: if RETRIEVAL accuracy is high but generalisation stays at chance,
addressing works and the failure is specifically the cross-sample comparison. If
retrieval is ALSO poor, the failure is far more basic than shape matching.

Base config = exp15 (4 vs 9, anonymised labels, OBS_FRAC=0.5, eff batch 8).
"""

import json, time, logging
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp25"
JSONL = Path(__file__).parent / "results.jsonl"

DATASET = "mnist"
K = 32; N_CLASSES = 10; POS_PIX = 784; POS_LABEL = 784; N_POS = 785
N_CONTENT = K + N_CLASSES; MASK_ID = N_CONTENT; N_VAL = N_CONTENT + 1; V_REFS = 64
HEAD_DIM = 32
D, N_LAYERS = 256, 4
OBS_FRAC = 0.5                          # fraction of the image each sample observes
N_CTX = int(round(OBS_FRAC * POS_PIX))  # 392
N_QP = 16
N_RETR = 16                            # RETRIEVAL-only pixel queries per sample (address IS in context)
N_SUP, N_QRY = 10, 6                    # training: random support + query
Q_EVAL = 8
MICRO_BATCH, ACCUM = 4, 2              # effective batch = 8, via gradient accumulation
ANON_LABELS = True
HYPER_MODE = "film"              # "weights" = generated low-rank W1+W2 ; "film" = FiLM + generated W2
R_LOW, H_HID = 64, 64          # low-rank width, hidden width of the generated MLP                     # per-episode random class→label-token permutation
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

    # ── context-generated function: theta = g(support) ; logits = f_theta(query) ──
    k2 = jax.random.split(jax.random.PRNGKey(SEED + 101), 12); j = iter(k2)
    # deep-sets encoder over (sample summary, its label slot)
    p["ds_W1"] = lin(next(j), (Dm + N_TASK, Dm)); p["ds_b1"] = jnp.zeros(Dm)
    p["ds_W2"] = lin(next(j), (Dm, Dm));          p["ds_b2"] = jnp.zeros(Dm)
    # fixed learned bases for the low-rank first layer of f
    p["U"] = lin(next(j), (Dm, R_LOW))
    p["V"] = lin(next(j), (R_LOW, H_HID))
    # generators: h -> parameters of f
    p["gen_a"]  = lin(next(j), (Dm, R_LOW));  p["gen_a_b"]  = jnp.zeros(R_LOW)      # low-rank middle
    p["gen_b1"] = lin(next(j), (Dm, H_HID));  p["gen_b1_b"] = jnp.zeros(H_HID)      # hidden bias
    p["gen_g"]  = lin(next(j), (Dm, H_HID));  p["gen_g_b"]  = jnp.ones(H_HID)       # FiLM scale
    p["gen_W2"] = lin(next(j), (Dm, H_HID * N_TASK)) * 0.1
    p["gen_W2_b"] = jnp.zeros(H_HID * N_TASK)                                        # output layer
    p["W1_fixed"] = lin(next(j), (Dm, H_HID))                                        # FiLM variant
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


def forward(p, pos, val, ref, want_states=False):
    x = onehot_mm(pos, p["pos_emb"], N_POS) + onehot_mm(val, p["val_emb"], N_VAL) + onehot_mm(ref, p["ref_emb"], V_REFS)
    for Lp in p["layers"]:
        x = _layer(Lp, x)
    xf = ln(x, p["lnf_g"], p["lnf_b"])
    logits = xf @ p["head_W"] + p["head_b"]
    return (logits, xf) if want_states else logits


def pool_by_ref(x, ref):
    """Mean-pool token states into one summary per ref.  x (B,N,D), ref (B,N) -> (B,V_REFS,D)."""
    oh = jax.nn.one_hot(ref, V_REFS, dtype=jnp.float32)            # (B,N,V)
    tot = jnp.einsum("bnv,bnd->bvd", oh, x)
    cnt = oh.sum(1)[..., None] + 1e-6
    return tot / cnt


def gather_ref(s_all, idx):
    """s_all (B,V,D), idx (B,K) -> (B,K,D)"""
    oh = jax.nn.one_hot(idx, V_REFS, dtype=jnp.float32)
    return jnp.einsum("bkv,bvd->bkd", oh, s_all)


def context_fn(p, s_sup, sup_lab, s_qry):
    """theta = g(support) ; return f_theta(query) logits.  No query-support comparison."""
    # 1 · compile the support set (permutation-invariant deep-sets encoder)
    lab_oh = jax.nn.one_hot(sup_lab, N_TASK, dtype=jnp.float32)     # (B,M,N_TASK)
    u = jnp.concatenate([s_sup, lab_oh], -1) @ p["ds_W1"] + p["ds_b1"]
    u = jax.nn.gelu(u) @ p["ds_W2"] + p["ds_b2"]
    h = u.mean(1)                                                   # (B,D)

    # 2 · generate the parameters of f
    if HYPER_MODE == "weights":
        a = h @ p["gen_a"] + p["gen_a_b"]                           # (B,R)
        W1 = jnp.einsum("dr,br,rk->bdk", p["U"], a, p["V"])         # (B,D,H) low-rank
    else:                                                            # FiLM: fixed W1, generated scale
        g = h @ p["gen_g"] + p["gen_g_b"]                           # (B,H)
        W1 = p["W1_fixed"][None] * g[:, None, :]
    b1 = h @ p["gen_b1"] + p["gen_b1_b"]                            # (B,H)
    W2 = (h @ p["gen_W2"] + p["gen_W2_b"]).reshape(-1, H_HID, N_TASK)

    # 3 · run the query through it  (GELU: must be nonlinear, else this is prototypes)
    z = jax.nn.gelu(jnp.einsum("bkd,bdh->bkh", s_qry, W1) + b1[:, None, :])
    return jnp.einsum("bkh,bhc->bkc", z, W2)                        # (B,K,N_TASK)


def loss_fn(p, pos, val, ref, target, isq, is_lab, is_retr, sup_ref, sup_lab, qry_ref, qry_lab):
    """Same objective as before; aux splits it into the PIXEL and LABEL components.

    Labels are only ~1/17 of the scored tokens (16 masked pixels + 1 label per
    sample), so the combined loss is dominated by pixels and hides whether the
    label task is being learned at all. Report the two separately.
    """
    logits, xf = forward(p, pos, val, ref, want_states=True)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(target, 0, N_CONTENT - 1))
    # token head now trains pixels + RETRIEVAL labels only (generalise-labels move to f_theta)
    tok_m = isq * (1 - is_lab * (1 - is_retr))
    tok_loss = (ce * tok_m).sum() / (tok_m.sum() + 1e-6)
    # the context-generated function predicts the held-out labels
    s_all = pool_by_ref(xf, ref)
    hyp = context_fn(p, gather_ref(s_all, sup_ref), sup_lab, gather_ref(s_all, qry_ref))
    hyp_ce = optax.softmax_cross_entropy_with_integer_labels(hyp, qry_lab)
    hyp_loss = hyp_ce.mean()
    hyp_acc = jnp.mean(jnp.argmax(hyp, -1) == qry_lab)
    loss = tok_loss + hyp_loss
    correct = (jnp.argmax(logits, -1) == target)
    gen = 1 - is_retr
    m = dict(pix_gen=isq * (1 - is_lab) * gen, pix_retr=isq * (1 - is_lab) * is_retr,
             lab_retr=is_lab * is_retr)
    acc = {k: (correct * v).sum() / (v.sum() + 1e-6) for k, v in m.items()}
    pix_loss = (ce * m["pix_gen"]).sum() / (m["pix_gen"].sum() + 1e-6)
    return loss, (pix_loss, hyp_loss, hyp_acc, acc["lab_retr"], acc["pix_retr"])


@partial(jax.jit, static_argnums=(0,))
def train_step(opt, p, st, pos, val, ref, tgt, isq, is_lab, is_retr, sup_ref, sup_lab, qry_ref, qry_lab):
    """Gradient accumulation: leading axis of each arg is ACCUM micro-batches."""
    def micro(_, xs):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, *xs)
        return None, (loss, aux, g)
    _, (losses, auxes, grads) = jax.lax.scan(micro, None, (pos, val, ref, tgt, isq, is_lab, is_retr, sup_ref, sup_lab, qry_ref, qry_lab))
    g = jax.tree_util.tree_map(lambda x: x.mean(0), grads)     # average grads over micro-batches
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, losses.mean(), tuple(a.mean() for a in auxes)


def _bin(px): return int(np.floor(px / 255.0 * (K - 1)))


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
    sup_ref = np.zeros((MICRO_BATCH, N_SUP), np.int32); sup_lab = np.zeros((MICRO_BATCH, N_SUP), np.int32)
    qry_ref = np.zeros((MICRO_BATCH, N_QRY), np.int32); qry_lab = np.zeros((MICRO_BATCH, N_QRY), np.int32)
    for b in range(MICRO_BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        perm = _perm(rng)                        # fresh label semantics for THIS episode
        si = rng.integers(0, Xs.shape[0], N_SUP); qi = rng.integers(0, Xs.shape[0], N_QRY)
        sup_classes = set(int(ys[j]) for j in si)
        for k, j in enumerate(si):
            t = _emit(arrs, b, t, Xs[j], int(perm[int(ys[j])]), int(refs[k]), True, True, pool, rng.permutation(n_pool), rng)
            sup_ref[b, k], sup_lab[b, k] = int(refs[k]), int(perm[int(ys[j])])
        for k, j in enumerate(qi):
            cls = int(ys[j])
            t = _emit(arrs, b, t, Xs[j], int(perm[cls]), int(refs[N_SUP + k]), False,
                      cls in sup_classes, pool, rng.permutation(n_pool), rng)
            qry_ref[b, k], qry_lab[b, k] = int(refs[N_SUP + k]), int(perm[cls])
    return tuple(np.asarray(a) for a in arrs) + (sup_ref, sup_lab, qry_ref, qry_lab)


def build_eval_bal(Xtr, ytr, cls_idx, Xq, yq, rng):
    S, n_pool = N_TASK + Q_EVAL, min(2 * N_CTX, POS_PIX); arrs = _alloc(S, N_TASK)
    sup_ref = np.zeros((MICRO_BATCH, N_TASK), np.int32); sup_lab = np.zeros((MICRO_BATCH, N_TASK), np.int32)
    qry_ref = np.zeros((MICRO_BATCH, Q_EVAL), np.int32); qry_lab = np.zeros((MICRO_BATCH, Q_EVAL), np.int32)
    for b in range(MICRO_BATCH):
        pool = rng.permutation(POS_PIX)[:n_pool]; refs = rng.permutation(V_REFS)[:S]; t = 0
        perm = _perm(rng)                        # fresh label semantics for THIS episode
        reps = [(Xtr[cls_idx[c][rng.integers(len(cls_idx[c]))]], c, True) for c in range(N_TASK)]
        qi = rng.integers(0, Xq.shape[0], Q_EVAL)
        qry = [(Xq[j], int(yq[j]), False) for j in qi]
        for k, (img, cls, given) in enumerate(reps + qry):
            t = _emit(arrs, b, t, img, int(perm[cls]), int(refs[k]), given, True, pool, rng.permutation(n_pool), rng)
            if given: sup_ref[b, k], sup_lab[b, k] = int(refs[k]), int(perm[cls])
            else:     qry_ref[b, k-N_TASK], qry_lab[b, k-N_TASK] = int(refs[k]), int(perm[cls])
    return tuple(jnp.asarray(a) for a in arrs) + tuple(jnp.asarray(a) for a in (sup_ref, sup_lab, qry_ref, qry_lab))


@jax.jit
def eval_metrics(p, pos, val, ref, tgt, isq, is_lab, is_retr, sup_ref, sup_lab, qry_ref, qry_lab):
    """generalise-label now comes from f_theta; the rest from the token head."""
    logits, xf = forward(p, pos, val, ref, want_states=True)
    correct = (jnp.argmax(logits, -1) == tgt)
    is_pix = isq * (1 - is_lab); gen = 1 - is_retr
    lab_retr = is_lab * is_retr
    ink_gen = is_pix * gen * (tgt > 0); ink_retr = is_pix * is_retr * (tgt > 0)
    f = lambda m: (correct * m).sum() / (m.sum() + 1e-6)
    s_all = pool_by_ref(xf, ref)
    hyp = context_fn(p, gather_ref(s_all, sup_ref), sup_lab, gather_ref(s_all, qry_ref))
    return jnp.mean(jnp.argmax(hyp, -1) == qry_lab), f(lab_retr), f(ink_gen), f(ink_retr)


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
        stacked = tuple(jnp.asarray(np.stack([m[i] for m in mics])) for i in range(11))  # + sup/qry ref,lab
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
    row = {"experiment": EXP_NAME, "name": f"4v9 context-generated function ({HYPER_MODE}): theta=g(support), logits=f_theta(query)",
           "time_s": elapsed, "n_params": n_params(p), "obs_frac": OBS_FRAC, "n_ctx": N_CTX,
           "task_digits": list(TASK_DIGITS), "n_task": N_TASK, "chance": 1 / N_TASK,
           "anon_labels": ANON_LABELS, "control_exp10_label": 0.828,
           "eff_batch": MICRO_BATCH * ACCUM, "context_len": ctx_len, **final, "history": hist}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
