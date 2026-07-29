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
EXP_NAME = "exp26"
JSONL = Path(__file__).parent / "results.jsonl"

DATASET = "mnist"
K = 32; N_CLASSES = 10; POS_PIX = 784; POS_LABEL = 784; N_POS = 785
N_CONTENT = K + N_CLASSES; MASK_ID = N_CONTENT; N_VAL = N_CONTENT + 1; V_REFS = 64
HEAD_DIM = 32
D, N_LAYERS = 256, 4
DK = DV = 64                   # KDA per-head key/value dim -> state is DV x DK per head
N_HEADS_K = D // DK            # 4 heads
OBS_FRAC = 0.5                          # fraction of the image each sample observes
N_CTX = int(round(OBS_FRAC * POS_PIX))  # 392
N_QP = 16
N_RETR = 16                            # RETRIEVAL-only pixel queries per sample (address IS in context)
N_SUP, N_QRY = 10, 6                    # training: random support + query
Q_EVAL = 8
MICRO_BATCH, ACCUM = 4, 2              # effective batch = 8, via gradient accumulation
ANON_LABELS = True                     # per-episode random class→label-token permutation
TASK_DIGITS = (4, 9)                   # 2-way HARD pair (classic MNIST confusion)
N_TASK = len(TASK_DIGITS)              # → chance = 1/N_TASK = 0.50
LR, SEED = 3e-4, 0
NUM_STEPS, EVAL_EVERY = 8000, 1000


def init(key, Dm, L):
    g = jax.random.split(key, 3 + L * 12 + 3); i = iter(g)
    lin = lambda k, s: jax.random.normal(k, s) * (1.0 / s[0] ** 0.5)
    p = {"pos_emb": jax.random.normal(next(i), (N_POS, Dm)) * 0.02,
         "val_emb": jax.random.normal(next(i), (N_VAL, Dm)) * 0.02,
         "ref_emb": jax.random.normal(next(i), (V_REFS, Dm)) * 0.02, "layers": []}
    for _ in range(L):
        p["layers"].append(dict(ln1_g=jnp.ones(Dm), ln1_b=jnp.zeros(Dm),
                                Wq=lin(next(i), (Dm, Dm)), Wk=lin(next(i), (Dm, Dm)), Wv=lin(next(i), (Dm, Dm)),
                                Wa=lin(next(i), (Dm, Dm)) * 0.1, ba=jnp.full((Dm,), 3.0),   # per-channel decay, init alpha~0.95
                                Wb=lin(next(i), (Dm, N_HEADS_K)), bb=jnp.zeros(N_HEADS_K),  # write strength
                                Wo=lin(next(i), (Dm, Dm)),
                                ln2_g=jnp.ones(Dm), ln2_b=jnp.zeros(Dm), W1=lin(next(i), (Dm, 4 * Dm)),
                                b1=jnp.zeros(4 * Dm), W2=lin(next(i), (4 * Dm, Dm)), b2=jnp.zeros(Dm)))
    p["lnf_g"] = jnp.ones(Dm); p["lnf_b"] = jnp.zeros(Dm)
    p["head_W"] = lin(next(i), (Dm, N_CONTENT)); p["head_b"] = jnp.zeros(N_CONTENT)
    return p


def n_params(p): return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def kda(x, Lp, is_ctx):
    """Kimi Delta Attention: a matrix-valued memory written by the delta rule.

        forget   S~ = S . Diag(alpha_t)        per-channel decay (KDA's refinement)
        predict  vhat = S~ k_t                  what is currently stored at this key
        correct  e = beta_t (v_t - vhat)        the error
        write    S = S~ + e k_t^T
        read     o_t = S q_t / sqrt(dk)

    Context tokens WRITE (in shuffled order, so decay is unbiased across samples);
    query tokens never write (beta gated to 0) and every token READS the completed
    memory. There is no pooling anywhere: the state is the aggregate.
    """
    B, N, Dm = x.shape; H = N_HEADS_K
    sh = lambda t: t.reshape(B, N, H, DK).transpose(0, 2, 1, 3)          # (B,H,N,DK)
    q = sh(x @ Lp["Wq"]); k = sh(x @ Lp["Wk"]); v = sh(x @ Lp["Wv"])
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)          # DeltaNet convention
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    alpha = jax.nn.sigmoid(sh(x @ Lp["Wa"] + Lp["ba"]))                  # (B,H,N,DK) per-channel
    beta = jax.nn.sigmoid(x @ Lp["Wb"] + Lp["bb"]).transpose(0, 2, 1)    # (B,H,N)

    gate = is_ctx[:, None, :]                                            # (B,1,N) 1 where context
    alpha = alpha * gate[..., None] + (1.0 - gate[..., None])            # queries: no decay
    beta = beta * gate                                                   # queries: no write

    def step(S, t):
        a_t, k_t, v_t, b_t = t                                           # (B,H,DK),(B,H,DK),(B,H,DV),(B,H)
        S = S * a_t[:, :, None, :]                                       # right-mult by Diag(alpha)
        vhat = jnp.einsum("bhvk,bhk->bhv", S, k_t)
        e = b_t[..., None] * (v_t - vhat)
        return S + jnp.einsum("bhv,bhk->bhvk", e, k_t), None

    seq = (alpha.transpose(2, 0, 1, 3), k.transpose(2, 0, 1, 3),
           v.transpose(2, 0, 1, 3), beta.transpose(2, 0, 1))             # scan over tokens
    S, _ = jax.lax.scan(step, jnp.zeros((B, H, DK, DK)), seq)
    o = jnp.einsum("bhvk,bhnk->bhnv", S, q) / DK ** 0.5                  # every token reads final S
    return o.transpose(0, 2, 1, 3).reshape(B, N, Dm) @ Lp["Wo"]


def onehot_mm(ids, table, n):
    return jnp.einsum("bnk,kd->bnd", jax.nn.one_hot(ids, n, dtype=jnp.float32), table)


@jax.checkpoint
def _layer(Lp, x, is_ctx):
    x = x + kda(ln(x, Lp["ln1_g"], Lp["ln1_b"]), Lp, is_ctx)
    return x + (jax.nn.gelu(ln(x, Lp["ln2_g"], Lp["ln2_b"]) @ Lp["W1"] + Lp["b1"]) @ Lp["W2"] + Lp["b2"])


def forward(p, pos, val, ref, is_ctx):
    x = onehot_mm(pos, p["pos_emb"], N_POS) + onehot_mm(val, p["val_emb"], N_VAL) + onehot_mm(ref, p["ref_emb"], V_REFS)
    for Lp in p["layers"]:
        x = _layer(Lp, x, is_ctx)
    return ln(x, p["lnf_g"], p["lnf_b"]) @ p["head_W"] + p["head_b"]


def loss_fn(p, pos, val, ref, target, isq, is_lab, is_retr):
    """Same objective as before; aux splits it into the PIXEL and LABEL components.

    Labels are only ~1/17 of the scored tokens (16 masked pixels + 1 label per
    sample), so the combined loss is dominated by pixels and hides whether the
    label task is being learned at all. Report the two separately.
    """
    logits = forward(p, pos, val, ref, 1.0 - isq)                    # single forward pass
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(target, 0, N_CONTENT - 1))
    loss = (ce * isq).sum() / (isq.sum() + 1e-6)          # unchanged training objective
    correct = (jnp.argmax(logits, -1) == target)
    gen = 1 - is_retr
    m = dict(pix_gen=isq * (1 - is_lab) * gen, pix_retr=isq * (1 - is_lab) * is_retr,
             lab_gen=is_lab * gen,             lab_retr=is_lab * is_retr)
    acc = {k: (correct * v).sum() / (v.sum() + 1e-6) for k, v in m.items()}
    pix_loss = (ce * m["pix_gen"]).sum() / (m["pix_gen"].sum() + 1e-6)
    lab_loss = (ce * m["lab_gen"]).sum() / (m["lab_gen"].sum() + 1e-6)
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



def _shuffle(arrs, rng):
    """Shuffle token ORDER within each episode.

    The episode is a set, and KDA is a sequence model with decay: whatever is written
    first fades most. Emitting samples in contiguous blocks would systematically starve
    the first sample. Shuffling makes the decay unbiased across samples instead of
    trying to suppress it.
    """
    out = [a.copy() for a in arrs]
    for b in range(arrs[0].shape[0]):
        perm = rng.permutation(arrs[0].shape[1])
        for a in out:
            a[b] = a[b][perm]
    return out

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
    return tuple(np.asarray(a) for a in _shuffle(arrs, rng))


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
    return tuple(jnp.asarray(a) for a in _shuffle(arrs, rng))


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
    row = {"experiment": EXP_NAME, "name": f"4v9 + RETRIEVAL-only training data; reports retrieval vs generalisation separately",
           "time_s": elapsed, "n_params": n_params(p), "obs_frac": OBS_FRAC, "n_ctx": N_CTX,
           "task_digits": list(TASK_DIGITS), "n_task": N_TASK, "chance": 1 / N_TASK,
           "anon_labels": ANON_LABELS, "control_exp10_label": 0.828,
           "eff_batch": MICRO_BATCH * ACCUM, "context_len": ctx_len, **final, "history": hist}
    with open(JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
