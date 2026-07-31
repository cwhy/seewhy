"""Can EGGROLL-style ES escape the ln-2 plateau that gradient descent cannot?

Why this is the right tool for what we measured. On the PCA task the failure looks like
an OPTIMISATION problem, not a capacity one:

  * PCA 0v1 is bimodal under SGD — ~4 of 15 runs reach ~1.000, the rest sit at chance,
    and fixed-seed reruns disagree (GPU nondeterminism alone flips the outcome). So a
    solution EXISTS in the landscape and gradient descent finds it only sometimes.
  * The training label loss of a failing run sits exactly at ln 2 = 0.6931 — the model
    emits the uniform prior and gradient descent has no descent direction to follow.
  * Depth does not change the success rate (~20% at both L=4 and L=8).

ES does not follow gradients. It measures FITNESS DIFFERENCES at a finite perturbation
scale sigma, so it can cross a barrier that an infinitesimal gradient cannot see. That
is exactly the failure mode here.

Method — EGGROLL (Sarkar et al.), following projects/es/experiments9.py:
    perturbation of a weight matrix M (m x n) is  E = (A B^T) * sigma,  A: m x r, B: n x r
    antithetic pairs:  fitness = CE(theta + E) - CE(theta - E)
    update:            grad_est = sum_i fitness_i * E_i / (n_pop * rank * sigma^2)
Note their finding, adopted here: the "correct" eps/rank normalisation is too small to
escape a plateau from a cold start; the effective std = sigma * sqrt(rank) is what works.

Design: SGD first (which reliably learns retrieval and usually stalls at chance on the
label task), then ES on the GENERALISE-LABEL objective alone, so every unit of search
pressure goes to the term that is stuck.

  escapes  -> the solution is nearby but gradient-inaccessible; the failure is optimisation
  flat     -> no solution reachable from where SGD lands, at this perturbation scale

usage: uv run python projects/universal-ar/scripts/es_escape.py 32   # 0v1 (solution exists)
       uv run python projects/universal-ar/scripts/es_escape.py 31   # 4v9 (none known)
"""
import sys, json, importlib.util
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
WHICH = sys.argv[1] if len(sys.argv) > 1 else "32"
spec = importlib.util.spec_from_file_location("e", str(ROOT / f"projects/universal-ar/experiments{WHICH}.py"))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
from shared_lib.datasets import load_supervised_image

# small model: ES materialises n_pop full perturbations, so parameter count is the binding
# constraint. D=128/L=2 keeps the whole population in memory at n_pop=64.
D_ES, L_ES = 128, 2
SGD_STEPS = 6000
SCOPE = sys.argv[2] if len(sys.argv) > 2 else "qk_head"   # "qk_head" | "full"
SIGMA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
ES_GENS, N_POP, RANK, ES_LR = 400, 256, 8, 3e-3

# ── data (same PCA pipeline as the experiment) ──
d = load_supervised_image("mnist")
Xtr = np.asarray(d.X.reshape(d.n_samples, -1), np.float32)
Xte = np.asarray(d.X_test.reshape(d.n_test_samples, -1), np.float32)
ytr = np.asarray(d.y); yte = np.asarray(d.y_test)
kk = np.isin(ytr, e.TASK_DIGITS)
mu = Xtr[kk].mean(0)
_, _, Vt = np.linalg.svd(Xtr[kk] - mu, full_matrices=False)
Pc = Vt[:e.N_PC]
proj = lambda A: (A - mu) @ Pc.T
Atr = proj(Xtr)
edges = [np.quantile(Atr[kk][:, j], np.linspace(0, 1, e.K + 1)[1:-1]) for j in range(e.N_PC)]
binit = lambda A: np.stack([np.digitize(A[:, j], edges[j]) for j in range(e.N_PC)], 1).astype(np.float32)
rm = {v: i for i, v in enumerate(e.TASK_DIGITS)}
kt = np.isin(yte, e.TASK_DIGITS)
X = binit(Atr)[kk]; y = np.array([rm[int(v)] for v in ytr[kk]], np.int32)
XT = binit(proj(Xte))[kt]; yT = np.array([rm[int(v)] for v in yte[kt]], np.int32)
cls = [np.where(y == c)[0] for c in range(e.N_TASK)]

p = e.init(jax.random.PRNGKey(0), D_ES, L_ES)
n_par = e.n_params(p)
print(f"task {e.TASK_DIGITS}  D={D_ES} L={L_ES}  params {n_par/1e3:.0f}k  "
      f"ES: n_pop={N_POP} rank={RANK} sigma={SIGMA}", flush=True)


def fitness_ce(params, arrs):
    """Generalise-label CE (the stuck term) PLUS retrieval CE.

    The first attempt optimised the stuck term alone, and ES duly destroyed retrieval
    (1.000 -> 0.47) while gaining nothing. Including retrieval keeps the working
    machinery intact, so any movement on the target is a real escape rather than a
    trade against something the model already had.
    """
    pos, val, ref, tgt, isq, is_lab, is_retr = arrs
    logits = e.forward(params, pos, val, ref)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(tgt, 0, e.N_CONTENT - 1))
    mg = is_lab * (1 - is_retr)
    mr = isq * is_retr
    return ((ce * mg).sum() / (mg.sum() + 1e-6)) + ((ce * mr).sum() / (mr.sum() + 1e-6))


# ── phase 1 · SGD, repeated until it lands in the FAILED mode ──
# The task is bimodal (~25% of runs solve it). ES is only meaningful when started from a
# stalled model: a run that already reached 1.000 has no plateau to escape, and reporting
# "ES kept it at 1.000" would be vacuous.
def run_sgd(seed):
    q = e.init(jax.random.PRNGKey(seed), D_ES, L_ES)
    sch = optax.warmup_cosine_decay_schedule(0., e.LR, 200, SGD_STEPS)
    o = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sch, weight_decay=1e-4))
    s_ = o.init(q); r = np.random.default_rng(seed)
    for _ in range(SGD_STEPS):
        mics = [e.build_train(X, y, r) for _ in range(e.ACCUM)]
        stk = tuple(jnp.asarray(np.stack([mm[j] for mm in mics])) for j in range(7))
        q, s_, _l, _a = e.train_step(o, q, s_, *stk)
    return q, e.evaluate(q, X, y, cls, XT, yT, 1), r

for seed in range(8):
    p, m, rng = run_sgd(seed)
    print(f"SGD seed {seed}:  generalise label {float(m[0]):.3f}   retrieval {float(m[1]):.3f}"
          + ("   <- STALLED, using this" if float(m[0]) < 0.6 else "   (solved, retry)"), flush=True)
    if float(m[0]) < 0.6:
        break
else:
    print("no stalled run found in 8 seeds — nothing to escape", flush=True); sys.exit(0)

# ── phase 2 · EGGROLL ES on the stuck objective ──
paths_leaves, treedef = jax.tree_util.tree_flatten_with_path(p)
leaves0 = [v for _, v in paths_leaves]
names = [jax.tree_util.keystr(path) for path, _ in paths_leaves]


def _layer_of(path):
    for k in path:
        if hasattr(k, "idx"):
            return k.idx
    return None


if SCOPE == "qk_head":
    # last transformer block's attention + the output head. Q/K is exactly where a learned
    # comparison metric would have to live, so this is the smallest defensible subspace.
    ACTIVE = [i for i, (path, _) in enumerate(paths_leaves)
              if (_layer_of(path) == L_ES - 1 and any(t in names[i] for t in ("Wqkv", "Wo")))
              or "head_" in names[i]]
else:
    ACTIVE = list(range(len(leaves0)))
assert len(ACTIVE) >= 3, f"scope selector matched too few tensors: {[names[i] for i in ACTIVE]}"
n_search = sum(int(np.prod(leaves0[i].shape)) for i in ACTIVE)
print(f"ES scope={SCOPE}: {len(ACTIVE)} tensors, {n_search/1e3:.1f}k of {n_par/1e3:.0f}k dims "
      f"searched  (pop/dim = 1:{n_search/N_POP:.0f})", flush=True)
for i in ACTIVE:
    print(f"    {names[i]:<28} {leaves0[i].shape}", flush=True)


def make_eps(key):
    out = []
    for idx, v in enumerate(leaves0):
        if idx not in ACTIVE:
            out.append(jnp.zeros_like(v)); continue
        sk = jax.random.fold_in(key, idx)
        if v.ndim == 2:
            mm, nn = v.shape
            ab = jax.random.normal(sk, (mm + nn, RANK))
            out.append((ab[:mm] @ ab[mm:].T) * SIGMA)      # std = sigma*sqrt(rank), per es/exp8
        else:
            out.append(jax.random.normal(sk, v.shape) * SIGMA)
    return out


def pair(key, leaves, arrs):
    eps = make_eps(key)
    pos_p = jax.tree_util.tree_unflatten(treedef, [a + b for a, b in zip(leaves, eps)])
    neg_p = jax.tree_util.tree_unflatten(treedef, [a - b for a, b in zip(leaves, eps)])
    return fitness_ce(pos_p, arrs) - fitness_ce(neg_p, arrs), eps


@jax.jit
def es_step(leaves, opt_state, arrs, keys):
    fit, epses = jax.vmap(pair, in_axes=(0, None, None))(keys, leaves, arrs)
    grad = [jnp.einsum("i,i...->...", fit, ep) / (N_POP * RANK * SIGMA ** 2) for ep in epses]
    upd, opt_state = tx.update(grad, opt_state, leaves)
    return optax.apply_updates(leaves, upd), opt_state, jnp.mean(jnp.abs(fit))


tx = optax.adam(ES_LR)
leaves = list(leaves0)
es_state = tx.init(leaves)
key = jax.random.PRNGKey(1)
hist = []
for g in range(1, ES_GENS + 1):
    arrs = tuple(jnp.asarray(a) for a in e.build_train(X, y, rng))
    key, sk = jax.random.split(key)
    leaves, es_state, fmag = es_step(leaves, es_state, arrs, jax.random.split(sk, N_POP))
    if g % 50 == 0 or g == 1:
        pg = jax.tree_util.tree_unflatten(treedef, leaves)
        mm = e.evaluate(pg, X, y, cls, XT, yT, 1)
        hist.append(dict(gen=g, gen_lab=float(mm[0]), retr_lab=float(mm[1]), fit=float(fmag)))
        print(f"ES gen {g:4d}  generalise label {float(mm[0]):.3f}  retrieval {float(mm[1]):.3f}  "
              f"|fitness| {float(fmag):.2e}", flush=True)

json.dump({"task": list(e.TASK_DIGITS), "sgd_gen_lab": float(m[0]), "es": hist},
          open(ROOT / f"projects/universal-ar/es_escape_{WHICH}_{SCOPE}_{SIGMA}.json", "w"), indent=1)
print("saved", flush=True)
