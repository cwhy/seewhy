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
SGD_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
ES_GENS, N_POP, RANK, SIGMA, ES_LR = 400, 64, 8, 0.002, 3e-3

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


def gen_label_ce(params, arrs):
    """CE on the GENERALISE-label tokens only — the term that is stuck at ln 2."""
    pos, val, ref, tgt, isq, is_lab, is_retr = arrs
    logits = e.forward(params, pos, val, ref)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.clip(tgt, 0, e.N_CONTENT - 1))
    m = is_lab * (1 - is_retr)
    return (ce * m).sum() / (m.sum() + 1e-6)


# ── phase 1 · SGD ──
sch = optax.warmup_cosine_decay_schedule(0., e.LR, 200, SGD_STEPS)
opt = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sch, weight_decay=1e-4))
st = opt.init(p); rng = np.random.default_rng(0)
for i in range(1, SGD_STEPS + 1):
    mics = [e.build_train(X, y, rng) for _ in range(e.ACCUM)]
    stk = tuple(jnp.asarray(np.stack([m[j] for m in mics])) for j in range(7))
    p, st, loss, aux = e.train_step(opt, p, st, *stk)
m = e.evaluate(p, X, y, cls, XT, yT, 1)
print(f"after SGD:  generalise label {float(m[0]):.3f}   retrieval label {float(m[1]):.3f}", flush=True)

# ── phase 2 · EGGROLL ES on the stuck objective ──
leaves0, treedef = jax.tree_util.tree_flatten(p)


def make_eps(key):
    out = []
    for idx, v in enumerate(leaves0):
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
    return gen_label_ce(pos_p, arrs) - gen_label_ce(neg_p, arrs), eps


@jax.jit
def es_step(leaves, opt_state, arrs, keys):
    fit, epses = jax.vmap(pair, in_axes=(0, None, None))(keys, leaves, arrs)
    grad = [jnp.einsum("i,i...->...", fit, ep) / (N_POP * RANK * SIGMA ** 2) for ep in epses]
    upd, opt_state = tx.update(grad, opt_state, leaves)
    return optax.apply_updates(leaves, upd), opt_state, jnp.mean(jnp.abs(fit))


tx = optax.adam(ES_LR)
leaves = list(leaves0 := jax.tree_util.tree_flatten(p)[0])
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
          open(ROOT / f"projects/universal-ar/es_escape_{WHICH}.json", "w"), indent=1)
print("saved", flush=True)
