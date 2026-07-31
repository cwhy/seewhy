"""Does DEPTH buy the missing comparison circuit? Sweep on the simplest failure case.

The PCA configuration is the cleanest place to ask. Every confound is gone: 32 dense
components per sample, ALL observed, no pooling loss, no partial observation, ~666
tokens per episode, and the 4v9 discriminative direction is linearly available
(supervised linear probe on these exact binned features: 0.946).

And it fails at chance for BOTH pairs under anonymised labels — including 0-vs-1, which
every raw-pixel setup solves outright. PCA removes the position-independent value
channel (a bare ink-count, measured sufficient at 0.889 for 1-shot 0v1), so the model
has to learn position-specific comparison or score chance. It scores chance.

Depth was tried once before (exp18: 8 layers, raw pixels, 4v9) and did nothing, but never
here, where the task is stripped to its bones and a run costs ~2 minutes rather than 2
hours. If more sequential hops can build the circuit, this is where it should appear.

    usage:  uv run python projects/universal-ar/scripts/depth_sweep.py 32   # 0v1
            uv run python projects/universal-ar/scripts/depth_sweep.py 31   # 4v9
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

DEPTHS = [4, 8, 12, 16]
STEPS = e.NUM_STEPS

# ── data: same PCA pipeline as the experiment ──
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
Xtr_b, Xte_b = binit(Atr), binit(proj(Xte))
rm = {v: i for i, v in enumerate(e.TASK_DIGITS)}
kt = np.isin(yte, e.TASK_DIGITS)
X = Xtr_b[kk]; y = np.array([rm[int(v)] for v in ytr[kk]], np.int32)
XT = Xte_b[kt]; yT = np.array([rm[int(v)] for v in yte[kt]], np.int32)
cls = [np.where(y == c)[0] for c in range(e.N_TASK)]
print(f"task {e.TASK_DIGITS}  train {X.shape}  test {XT.shape}  steps {STEPS}", flush=True)

res = {}
for L in DEPTHS:
    p = e.init(jax.random.PRNGKey(0), e.D, L)
    sch = optax.warmup_cosine_decay_schedule(0., e.LR, 200, STEPS)
    opt = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sch, weight_decay=1e-4))
    st = opt.init(p); rng = np.random.default_rng(0)
    for i in range(1, STEPS + 1):
        mics = [e.build_train(X, y, rng) for _ in range(e.ACCUM)]
        stk = tuple(jnp.asarray(np.stack([m[j] for m in mics])) for j in range(7))
        p, st, loss, aux = e.train_step(opt, p, st, *stk)
    m = e.evaluate(p, X, y, cls, XT, yT, 1)
    gen_lab, retr_lab = float(m[0]), float(m[1])
    res[L] = dict(gen_lab=gen_lab, retr_lab=retr_lab, n_params=e.n_params(p))
    print(f"L={L:2d}  params {e.n_params(p)/1e6:5.2f}M   generalise label {gen_lab:.3f}   "
          f"retrieval label {retr_lab:.3f}   (chance {1/e.N_TASK:.2f})", flush=True)

json.dump({"task": list(e.TASK_DIGITS), "depths": res},
          open(ROOT / f"projects/universal-ar/depth_sweep_{WHICH}.json", "w"), indent=1)
print("saved", flush=True)
