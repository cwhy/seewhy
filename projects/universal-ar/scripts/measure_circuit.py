"""Train briefly, then MEASURE the retrieval circuit in the trained weights.

How does a query (MASK, pos=b, ref=c) recover the value a?
  SELECT     score = x_q^T (Wq Wk^T / sqrt(32)) x_j   must fire only when pos AND ref match
  TRANSPORT  out   = Wo Wv x_j                        carries E_val[a] into the query residual
  DECODE     logit = head_W^T LN(...)                 must read that back out as value a

So the OV copy matrix  C = E_val @ Wv @ Wo @ head_W  should be ~diagonal, and the QK
form should be diagonal-dominant on pos and ref but NOT on value.

Usage (server): uv run python projects/universal-ar/scripts/measure_circuit.py
"""
import sys, json, importlib.util
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e", str(ROOT / "projects/universal-ar/experiments20.py"))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
from shared_lib.datasets import load_supervised_image

STEPS = 3000

d = load_supervised_image("mnist")
X = np.asarray(d.X.reshape(d.n_samples, -1), np.float32); y = np.asarray(d.y)
keep = np.isin(y, e.TASK_DIGITS); rm = {v: i for i, v in enumerate(e.TASK_DIGITS)}
X = X[keep]; y = np.array([rm[int(v)] for v in y[keep]], np.int32)
cls = [np.where(y == c)[0] for c in range(e.N_TASK)]

p = e.init(jax.random.PRNGKey(0), e.D, e.N_LAYERS)
sch = optax.warmup_cosine_decay_schedule(0., 3e-4, 200, STEPS)
opt = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sch, weight_decay=1e-4))
st = opt.init(p); rng = np.random.default_rng(0)
for i in range(1, STEPS + 1):
    mics = [e.build_train(X, y, rng) for _ in range(e.ACCUM)]
    stk = tuple(jnp.asarray(np.stack([m[j] for m in mics])) for j in range(7))
    p, st, loss, aux = e.train_step(opt, p, st, *stk)
    if i % 1000 == 0:
        m = e.evaluate(p, X, y, cls, X, y, 1)
        print(f"step {i}  retrieval lab {m[1]:.3f}  pix {m[3]:.3f}", flush=True)

acc = e.evaluate(p, X, y, cls, X, y, 1)
print(f"FINAL retrieval  lab {acc[1]:.3f}  pix {acc[3]:.3f}", flush=True)

H, HD = e.D // e.HEAD_DIM, e.HEAD_DIM
Ev, Ep, Er = np.asarray(p["val_emb"]), np.asarray(p["pos_emb"]), np.asarray(p["ref_emb"])
Wh = np.asarray(p["head_W"])
res = {"retr_lab": float(acc[1]), "retr_pix": float(acc[3]), "steps": STEPS, "layers": []}


def diag_dom(E, Mq):
    """How far the diagonal of E M E^T stands above the off-diagonal, in sigma."""
    S = E @ Mq @ E.T
    n = S.shape[0]; dg = np.diag(S)
    off = (S.sum() - dg.sum()) / (n * n - n)
    return float((dg.mean() - off) / (S.std() + 1e-9))


for li, L in enumerate(p["layers"]):
    Wqkv = np.asarray(L["Wqkv"]); Wo = np.asarray(L["Wo"])
    Wq, Wk, Wv = Wqkv[:, :e.D], Wqkv[:, e.D:2*e.D], Wqkv[:, 2*e.D:]
    lay = {"layer": li, "heads": []}
    for h in range(H):
        sl = slice(h * HD, (h + 1) * HD)
        Mq = Wq[:, sl] @ Wk[:, sl].T / np.sqrt(HD)            # QK bilinear form
        Ccopy = Ev @ Wv[:, sl] @ Wo[sl, :] @ Wh               # OV copy matrix (43, 42)
        sq = Ccopy[:e.N_CONTENT]
        lay["heads"].append({
            "head": h,
            "pos_dd": diag_dom(Ep[:200], Mq),
            "ref_dd": diag_dom(Er, Mq),
            "val_dd": diag_dom(Ev, Mq),
            "copy_top1": float(np.mean(np.argmax(sq, axis=1) == np.arange(e.N_CONTENT))),
        })
    res["layers"].append(lay)
    top = max(lay["heads"], key=lambda z: z["copy_top1"])
    print(f"layer {li}: best copy_top1 {top['copy_top1']:.3f} (head {top['head']})  "
          f"pos_dd {top['pos_dd']:+.2f}  ref_dd {top['ref_dd']:+.2f}  val_dd {top['val_dd']:+.2f}", flush=True)

best = max((h for l in res["layers"] for h in l["heads"]), key=lambda z: z["copy_top1"])
print("BEST COPY HEAD:", json.dumps(best), flush=True)
json.dump(res, open(ROOT / "projects/universal-ar/circuit.json", "w"), indent=1)
print("saved circuit.json", flush=True)
