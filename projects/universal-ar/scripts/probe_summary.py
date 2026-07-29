"""Is the per-sample POOLED SUMMARY the bottleneck?

exp24/exp25 replaced the readout with a context-generated function and still landed at
ln 2. Both — and every earlier attempt — read from the same object: ~400 pixel tokens
mean-pooled per ref into one 256-d vector. If 4 and 9 are not separable *in that
vector*, no downstream head can help.

This trains the token transformer as usual, then freezes it and fits a plain SUPERVISED
probe (linear, and a 2-layer MLP) on the pooled summaries with TRUE labels — no
in-context anything, no anonymisation. That is the easiest possible use of the summary.

  probe accuracy high  -> the summary carries the shape; the failure is downstream
  probe accuracy ~0.5  -> the summary itself is uninformative; pooling is the bottleneck

Reference: a nearest-neighbour classifier on raw pixels gets ~0.58 at 1-shot and
0.787 at 20-shot on 4-vs-9, and a supervised linear probe on raw pixels should be ~0.95+.

Usage (server): uv run python projects/universal-ar/scripts/probe_summary.py
"""
import sys, json, importlib.util
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp, optax

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e", str(ROOT / "projects/universal-ar/experiments24.py"))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
from shared_lib.datasets import load_supervised_image

STEPS, PROBE_STEPS = 2500, 3000

d = load_supervised_image("mnist")
X = np.asarray(d.X.reshape(d.n_samples, -1), np.float32); y = np.asarray(d.y)
keep = np.isin(y, e.TASK_DIGITS); rm = {v: i for i, v in enumerate(e.TASK_DIGITS)}
X = X[keep]; y = np.array([rm[int(v)] for v in y[keep]], np.int32)
Xte_all = np.asarray(d.X_test.reshape(d.n_test_samples, -1), np.float32); yte_all = np.asarray(d.y_test)
kt = np.isin(yte_all, e.TASK_DIGITS)
Xte = Xte_all[kt]; yte = np.array([rm[int(v)] for v in yte_all[kt]], np.int32)
print(f"4v9 train {X.shape} test {Xte.shape}", flush=True)

# ── 1 · train the token transformer exactly as in exp24 ──
p = e.init(jax.random.PRNGKey(0), e.D, e.N_LAYERS)
sch = optax.warmup_cosine_decay_schedule(0., 3e-4, 200, STEPS)
opt = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sch, weight_decay=1e-4))
st = opt.init(p); rng = np.random.default_rng(0)
for i in range(1, STEPS + 1):
    mics = [e.build_train(X, y, rng) for _ in range(e.ACCUM)]
    stk = tuple(jnp.asarray(np.stack([m[j] for m in mics])) for j in range(11))
    p, st, loss, aux = e.train_step(opt, p, st, *stk)
    if i % 1000 == 0:
        print(f"  step {i} loss {float(loss):.3f} retr_lab {float(aux[3]):.3f}", flush=True)

# ── 2 · harvest pooled summaries with TRUE labels ──
@jax.jit
def summaries(p, pos, val, ref):
    _, xf = e.forward(p, pos, val, ref, want_states=True)
    return e.pool_by_ref(xf, ref)

def harvest(Xs, ys, n_ep, seed):
    r = np.random.default_rng(seed); S, L = [], []
    for _ in range(n_ep):
        arrs = e.build_train(Xs, ys, r)
        pos, val, ref = (jnp.asarray(arrs[k]) for k in range(3))
        s_all = np.asarray(summaries(p, pos, val, ref))          # (B, V_REFS, D)
        sup_ref, sup_lab = np.asarray(arrs[7]), np.asarray(arrs[8])
        qry_ref, qry_lab = np.asarray(arrs[9]), np.asarray(arrs[10])
        for b in range(e.MICRO_BATCH):
            for rr, ll in list(zip(sup_ref[b], sup_lab[b])) + list(zip(qry_ref[b], qry_lab[b])):
                S.append(s_all[b, rr]); L.append(ll)
    return np.stack(S), np.array(L, np.int32)


# IMPORTANT: with ANON_LABELS on, the harvested label is the per-episode PERMUTED slot —
# the same digit is slot 0 in one episode and slot 1 in the next, so a probe trained on
# those slots sits at exactly 0.50 no matter how good the summary is. Turn anonymisation
# OFF for harvesting only, so the recorded slot IS the true class. (The encoder stays
# exactly as trained; this only changes the label token, 1 of ~425 tokens per sample.)
e.ANON_LABELS = False

Str, Ltr = harvest(X, y, 60, 1)
Ste, Lte = harvest(Xte, yte, 20, 2)
print(f"summaries: train {Str.shape} test {Ste.shape}  class balance "
      f"{Ltr.mean():.2f}/{Lte.mean():.2f}", flush=True)

# ── 3 · fit probes (linear and MLP) on the summaries ──
def probe(Str, Ltr, Ste, Lte, hidden=0, steps=PROBE_STEPS, tag=""):
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    D_ = Str.shape[1]
    if hidden:
        w = {"W1": jax.random.normal(k[0], (D_, hidden)) / D_**.5, "b1": jnp.zeros(hidden),
             "W2": jax.random.normal(k[1], (hidden, 2)) / hidden**.5, "b2": jnp.zeros(2)}
        fwd = lambda w, x: jax.nn.gelu(x @ w["W1"] + w["b1"]) @ w["W2"] + w["b2"]
    else:
        w = {"W": jax.random.normal(k[0], (D_, 2)) / D_**.5, "b": jnp.zeros(2)}
        fwd = lambda w, x: x @ w["W"] + w["b"]
    o = optax.adamw(1e-3, weight_decay=1e-4); s_ = o.init(w)
    Xtr_, Ytr_ = jnp.asarray(Str), jnp.asarray(Ltr)
    @jax.jit
    def step(w, s_):
        def lf(w): return optax.softmax_cross_entropy_with_integer_labels(fwd(w, Xtr_), Ytr_).mean()
        l, g = jax.value_and_grad(lf)(w); u, s_ = o.update(g, s_, w)
        return optax.apply_updates(w, u), s_, l
    for _ in range(steps):
        w, s_, l = step(w, s_)
    acc = lambda Xa, Ya: float(jnp.mean(jnp.argmax(fwd(w, jnp.asarray(Xa)), -1) == jnp.asarray(Ya)))
    print(f"  probe {tag:<12} train {acc(Str,Ltr):.3f}  test {acc(Ste,Lte):.3f}", flush=True)
    return acc(Ste, Lte)

print("\nPROBE ON POOLED SUMMARIES (supervised, true labels, frozen encoder):", flush=True)
lin = probe(Str, Ltr, Ste, Lte, 0, tag="linear")
mlp = probe(Str, Ltr, Ste, Lte, 256, tag="MLP-256")

# ── 4 · reference: same probe on RAW PIXELS ──
def raw(Xs, ys, n):
    r = np.random.default_rng(7); i = r.integers(0, Xs.shape[0], n)
    return np.floor(Xs[i] / 255.0 * 31).astype(np.float32), ys[i]
Rtr, rtr = raw(X, y, 4000); Rte, rte = raw(Xte, yte, 1500)
print("\nREFERENCE — same probes on RAW PIXELS (784-d):", flush=True)
rlin = probe(Rtr, rtr, Rte, rte, 0, tag="linear")
rmlp = probe(Rtr, rtr, Rte, rte, 256, tag="MLP-256")

json.dump({"summary_linear": lin, "summary_mlp": mlp, "raw_linear": rlin, "raw_mlp": rmlp},
          open(ROOT / "projects/universal-ar/probe.json", "w"), indent=1)
print("\nsaved probe.json", flush=True)
