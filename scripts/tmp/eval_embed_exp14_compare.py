"""
eval_embed_exp14_compare.py — Side-by-side embed eval: exp14 vs exp14_abl.

Compares encoder embedding quality between:
  - exp14:     full model (recon + kNN variation loss)
  - exp14_abl: ablation  (recon loss only, no kNN)

Both use identical architecture (LATENT_DIM=64, deconvnet decoder).
Metrics: linear probe accuracy + kNN (L2 + cosine, k=1,5,10,20).

Usage:
    uv run python scripts/tmp/eval_embed_exp14_compare.py
"""

import sys, pickle, json, time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import faiss
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT    = Path(__file__).parent.parent.parent
PROJECT = ROOT / "projects" / "variations"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from shared_lib.datasets import load_supervised_image

PROBE_LR     = 1e-3
PROBE_EPOCHS = 30
PROBE_BATCH  = 256
KS           = [1, 5, 10, 20]
N_CLASSES    = 10

# ── Load MNIST once ────────────────────────────────────────────────────────────

data    = load_supervised_image("mnist")
X_train = np.array(data.X.reshape(data.n_samples, 784).astype(np.float32))
Y_train = np.array(data.y, dtype=np.int32)
X_test  = np.array(data.X_test.reshape(data.n_test_samples, 784).astype(np.float32))
Y_test  = np.array(data.y_test, dtype=np.int32)
logging.info(f"MNIST: {len(X_train)} train, {len(X_test)} test")

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_params(exp_name):
    with open(PROJECT / f"params_{exp_name}.pkl", "rb") as f:
        return {k: jnp.array(v) for k, v in pickle.load(f).items()}

def extract_embeddings(encode_fn, params, X, batch=512):
    parts = []
    for i in range(0, len(X), batch):
        xb = jnp.array(X[i:i+batch] / 255.0)
        parts.append(np.array(encode_fn(params, xb)))
    return np.concatenate(parts, 0).astype(np.float32)

def linear_probe(E_tr, Y_tr, E_te, Y_te, latent_dim):
    n_fit = (len(E_tr) // PROBE_BATCH) * PROBE_BATCH
    E_b   = jnp.array(E_tr[:n_fit].reshape(-1, PROBE_BATCH, latent_dim))
    Y_b   = jnp.array(Y_tr[:n_fit].reshape(-1, PROBE_BATCH))

    W, b        = jnp.zeros((latent_dim, N_CLASSES)), jnp.zeros(N_CLASSES)
    probe_opt   = optax.adam(PROBE_LR)
    probe_state = probe_opt.init((W, b))

    def probe_step(carry, inputs):
        (W, b), state = carry
        e, y = inputs
        def ce(wb):
            return optax.softmax_cross_entropy_with_integer_labels(e @ wb[0] + wb[1], y).mean()
        loss, grads = jax.value_and_grad(ce)((W, b))
        updates, new_state = probe_opt.update(grads, state, (W, b))
        return (optax.apply_updates((W, b), updates), new_state), loss

    @jax.jit
    def probe_epoch(W, b, state):
        (W, b), state = jax.lax.scan(probe_step, ((W, b), state), (E_b, Y_b))[0]
        return W, b, state

    for _ in range(PROBE_EPOCHS):
        W, b, probe_state = probe_epoch(W, b, probe_state)

    logits = jnp.array(E_te) @ W + b
    return float((jnp.argmax(logits, 1) == jnp.array(Y_te)).mean())

def knn_l2(E_tr, Y_tr, E_te, Y_te, k):
    index = faiss.IndexFlatL2(E_tr.shape[1])
    index.add(E_tr)
    _, I = index.search(E_te, k)
    preds = np.array([np.bincount(Y_tr[I[i]], minlength=N_CLASSES).argmax() for i in range(len(E_te))])
    return float((preds == Y_te).mean())

def knn_cos(E_tr, Y_tr, E_te, Y_te, k):
    def norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return knn_l2(norm(E_tr), Y_tr, norm(E_te), Y_te, k)

def eval_model(exp_name, encode_fn, latent_dim):
    logging.info(f"\n── {exp_name} ──")
    params = load_params(exp_name)

    t0 = time.perf_counter()
    E_tr = extract_embeddings(encode_fn, params, X_train)
    E_te = extract_embeddings(encode_fn, params, X_test)
    logging.info(f"  Embeddings: {E_tr.shape}  ({time.perf_counter()-t0:.1f}s)")

    probe = linear_probe(E_tr, Y_tr=Y_train, E_te=E_te, Y_te=Y_test, latent_dim=latent_dim)
    logging.info(f"  Linear probe: {probe:.1%}")

    knn_res = {}
    for k in KS:
        knn_res[f"knn_l2_k{k}"]  = knn_l2(E_tr, Y_train, E_te, Y_test, k)
        knn_res[f"knn_cos_k{k}"] = knn_cos(E_tr, Y_train, E_te, Y_test, k)
    logging.info(f"  kNN done")

    row = {
        "experiment"  : f"{exp_name}-embed-eval",
        "source_exp"  : exp_name,
        "dataset"     : "mnist",
        "latent_dim"  : latent_dim,
        "probe_epochs": PROBE_EPOCHS,
        "probe_lr"    : PROBE_LR,
        "linear_probe": round(probe, 6),
        **{k: round(v, 6) for k, v in knn_res.items()},
        "time_s"      : round(time.perf_counter() - t0, 1),
    }
    # append to results.jsonl (skip if already present)
    JSONL = PROJECT / "results.jsonl"
    existing = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try: existing.add(json.loads(line).get("experiment"))
            except: pass
    if row["experiment"] not in existing:
        with open(JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")

    return probe, knn_res

# ── Run both models ────────────────────────────────────────────────────────────

from experiments14     import encode as encode14,     LATENT_DIM as LATENT14
from experiments14_abl import encode as encode14_abl, LATENT_DIM as LATENT14_abl

results = {}
results["exp14"]     = eval_model("exp14",     encode14,     LATENT14)
results["exp14_abl"] = eval_model("exp14_abl", encode14_abl, LATENT14_abl)

# ── Print comparison ───────────────────────────────────────────────────────────

print("\n" + "="*72)
print("exp14 encoder ablation — MNIST embedding eval (LATENT_DIM=64)")
print(f"{'':30s}  {'exp14 (full)':>14}  {'exp14_abl (recon only)':>22}")
print("="*72)

probe14, knn14     = results["exp14"]
probe_abl, knn_abl = results["exp14_abl"]

print(f"  {'Linear probe (30 ep Adam)':28s}  {probe14:>14.1%}  {probe_abl:>22.1%}")
print()
for k in KS:
    l14   = knn14[f"knn_l2_k{k}"]
    la    = knn_abl[f"knn_l2_k{k}"]
    print(f"  {f'L2-kNN  k={k}':28s}  {l14:>14.1%}  {la:>22.1%}")
print()
for k in KS:
    c14   = knn14[f"knn_cos_k{k}"]
    ca    = knn_abl[f"knn_cos_k{k}"]
    print(f"  {f'cos-kNN k={k}':28s}  {c14:>14.1%}  {ca:>22.1%}")

print("="*72)
delta_probe = probe14 - probe_abl
print(f"\n  Linear probe delta (full − abl):  {delta_probe:+.1%}")
delta_k5    = knn14["knn_l2_k5"] - knn_abl["knn_l2_k5"]
print(f"  5-NN L2 delta:                    {delta_k5:+.1%}")
