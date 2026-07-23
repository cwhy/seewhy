"""
eval_embed_exp14.py — Linear probe + kNN classification on exp14 encoder embeddings.

Evaluates how well the encoder from the variations exp14 model (trained with
inverted-IMLE on MNIST, no class labels) captures class-discriminative information.

Metrics:
  - Linear probe accuracy (JAX, Adam, 30 epochs, frozen encoder)
  - kNN accuracy (FAISS, L2 + cosine, k=1,5,10,20)

Usage:
    uv run python scripts/tmp/eval_embed_exp14.py
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
from experiments14 import encode, LATENT_DIM

EXP_NAME    = "exp14"
PROBE_LR    = 1e-3
PROBE_EPOCHS = 30
PROBE_BATCH  = 256
KS          = [1, 5, 10, 20]
N_CLASSES   = 10

# ── Load params ────────────────────────────────────────────────────────────────

with open(PROJECT / f"params_{EXP_NAME}.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

logging.info(f"Loaded params for {EXP_NAME}. LATENT_DIM={LATENT_DIM}")

# ── Load MNIST ─────────────────────────────────────────────────────────────────

data    = load_supervised_image("mnist")
X_train = np.array(data.X.reshape(data.n_samples, 784).astype(np.float32))
Y_train = np.array(data.y, dtype=np.int32)
X_test  = np.array(data.X_test.reshape(data.n_test_samples, 784).astype(np.float32))
Y_test  = np.array(data.y_test, dtype=np.int32)
logging.info(f"MNIST: {len(X_train)} train, {len(X_test)} test")

# ── Extract embeddings ─────────────────────────────────────────────────────────

def extract_embeddings(X, batch=512):
    parts = []
    for i in range(0, len(X), batch):
        xb = jnp.array(X[i:i+batch] / 255.0)
        parts.append(np.array(encode(params, xb)))
    return np.concatenate(parts, 0).astype(np.float32)

t0 = time.perf_counter()
E_train = extract_embeddings(X_train)
E_test  = extract_embeddings(X_test)
logging.info(f"Embeddings: train {E_train.shape}, test {E_test.shape}  ({time.perf_counter()-t0:.1f}s)")

# ── Linear probe (JAX Adam, frozen encoder) ────────────────────────────────────

n_fit = (len(E_train) // PROBE_BATCH) * PROBE_BATCH
E_b   = jnp.array(E_train[:n_fit].reshape(-1, PROBE_BATCH, LATENT_DIM))
Y_b   = jnp.array(Y_train[:n_fit].reshape(-1, PROBE_BATCH))

W, b        = jnp.zeros((LATENT_DIM, N_CLASSES)), jnp.zeros(N_CLASSES)
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

t1 = time.perf_counter()
for ep in range(PROBE_EPOCHS):
    W, b, probe_state = probe_epoch(W, b, probe_state)

logits      = jnp.array(E_test) @ W + b
probe_acc   = float((jnp.argmax(logits, 1) == jnp.array(Y_test)).mean())
probe_time  = time.perf_counter() - t1
logging.info(f"Linear probe: {probe_acc:.1%}  ({probe_time:.1f}s, {PROBE_EPOCHS} epochs)")

# ── kNN (FAISS, L2 and cosine) ─────────────────────────────────────────────────

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

t2 = time.perf_counter()
knn_results = {}
for k in KS:
    knn_results[f"knn_l2_k{k}"]  = knn_l2(E_train, Y_train, E_test, Y_test, k)
    knn_results[f"knn_cos_k{k}"] = knn_cos(E_train, Y_train, E_test, Y_test, k)
logging.info(f"kNN done ({time.perf_counter()-t2:.1f}s)")

# ── Results ────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print(f"exp14 encoder embedding eval — MNIST (LATENT_DIM={LATENT_DIM})")
print("="*60)
print(f"  Linear probe ({PROBE_EPOCHS} epochs Adam):  {probe_acc:.1%}")
print()
print(f"  {'k':>4}  {'L2-kNN':>8}  {'cos-kNN':>8}")
for k in KS:
    l2  = knn_results[f"knn_l2_k{k}"]
    cos = knn_results[f"knn_cos_k{k}"]
    print(f"  {k:>4}  {l2:>8.1%}  {cos:>8.1%}")
print("="*60)

# ── Save to JSONL ──────────────────────────────────────────────────────────────

result = {
    "experiment"    : f"{EXP_NAME}-embed-eval",
    "source_exp"    : EXP_NAME,
    "dataset"       : "mnist",
    "latent_dim"    : LATENT_DIM,
    "probe_epochs"  : PROBE_EPOCHS,
    "probe_lr"      : PROBE_LR,
    "linear_probe"  : round(probe_acc, 6),
    **{k: round(v, 6) for k, v in knn_results.items()},
    "time_s"        : round(time.perf_counter() - t0, 1),
}

JSONL = PROJECT / "results.jsonl"
with open(JSONL, "a") as f:
    f.write(json.dumps(result) + "\n")

print(f"\nSaved to {JSONL}")
