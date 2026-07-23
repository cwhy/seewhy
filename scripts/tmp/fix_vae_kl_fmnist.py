"""
fix_vae_kl_fmnist.py — Re-evaluate Fashion-MNIST probe for VAE KL sweep.

The original VAE KL experiments mistakenly trained the probe on MNIST labels
and tested on F-MNIST (label mismatch, ~10% = chance).

Correct approach (matching knn_comparison): train probe on F-MNIST train
embeddings (using MNIST-trained encoder), test on F-MNIST test embeddings.

Updates probe_acc_fmnist in-place in results_vae_kl.jsonl.
"""
import importlib.util, json, pickle, sys
from pathlib import Path

import jax, jax.numpy as jnp, optax, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_image

SSL   = Path(__file__).parent.parent.parent / "projects" / "ssl"
JSONL = SSL / "results_vae_kl.jsonl"

BATCH_SIZE   = 256
PROBE_EPOCHS = 10

def load_module(path):
    spec = importlib.util.spec_from_file_location("_m", path)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def extract_emb(encode_fn, params, X):
    parts = []
    for i in range(0, len(X), 512):
        parts.append(np.array(encode_fn(params, jnp.array(np.array(X[i:i+512])))))
    return np.concatenate(parts, 0).astype(np.float32)

def linear_probe(E_tr, Y_tr, E_te, Y_te):
    LATENT = E_tr.shape[1]
    n_tr   = (len(E_tr) // BATCH_SIZE) * BATCH_SIZE
    E_b    = jnp.array(E_tr[:n_tr].reshape(-1, BATCH_SIZE, LATENT))
    Y_b    = jnp.array(Y_tr[:n_tr].reshape(-1, BATCH_SIZE))
    W, b   = jnp.zeros((LATENT, 10)), jnp.zeros(10)
    opt    = optax.adam(1e-3)
    state  = opt.init((W, b))

    def step(carry, inputs):
        (W, b), s = carry
        e, y = inputs
        loss, g = jax.value_and_grad(
            lambda wb: optax.softmax_cross_entropy_with_integer_labels(
                e @ wb[0] + wb[1], y).mean()
        )((W, b))
        upd, s = opt.update(g, s, (W, b))
        return (optax.apply_updates((W, b), upd), s), loss

    @jax.jit
    def epoch(W, b, s, E_b, Y_b):
        (W, b), s = jax.lax.scan(step, ((W, b), s), (E_b, Y_b))[0]
        return W, b, s

    for _ in range(PROBE_EPOCHS):
        W, b, state = epoch(W, b, state, E_b, Y_b)
    logits = jnp.array(E_te) @ W + b
    return float((jnp.argmax(logits, 1) == jnp.array(Y_te)).mean())

# Load datasets
fmnist   = load_supervised_image("fashion_mnist")
X_tr_f   = fmnist.X.reshape(fmnist.n_samples,           784).astype(np.float32)
X_te_f   = fmnist.X_test.reshape(fmnist.n_test_samples, 784).astype(np.float32)
Y_tr_f   = np.array(fmnist.y,      dtype=np.int32)
Y_te_f   = np.array(fmnist.y_test, dtype=np.int32)

# Load existing results
rows = []
with open(JSONL) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

EXPS = [
    "exp_vae_kl_b0",
    "exp_vae_kl_b1e4",
    "exp_vae_kl_b1e3",
    "exp_vae_kl_b1e2",
    "exp_vae_kl_b1e1",
    "exp_vae_kl_b1",
]

updated = []
for r in rows:
    exp_name = r["experiment"]
    if exp_name not in EXPS:
        updated.append(r)
        continue

    pkl_path = SSL / f"params_{exp_name}.pkl"
    if not pkl_path.exists():
        print(f"  {exp_name}: no pkl, skipping")
        updated.append(r)
        continue

    mod    = load_module(SSL / f"{exp_name}.py")
    params = {k: jnp.array(v) for k, v in pickle.load(open(pkl_path, "rb")).items()}

    print(f"  {exp_name} (beta={r['beta']}): extracting F-MNIST embeddings...")
    E_tr_f = extract_emb(mod.encode, params, X_tr_f)
    E_te_f = extract_emb(mod.encode, params, X_te_f)
    acc = linear_probe(E_tr_f, Y_tr_f, E_te_f, Y_te_f)
    print(f"    => fmnist probe = {acc:.1%}  (was {r['probe_acc_fmnist']:.1%})")
    r["probe_acc_fmnist"] = round(acc, 6)
    updated.append(r)

with open(JSONL, "w") as f:
    for r in updated:
        f.write(json.dumps(r) + "\n")

print(f"\nUpdated {JSONL}")
for r in sorted(updated, key=lambda r: r.get("beta", 999)):
    if r.get("experiment", "").startswith("exp_vae_kl"):
        print(f"  beta={r['beta']}  mnist={r['probe_acc_mnist']:.1%}  fmnist={r['probe_acc_fmnist']:.1%}")
