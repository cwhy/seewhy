"""
knn_eval.py — KNN evaluation on frozen SSL embeddings (MNIST + Fashion-MNIST).

For each trained model, extracts embeddings, builds a FAISS index on the
training set, and evaluates k-NN accuracy (k=1,5,10,20) on the test set.
Compares against the linear probe accuracy already logged in results.jsonl.

Usage:
    uv run python projects/ssl/scripts/knn_eval.py
"""

import importlib.util, pickle, json, sys, time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import faiss
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image

SSL    = Path(__file__).parent.parent
JSONL  = SSL / "results_knn.jsonl"
ROOT   = Path(__file__).parent.parent.parent.parent

KS = [1, 5, 10, 20]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_encode(exp_file):
    """Dynamically import encode() and LATENT_DIM from an experiment file."""
    path = SSL / exp_file
    spec = importlib.util.spec_from_file_location("_exp", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.encode, mod.LATENT_DIM

def load_params(exp_name):
    p = pickle.load(open(SSL / f"params_{exp_name}.pkl", "rb"))
    return {k: jnp.array(v) for k, v in p.items()}

def extract_embeddings(encode_fn, params, X, batch=512):
    parts = []
    for i in range(0, len(X), batch):
        xb = jnp.array(np.array(X[i:i+batch]))
        parts.append(np.array(encode_fn(params, xb)))
    return np.concatenate(parts, 0)

def knn_accuracy(E_tr, Y_tr, E_te, Y_te, k):
    """L2 KNN using FAISS."""
    dim = E_tr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(E_tr.astype(np.float32))
    _, I = index.search(E_te.astype(np.float32), k)
    # majority vote
    preds = np.array([np.bincount(Y_tr[I[i]], minlength=10).argmax() for i in range(len(E_te))])
    return float((preds == Y_te).mean())

def knn_accuracy_cosine(E_tr, Y_tr, E_te, Y_te, k):
    """Cosine KNN using FAISS (normalise first → L2 = cosine)."""
    def norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + 1e-8)
    return knn_accuracy(norm(E_tr), Y_tr, norm(E_te), Y_te, k)

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

def linear_probe_acc(exp_name, results_jsonl=SSL / "results.jsonl"):
    """Look up already-computed linear probe accuracy."""
    with open(results_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except: continue
            if r.get("experiment") == exp_name:
                return r.get("final_probe_acc")
    return None

# ── Experiment registry ───────────────────────────────────────────────────────
# (dataset_key, display_name, exp_name, exp_file)

EXPERIMENTS = [
    # ── MNIST LATENT=1024 ─────────────────────────────────────────────────────
    ("mnist",         "MLP-shallow",  "exp_mlp_ema_1024",        "exp_mlp_ema_1024.py"),
    ("mnist",         "MLP-deep",     "exp_mlp2_ema_1024",       "exp_mlp2_ema_1024.py"),
    ("mnist",         "ConvNet-v1",   "exp_conv_ema_1024",       "exp_conv_ema_1024.py"),
    ("mnist",         "ConvNet-v2",   "exp_conv2_ema_1024",      "exp_conv2_ema_1024.py"),
    ("mnist",         "ViT",          "exp_vit_ema_1024",        "exp_vit_ema_1024.py"),
    ("mnist",         "ViT-patch",    "exp_vit_patch_ema",       "exp_vit_patch_ema.py"),
    ("mnist",         "Gram-dual",    "exp_ema_1024",            "exp_ema_1024.py"),
    ("mnist",         "Gram-single",  "exp_gram_single_ema_1024","exp_gram_single_ema_1024.py"),
    ("mnist",         "Gram-GLU",     "exp_gram_glu_ema",        "exp_gram_glu_ema.py"),
    # ── Fashion-MNIST LATENT=1024 ─────────────────────────────────────────────
    ("fashion_mnist", "MLP-shallow",  "fmnist_exp_mlp_ema",           "fmnist_exp_mlp_ema.py"),
    ("fashion_mnist", "MLP-deep",     "fmnist_exp_mlp2_ema",          "fmnist_exp_mlp2_ema.py"),
    ("fashion_mnist", "ConvNet-v1",   "fmnist_exp_conv_ema",          "fmnist_exp_conv_ema.py"),
    ("fashion_mnist", "ConvNet-v2",   "fmnist_exp_conv2_ema",         "fmnist_exp_conv2_ema.py"),
    ("fashion_mnist", "ViT",          "fmnist_exp_vit_ema",           "fmnist_exp_vit_ema.py"),
    ("fashion_mnist", "ViT-patch",    "fmnist_exp_vit_patch_ema",     "fmnist_exp_vit_patch_ema.py"),
    ("fashion_mnist", "Gram-dual",    "fmnist_exp_ema",               "fmnist_exp_ema.py"),
    ("fashion_mnist", "Gram-single",  "fmnist_exp_gram_single_ema",   "fmnist_exp_gram_single_ema.py"),
    ("fashion_mnist", "Gram-GLU",     "fmnist_exp_gram_glu_ema",      "fmnist_exp_gram_glu_ema.py"),
]

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info(f"FAISS version: {faiss.__version__}")

    # Cache datasets
    datasets = {}
    for ds_key in ("mnist", "fashion_mnist"):
        d = load_supervised_image(ds_key)
        datasets[ds_key] = {
            "X_tr": d.X.reshape(d.n_samples, 784).astype(np.float32),
            "Y_tr": np.array(d.y, dtype=np.int32),
            "X_te": d.X_test.reshape(d.n_test_samples, 784).astype(np.float32),
            "Y_te": np.array(d.y_test, dtype=np.int32),
        }
        logging.info(f"Loaded {ds_key}: {d.n_samples} train, {d.n_test_samples} test")

    # Load already-computed exp_names to allow incremental runs
    done = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                    done.add((r.get("dataset"), r.get("exp_name")))
                except: pass
    logging.info(f"Already done: {len(done)} experiments")

    all_rows = []
    current_ds = None
    current_embeddings = {}  # cache embeddings per dataset per exp_name

    for ds_key, display, exp_name, exp_file in EXPERIMENTS:
        if (ds_key, exp_name) in done:
            logging.info(f"[{ds_key}] {display} ({exp_name}) — SKIP (already in {JSONL.name})")
            continue
        logging.info(f"\n[{ds_key}] {display} ({exp_name})")
        t0 = time.perf_counter()

        try:
            encode_fn, latent_dim = load_encode(exp_file)
            params = load_params(exp_name)
        except Exception as e:
            logging.warning(f"  SKIP — could not load: {e}")
            continue

        ds = datasets[ds_key]
        logging.info(f"  Extracting embeddings (LATENT={latent_dim})...")
        E_tr = extract_embeddings(encode_fn, params, ds["X_tr"])
        E_te = extract_embeddings(encode_fn, params, ds["X_te"])
        embed_time = time.perf_counter() - t0
        logging.info(f"  Embeddings: train {E_tr.shape}, test {E_te.shape}  ({embed_time:.1f}s)")

        row = {
            "dataset"    : ds_key,
            "encoder"    : display,
            "exp_name"   : exp_name,
            "latent_dim" : latent_dim,
            "linear_probe": linear_probe_acc(exp_name),
        }

        # L2 KNN
        logging.info(f"  Running L2 KNN (k={KS})...")
        for k in KS:
            acc = knn_accuracy(E_tr, ds["Y_tr"], E_te, ds["Y_te"], k)
            row[f"knn_l2_k{k}"] = round(acc, 6)
            logging.info(f"    k={k:2d}  L2-KNN = {acc:.1%}")

        # Cosine KNN
        logging.info(f"  Running cosine KNN (k={KS})...")
        for k in KS:
            acc = knn_accuracy_cosine(E_tr, ds["Y_tr"], E_te, ds["Y_te"], k)
            row[f"knn_cos_k{k}"] = round(acc, 6)
            logging.info(f"    k={k:2d}  cos-KNN = {acc:.1%}")

        row["time_s"] = round(time.perf_counter() - t0, 1)
        append_result(row)
        all_rows.append(row)
        logging.info(f"  Done in {row['time_s']}s")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print(f"{'Dataset':<14} {'Encoder':<14} {'Linear':>7} {'1-NN(L2)':>9} {'5-NN(L2)':>9} {'10-NN':>7} {'5-NN(cos)':>10}")
    print("="*90)
    for r in all_rows:
        lp  = f"{r['linear_probe']:.1%}" if r.get("linear_probe") else "—"
        k1  = f"{r.get('knn_l2_k1', 0):.1%}"
        k5  = f"{r.get('knn_l2_k5', 0):.1%}"
        k10 = f"{r.get('knn_l2_k10', 0):.1%}"
        c5  = f"{r.get('knn_cos_k5', 0):.1%}"
        print(f"{r['dataset']:<14} {r['encoder']:<14} {lp:>7} {k1:>9} {k5:>9} {k10:>7} {c5:>10}")
