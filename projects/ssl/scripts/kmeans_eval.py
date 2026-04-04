"""
kmeans_eval.py — K-Means clustering evaluation on frozen SSL embeddings.

Fits K-Means (FAISS) on training embeddings, then assigns each cluster to its
majority class in the training set and evaluates assignment accuracy on the
test set.  Mirrors knn_eval.py — same encoder registry, appends to a
dedicated results_kmeans.jsonl.

Options
-------
--k K [K ...]   List of K values to sweep  (default: 128)
--proj DIM       Project embeddings to DIM via a fixed random Gaussian matrix
                 before clustering.  DIM=0 disables projection (default).
--datasets       Comma-separated list: mnist, fashion_mnist  (default: both)
--force          Re-run encoders that already have an entry in the JSONL

Random projection details
-------------------------
A (LATENT_DIM × DIM) Gaussian matrix is drawn once per encoder (seed=0) and
normalised so each column has unit L2 norm.  The projection compresses 1024-d
embeddings to DIM-d before FAISS k-means, which speeds up clustering and lets
us test whether lower-dimensional structure is sufficient for good clusters.

Usage:
    uv run python projects/ssl/scripts/kmeans_eval.py
    uv run python projects/ssl/scripts/kmeans_eval.py --k 10 64 128 --proj 256
    uv run python projects/ssl/scripts/kmeans_eval.py --k 128 --proj 0 256 512
"""

import argparse
import importlib.util
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image

SSL   = Path(__file__).parent.parent
JSONL = SSL / "results_kmeans.jsonl"
ROOT  = Path(__file__).parent.parent.parent.parent

# ── Encoder registry ──────────────────────────────────────────────────────────
# (dataset_key, display_name, exp_name, exp_file)
EXPERIMENTS = [
    # ── MNIST ─────────────────────────────────────────────────────────────────
    ("mnist", "MLP-shallow",  "exp_mlp_ema_1024",         "exp_mlp_ema_1024.py"),
    ("mnist", "MLP-deep",     "exp_mlp2_ema_1024",        "exp_mlp2_ema_1024.py"),
    ("mnist", "ConvNet-v1",   "exp_conv_ema_1024",        "exp_conv_ema_1024.py"),
    ("mnist", "ConvNet-v2",   "exp_conv2_ema_1024",       "exp_conv2_ema_1024.py"),
    ("mnist", "ViT",          "exp_vit_ema_1024",         "exp_vit_ema_1024.py"),
    ("mnist", "ViT-patch",    "exp_vit_patch_ema",        "exp_vit_patch_ema.py"),
    ("mnist", "Gram-dual",    "exp_ema_1024",             "exp_ema_1024.py"),
    ("mnist", "Gram-single",  "exp_gram_single_ema_1024", "exp_gram_single_ema_1024.py"),
    ("mnist", "Gram-GLU",     "exp_gram_glu_ema",         "exp_gram_glu_ema.py"),
    # ── Fashion-MNIST ─────────────────────────────────────────────────────────
    ("fashion_mnist", "MLP-shallow",  "fmnist_exp_mlp_ema",          "fmnist_exp_mlp_ema.py"),
    ("fashion_mnist", "MLP-deep",     "fmnist_exp_mlp2_ema",         "fmnist_exp_mlp2_ema.py"),
    ("fashion_mnist", "ConvNet-v1",   "fmnist_exp_conv_ema",         "fmnist_exp_conv_ema.py"),
    ("fashion_mnist", "ConvNet-v2",   "fmnist_exp_conv2_ema",        "fmnist_exp_conv2_ema.py"),
    ("fashion_mnist", "ViT",          "fmnist_exp_vit_ema",          "fmnist_exp_vit_ema.py"),
    ("fashion_mnist", "ViT-patch",    "fmnist_exp_vit_patch_ema",    "fmnist_exp_vit_patch_ema.py"),
    ("fashion_mnist", "Gram-dual",    "fmnist_exp_ema",              "fmnist_exp_ema.py"),
    ("fashion_mnist", "Gram-single",  "fmnist_exp_gram_single_ema",  "fmnist_exp_gram_single_ema.py"),
    ("fashion_mnist", "Gram-GLU",     "fmnist_exp_gram_glu_ema",     "fmnist_exp_gram_glu_ema.py"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_encode(exp_file):
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
    return np.concatenate(parts, 0).astype(np.float32)

def make_random_projection(latent_dim, proj_dim, seed=0):
    """Return a (latent_dim × proj_dim) random Gaussian projection matrix.
    Columns are L2-normalised so projected norms are preserved in expectation."""
    rng = np.random.RandomState(seed)
    R = rng.randn(latent_dim, proj_dim).astype(np.float32)
    R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-8
    return R

def apply_projection(E, R):
    """Project embedding matrix E (N, D) → (N, proj_dim) using matrix R."""
    return E @ R

def kmeans_cluster_acc(E_tr, Y_tr, E_te, Y_te, k, n_classes=10):
    """
    1. Fit FAISS k-means on E_tr.
    2. Assign each cluster to its majority class in Y_tr.
    3. Evaluate cluster-assignment accuracy on E_te.

    Returns (train_acc, test_acc, cluster_purities_mean).
    """
    dim = E_tr.shape[1]

    # ── Fit k-means ──────────────────────────────────────────────────────────
    km = faiss.Kmeans(dim, k, niter=100, verbose=False, seed=42, gpu=False)
    km.train(E_tr)

    # ── Assign training set to clusters ──────────────────────────────────────
    _, I_tr = km.index.search(E_tr, 1)
    cluster_ids_tr = I_tr[:, 0]  # (N_tr,)

    # ── Majority-vote: map each cluster → class ───────────────────────────────
    cluster_to_class = np.zeros(k, dtype=np.int32)
    cluster_purity   = np.zeros(k, dtype=np.float32)
    for c in range(k):
        mask = cluster_ids_tr == c
        if mask.sum() == 0:
            cluster_to_class[c] = 0
            cluster_purity[c]   = 0.0
            continue
        counts = np.bincount(Y_tr[mask], minlength=n_classes)
        cluster_to_class[c] = counts.argmax()
        cluster_purity[c]   = counts.max() / mask.sum()

    # ── Train accuracy ────────────────────────────────────────────────────────
    preds_tr = cluster_to_class[cluster_ids_tr]
    train_acc = float((preds_tr == Y_tr).mean())

    # ── Test accuracy ─────────────────────────────────────────────────────────
    _, I_te = km.index.search(E_te, 1)
    cluster_ids_te = I_te[:, 0]
    preds_te = cluster_to_class[cluster_ids_te]
    test_acc = float((preds_te == Y_te).mean())

    mean_purity = float(cluster_purity[cluster_purity > 0].mean()) if (cluster_purity > 0).any() else 0.0
    return train_acc, test_acc, mean_purity

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

def result_key(row):
    return (row.get("dataset"), row.get("exp_name"), row.get("k"), row.get("proj_dim", 0))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",        type=int, nargs="+", default=[128],
                        help="K values for k-means (default: 128)")
    parser.add_argument("--proj",     type=int, nargs="+", default=[0],
                        help="Random projection dims before clustering; 0 = no projection (default: 0)")
    parser.add_argument("--datasets", type=str, default="mnist,fashion_mnist",
                        help="Comma-separated datasets to evaluate")
    parser.add_argument("--force",    action="store_true",
                        help="Re-run experiments already in the JSONL")
    args = parser.parse_args()

    active_datasets = set(args.datasets.split(","))
    k_values  = sorted(set(args.k))
    proj_dims = sorted(set(args.proj))

    logging.info(f"JAX devices : {jax.devices()}")
    logging.info(f"FAISS build : {faiss.__version__}")
    logging.info(f"K values    : {k_values}")
    logging.info(f"Proj dims   : {proj_dims}  (0 = no projection)")
    logging.info(f"Datasets    : {sorted(active_datasets)}")

    # Load datasets
    datasets = {}
    for ds_key in ("mnist", "fashion_mnist"):
        if ds_key not in active_datasets:
            continue
        d = load_supervised_image(ds_key)
        datasets[ds_key] = {
            "X_tr": d.X.reshape(d.n_samples,           784).astype(np.float32),
            "Y_tr": np.array(d.y,       dtype=np.int32),
            "X_te": d.X_test.reshape(d.n_test_samples, 784).astype(np.float32),
            "Y_te": np.array(d.y_test,  dtype=np.int32),
        }
        n_classes = len(np.unique(datasets[ds_key]["Y_tr"]))
        logging.info(f"Loaded {ds_key}: {d.n_samples} train / {d.n_test_samples} test  ({n_classes} classes)")

    # Load already-done result keys
    done = set()
    if JSONL.exists() and not args.force:
        with open(JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add(result_key(r))
                except Exception:
                    pass
    logging.info(f"Already done: {len(done)} result rows")

    all_rows = []

    for ds_key, display, exp_name, exp_file in EXPERIMENTS:
        if ds_key not in active_datasets:
            continue

        # Check if we need any (k, proj) combinations for this encoder
        needed = [(k, pd) for k in k_values for pd in proj_dims
                  if (ds_key, exp_name, k, pd) not in done]
        if not needed:
            logging.info(f"[{ds_key}] {display} — all combos done, skip")
            continue

        logging.info(f"\n[{ds_key}] {display} ({exp_name})")
        t_enc = time.perf_counter()

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
        logging.info(f"  Embeddings: train {E_tr.shape}  test {E_te.shape}  ({time.perf_counter()-t_enc:.1f}s)")

        for proj_dim in proj_dims:
            if proj_dim > 0:
                R = make_random_projection(latent_dim, proj_dim, seed=0)
                E_tr_use = apply_projection(E_tr, R)
                E_te_use = apply_projection(E_te, R)
                proj_label = f"proj{proj_dim}"
            else:
                E_tr_use = E_tr
                E_te_use = E_te
                proj_label = "full"

            for k in k_values:
                if (ds_key, exp_name, k, proj_dim) in done:
                    logging.info(f"  k={k} {proj_label} — skip")
                    continue

                logging.info(f"  k={k}  {proj_label} ({E_tr_use.shape[1]}-d) → fitting k-means...")
                t0 = time.perf_counter()
                n_classes = len(np.unique(ds["Y_tr"]))
                tr_acc, te_acc, purity = kmeans_cluster_acc(
                    E_tr_use, ds["Y_tr"], E_te_use, ds["Y_te"], k, n_classes
                )
                elapsed = time.perf_counter() - t0
                logging.info(f"    train_acc={tr_acc:.1%}  test_acc={te_acc:.1%}  "
                             f"mean_purity={purity:.1%}  ({elapsed:.1f}s)")

                row = {
                    "dataset"     : ds_key,
                    "encoder"     : display,
                    "exp_name"    : exp_name,
                    "latent_dim"  : latent_dim,
                    "k"           : k,
                    "proj_dim"    : proj_dim,
                    "train_acc"   : round(tr_acc,   6),
                    "test_acc"    : round(te_acc,   6),
                    "mean_purity" : round(purity,   6),
                    "time_s"      : round(elapsed,  1),
                }
                append_result(row)
                all_rows.append(row)
                done.add(result_key(row))

    # ── Print comparison table ────────────────────────────────────────────────
    if not all_rows:
        # Load all rows from JSONL for the summary table
        if JSONL.exists():
            with open(JSONL) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_rows.append(json.loads(line))
                        except Exception:
                            pass

    # Group by (dataset, k, proj_dim)
    from itertools import groupby

    def group_key(r):
        return (r.get("dataset"), r.get("k"), r.get("proj_dim", 0))

    all_rows.sort(key=group_key)

    print()
    for (ds, k, pd), rows in groupby(all_rows, key=group_key):
        rows = sorted(rows, key=lambda r: -r.get("test_acc", 0))
        proj_label = f"proj→{pd}" if pd > 0 else "full"
        print("=" * 72)
        print(f"  Dataset: {ds:<16}  K={k:<6}  Proj: {proj_label}")
        print("=" * 72)
        print(f"  {'Encoder':<16} {'Train':>7} {'Test':>7} {'Purity':>8}")
        print("  " + "-" * 42)
        for r in rows:
            print(f"  {r['encoder']:<16} {r['train_acc']:>7.1%} {r['test_acc']:>7.1%} {r['mean_purity']:>8.1%}")
        print()


if __name__ == "__main__":
    main()
