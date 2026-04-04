"""
kmeans_comparison.py — Compare three training regimes on K-Means cluster accuracy.

For each encoder architecture runs K-Means (K=128) on:
  random  — randomly-initialised encoder (untrained), same architecture
  sigreg  — SIGReg-trained (all 9 architectures)
  ema     — EMA-JEPA-trained (all 9 architectures)

Results are appended to results_kmeans_comparison.jsonl.
A summary table is printed at the end.

Usage:
    uv run python projects/ssl/scripts/kmeans_comparison.py
    uv run python projects/ssl/scripts/kmeans_comparison.py --k 64 128 --dataset mnist
    uv run python projects/ssl/scripts/kmeans_comparison.py --force
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
from shared_lib.random_utils import infinite_safe_keys_from_key

SSL   = Path(__file__).parent.parent
JSONL = SSL / "results_kmeans_comparison.jsonl"

# ── Encoder registry ──────────────────────────────────────────────────────────
# (display_name, ema_file, ema_exp_name, sigreg_file, sigreg_exp_name or None)
REGISTRY = [
    ("MLP-shallow", "exp_mlp_ema_1024.py",         "exp_mlp_ema_1024",         "exp_sigreg_mlp.py",          "exp_sigreg_mlp"),
    ("MLP-deep",    "exp_mlp2_ema_1024.py",        "exp_mlp2_ema_1024",        "exp_sigreg_mlp2.py",         "exp_sigreg_mlp2"),
    ("ConvNet-v1",  "exp_conv_ema_1024.py",        "exp_conv_ema_1024",        "exp_sigreg_conv.py",         "exp_sigreg_conv"),
    ("ConvNet-v2",  "exp_conv2_ema_1024.py",       "exp_conv2_ema_1024",       "exp_sigreg_conv2.py",        "exp_sigreg_conv2"),
    ("ViT",         "exp_vit_ema_1024.py",         "exp_vit_ema_1024",         "exp_sigreg_vit.py",          "exp_sigreg_vit"),
    ("ViT-patch",   "exp_vit_patch_ema.py",        "exp_vit_patch_ema",        "exp_sigreg_vit_patch.py",    "exp_sigreg_vit_patch"),
    ("Gram-dual",   "exp_ema_1024.py",             "exp_ema_1024",             "exp12a.py",                  "exp12a"),
    ("Gram-single", "exp_gram_single_ema_1024.py", "exp_gram_single_ema_1024", "exp_sigreg_gram_single.py",  "exp_sigreg_gram_single"),
    ("Gram-GLU",    "exp_gram_glu_ema.py",         "exp_gram_glu_ema",         "exp_sigreg_gram_glu.py",     "exp_sigreg_gram_glu"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_module(exp_file):
    path = SSL / exp_file
    spec = importlib.util.spec_from_file_location("_exp", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_trained_params(exp_name):
    p = pickle.load(open(SSL / f"params_{exp_name}.pkl", "rb"))
    return {k: jnp.array(v) for k, v in p.items()}

def random_params(mod, seed=0):
    """Initialise encoder with random weights (no training)."""
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(seed))
    return mod.init_params(key_gen)

def extract_embeddings(encode_fn, params, X, batch=512):
    parts = []
    for i in range(0, len(X), batch):
        xb = jnp.array(np.array(X[i:i+batch]))
        parts.append(np.array(encode_fn(params, xb)))
    return np.concatenate(parts, 0).astype(np.float32)

def kmeans_cluster_acc(E_tr, Y_tr, E_te, Y_te, k, n_classes=10):
    dim = E_tr.shape[1]
    km = faiss.Kmeans(dim, k, niter=100, verbose=False, seed=42, gpu=False)
    km.train(E_tr)

    _, I_tr = km.index.search(E_tr, 1)
    cluster_ids_tr = I_tr[:, 0]

    cluster_to_class = np.zeros(k, dtype=np.int32)
    cluster_purity   = np.zeros(k, dtype=np.float32)
    for c in range(k):
        mask = cluster_ids_tr == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(Y_tr[mask], minlength=n_classes)
        cluster_to_class[c] = counts.argmax()
        cluster_purity[c]   = counts.max() / mask.sum()

    preds_tr  = cluster_to_class[cluster_ids_tr]
    train_acc = float((preds_tr == Y_tr).mean())

    _, I_te = km.index.search(E_te, 1)
    preds_te = cluster_to_class[I_te[:, 0]]
    test_acc = float((preds_te == Y_te).mean())

    mean_purity = float(cluster_purity[cluster_purity > 0].mean()) if (cluster_purity > 0).any() else 0.0
    return train_acc, test_acc, mean_purity

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

def result_key(r):
    return (r.get("dataset"), r.get("encoder"), r.get("training"), r.get("k"))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int, nargs="+", default=[128])
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    logging.info(f"JAX devices : {jax.devices()}")
    logging.info(f"Dataset     : {args.dataset}")
    logging.info(f"K values    : {args.k}")

    d = load_supervised_image(args.dataset)
    ds = {
        "X_tr": d.X.reshape(d.n_samples,           784).astype(np.float32),
        "Y_tr": np.array(d.y,       dtype=np.int32),
        "X_te": d.X_test.reshape(d.n_test_samples, 784).astype(np.float32),
        "Y_te": np.array(d.y_test,  dtype=np.int32),
    }
    n_classes = len(np.unique(ds["Y_tr"]))
    logging.info(f"Loaded {args.dataset}: {d.n_samples} train / {d.n_test_samples} test")

    # Load already-done keys
    done = set()
    if JSONL.exists() and not args.force:
        with open(JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(result_key(json.loads(line)))
                    except Exception:
                        pass
    logging.info(f"Already done: {len(done)} rows")

    all_rows = []

    for display, ema_file, ema_exp, sigreg_file, sigreg_exp in REGISTRY:
        logging.info(f"\n{'='*60}\n{display}\n{'='*60}")

        try:
            ema_mod = load_module(ema_file)
        except Exception as e:
            logging.warning(f"  Could not load EMA module {ema_file}: {e}")
            continue

        # Build list of (training_label, params_or_fn)
        candidates = []

        # 1 — Random baseline (always available)
        candidates.append(("random", lambda mod=ema_mod: random_params(mod)))

        # 2 — SIGReg (Gram-dual only)
        if sigreg_file and sigreg_exp:
            try:
                sigreg_mod    = load_module(sigreg_file)
                sigreg_params = load_trained_params(sigreg_exp)
                candidates.append(("sigreg", lambda p=sigreg_params: p))
                # Store the sigreg encode function alongside
                candidates[-1] = ("sigreg", lambda p=sigreg_params: p, sigreg_mod.encode, sigreg_mod.LATENT_DIM)
            except Exception as e:
                logging.warning(f"  SIGReg params not found for {sigreg_exp}: {e}")

        # 3 — EMA
        try:
            ema_params = load_trained_params(ema_exp)
            candidates.append(("ema", lambda p=ema_params: p))
        except Exception as e:
            logging.warning(f"  EMA params not found for {ema_exp}: {e}")

        for entry in candidates:
            if len(entry) == 2:
                training, params_fn = entry
                encode_fn  = ema_mod.encode
                latent_dim = ema_mod.LATENT_DIM
            else:
                training, params_fn, encode_fn, latent_dim = entry

            params = params_fn()

            for k in args.k:
                key = (args.dataset, display, training, k)
                if key in done:
                    logging.info(f"  [{training}] k={k} — skip")
                    continue

                logging.info(f"  [{training}] k={k}  extracting embeddings (LATENT={latent_dim})...")
                t0 = time.perf_counter()
                E_tr = extract_embeddings(encode_fn, params, ds["X_tr"])
                E_te = extract_embeddings(encode_fn, params, ds["X_te"])
                logging.info(f"    embeddings: {E_tr.shape}  ({time.perf_counter()-t0:.1f}s)")

                logging.info(f"    fitting k-means k={k}...")
                tr_acc, te_acc, purity = kmeans_cluster_acc(
                    E_tr, ds["Y_tr"], E_te, ds["Y_te"], k, n_classes
                )
                elapsed = time.perf_counter() - t0
                logging.info(f"    train={tr_acc:.1%}  test={te_acc:.1%}  purity={purity:.1%}  ({elapsed:.1f}s)")

                row = {
                    "dataset"    : args.dataset,
                    "encoder"    : display,
                    "training"   : training,
                    "k"          : k,
                    "latent_dim" : latent_dim,
                    "train_acc"  : round(tr_acc,  6),
                    "test_acc"   : round(te_acc,  6),
                    "mean_purity": round(purity,  6),
                    "time_s"     : round(elapsed, 1),
                }
                append_result(row)
                all_rows.append(row)
                done.add(key)

    # ── Load all rows for summary (always reload from JSONL) ─────────────────
    all_rows = []
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("dataset") == args.dataset:
                            all_rows.append(r)
                    except Exception:
                        pass

    # ── Summary table ─────────────────────────────────────────────────────────
    for k in sorted(set(r["k"] for r in all_rows)):
        rows_k = [r for r in all_rows if r["k"] == k]

        # Build per-encoder dict
        by_enc = {}
        for r in rows_k:
            by_enc.setdefault(r["encoder"], {})[r["training"]] = r

        training_methods = ["random", "sigreg", "ema"]

        print()
        print("=" * 74)
        print(f"  Dataset: {args.dataset:<16}  K={k}   (test accuracy)")
        print("=" * 74)
        hdr = f"  {'Encoder':<14}"
        for t in training_methods:
            hdr += f"  {t.upper():>8}"
        hdr += f"  {'EMA-rand':>9}  {'EMA-sig':>8}"
        print(hdr)
        print("  " + "-" * 56)

        for display, *_ in REGISTRY:
            d = by_enc.get(display, {})
            rand_acc = d.get("random",  {}).get("test_acc")
            sig_acc  = d.get("sigreg",  {}).get("test_acc")
            ema_acc  = d.get("ema",     {}).get("test_acc")

            def fmt(v): return f"{v:.1%}" if v is not None else "   —  "

            ema_rand_delta = f"+{(ema_acc - rand_acc)*100:.1f}pp" if (ema_acc is not None and rand_acc is not None) else "   —  "
            ema_sig_delta  = f"+{(ema_acc - sig_acc)*100:.1f}pp"  if (ema_acc is not None and sig_acc  is not None) else "   —  "

            print(f"  {display:<14}  {fmt(rand_acc):>8}  {fmt(sig_acc):>8}  {fmt(ema_acc):>8}  {ema_rand_delta:>9}  {ema_sig_delta:>8}")

        print()


if __name__ == "__main__":
    main()
