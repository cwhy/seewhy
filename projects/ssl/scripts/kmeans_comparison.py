"""
kmeans_comparison.py — Compare six training regimes on K-Means cluster accuracy.

For each encoder architecture runs K-Means (K=128) on:
  random  — randomly-initialised encoder (untrained), same architecture
  ae      — standard autoencoder (MSE reconstruction)
  dae     — denoising autoencoder (same augmentations as EMA/SIGReg)
  vae     — variational autoencoder (MSE + KL, encode returns mean)
  sigreg  — SIGReg-JEPA (spectral regulariser)
  ema     — EMA-JEPA (momentum target encoder)

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
# Each entry: dict with "name" and one key per training regime -> (file, exp_name).
# "random" is always derived from the ema module's init_params.
TRAINING_ORDER = ["random", "ae", "dae", "vae", "sigreg", "ema"]

REGISTRY = [
    dict(name="MLP-shallow",
         ema=    ("exp_mlp_ema_1024.py",         "exp_mlp_ema_1024"),
         sigreg= ("exp_sigreg_mlp.py",            "exp_sigreg_mlp"),
         ae=     ("exp_ae_mlp.py",               "exp_ae_mlp"),
         dae=    ("exp_dae_mlp.py",              "exp_dae_mlp"),
         vae=    ("exp_vae_mlp.py",              "exp_vae_mlp"),
    ),
    dict(name="MLP-deep",
         ema=    ("exp_mlp2_ema_1024.py",        "exp_mlp2_ema_1024"),
         sigreg= ("exp_sigreg_mlp2.py",           "exp_sigreg_mlp2"),
         ae=     ("exp_ae_mlp2.py",              "exp_ae_mlp2"),
         dae=    ("exp_dae_mlp2.py",             "exp_dae_mlp2"),
         vae=    ("exp_vae_mlp2.py",             "exp_vae_mlp2"),
    ),
    dict(name="ConvNet-v1",
         ema=    ("exp_conv_ema_1024.py",        "exp_conv_ema_1024"),
         sigreg= ("exp_sigreg_conv.py",           "exp_sigreg_conv"),
         ae=     ("exp_ae_conv.py",              "exp_ae_conv"),
         dae=    ("exp_dae_conv.py",             "exp_dae_conv"),
         vae=    ("exp_vae_conv.py",             "exp_vae_conv"),
    ),
    dict(name="ConvNet-v2",
         ema=    ("exp_conv2_ema_1024.py",       "exp_conv2_ema_1024"),
         sigreg= ("exp_sigreg_conv2.py",          "exp_sigreg_conv2"),
         ae=     ("exp_ae_conv2.py",             "exp_ae_conv2"),
         dae=    ("exp_dae_conv2.py",            "exp_dae_conv2"),
         vae=    ("exp_vae_conv2.py",            "exp_vae_conv2"),
    ),
    dict(name="ViT",
         ema=    ("exp_vit_ema_1024.py",         "exp_vit_ema_1024"),
         sigreg= ("exp_sigreg_vit.py",            "exp_sigreg_vit"),
         ae=     ("exp_ae_vit.py",               "exp_ae_vit"),
         dae=    ("exp_dae_vit.py",              "exp_dae_vit"),
         vae=    ("exp_vae_vit.py",              "exp_vae_vit"),
    ),
    dict(name="ViT-patch",
         ema=    ("exp_vit_patch_ema.py",        "exp_vit_patch_ema"),
         sigreg= ("exp_sigreg_vit_patch.py",      "exp_sigreg_vit_patch"),
         ae=     ("exp_ae_vit_patch.py",         "exp_ae_vit_patch"),
         dae=    ("exp_dae_vit_patch.py",        "exp_dae_vit_patch"),
         vae=    ("exp_vae_vit_patch.py",        "exp_vae_vit_patch"),
    ),
    dict(name="Gram-dual",
         ema=    ("exp_ema_1024.py",             "exp_ema_1024"),
         sigreg= ("exp12a.py",                   "exp12a"),
         ae=     ("exp_ae_gram_dual.py",         "exp_ae_gram_dual"),
         dae=    ("exp_dae_gram_dual.py",        "exp_dae_gram_dual"),
         vae=    ("exp_vae_gram_dual.py",        "exp_vae_gram_dual"),
    ),
    dict(name="Gram-single",
         ema=    ("exp_gram_single_ema_1024.py", "exp_gram_single_ema_1024"),
         sigreg= ("exp_sigreg_gram_single.py",   "exp_sigreg_gram_single"),
         ae=     ("exp_ae_gram_single.py",       "exp_ae_gram_single"),
         dae=    ("exp_dae_gram_single.py",      "exp_dae_gram_single"),
         vae=    ("exp_vae_gram_single.py",      "exp_vae_gram_single"),
    ),
    dict(name="Gram-GLU",
         ema=    ("exp_gram_glu_ema.py",         "exp_gram_glu_ema"),
         sigreg= ("exp_sigreg_gram_glu.py",      "exp_sigreg_gram_glu"),
         ae=     ("exp_ae_gram_glu.py",          "exp_ae_gram_glu"),
         dae=    ("exp_dae_gram_glu.py",         "exp_dae_gram_glu"),
         vae=    ("exp_vae_gram_glu.py",         "exp_vae_gram_glu"),
    ),
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

    for enc_entry in REGISTRY:
        display = enc_entry["name"]
        logging.info(f"\n{'='*60}\n{display}\n{'='*60}")

        ema_file, ema_exp = enc_entry["ema"]
        try:
            ema_mod = load_module(ema_file)
        except Exception as e:
            logging.warning(f"  Could not load EMA module {ema_file}: {e}")
            continue

        # Build list of (training, params_fn, encode_fn, latent_dim)
        candidates = []

        # 1 — Random baseline (always uses EMA module architecture)
        candidates.append(("random", lambda mod=ema_mod: random_params(mod),
                           ema_mod.encode, ema_mod.LATENT_DIM))

        # 2-5 — Trained regimes: ae, dae, vae, sigreg, ema
        for training in ["ae", "dae", "vae", "sigreg", "ema"]:
            if training not in enc_entry:
                continue
            t_file, t_exp = enc_entry[training]
            try:
                mod    = load_module(t_file)
                params = load_trained_params(t_exp)
                candidates.append((training, lambda p=params: p, mod.encode, mod.LATENT_DIM))
            except Exception as e:
                logging.warning(f"  {training} not available for {display}: {e}")

        for training, params_fn, encode_fn, latent_dim in candidates:
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

        print()
        print("=" * 96)
        print(f"  Dataset: {args.dataset:<16}  K={k}   (test accuracy)")
        print("=" * 96)
        hdr = f"  {'Encoder':<14}"
        for t in TRAINING_ORDER:
            hdr += f"  {t.upper():>8}"
        print(hdr)
        print("  " + "-" * 70)

        def fmt(v): return f"{v:.1%}" if v is not None else "   —  "

        for enc_entry in REGISTRY:
            display = enc_entry["name"]
            d = by_enc.get(display, {})
            vals = [d.get(t, {}).get("test_acc") for t in TRAINING_ORDER]
            print(f"  {display:<14}  " + "  ".join(f"{fmt(v):>8}" for v in vals))

        print()


if __name__ == "__main__":
    main()
