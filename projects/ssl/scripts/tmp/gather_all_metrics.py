"""Gather all metrics from all result files and print a unified table."""
import json
from pathlib import Path

SSL = Path(__file__).parent.parent.parent

def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except: pass
    return rows

ssl_rows   = load_jsonl(SSL / "results.jsonl")
knn_rows   = load_jsonl(SSL / "results_knn.jsonl")
km_rows    = load_jsonl(SSL / "results_kmeans.jsonl")
cmp_rows   = load_jsonl(SSL / "results_kmeans_comparison.jsonl")
sup_rows   = load_jsonl(SSL / "results_supervised.jsonl")

# ── Index helpers ──────────────────────────────────────────────────────────────

def ssl_probe(exp_name):
    for r in ssl_rows:
        if r.get("experiment") == exp_name:
            return r.get("final_probe_acc")
    return None

def knn(exp_name, dataset, metric, k):
    for r in knn_rows:
        if r.get("exp_name") == exp_name and r.get("dataset") == dataset:
            return r.get(f"knn_{metric}_k{k}")
    return None

def km(exp_name, dataset, K, proj=0):
    for r in km_rows:
        if (r.get("exp_name") == exp_name and r.get("dataset") == dataset
                and r.get("k") == K and r.get("proj_dim", 0) == proj):
            return r.get("test_acc")
    return None

def cmp(encoder, dataset, training, K=128):
    for r in cmp_rows:
        if (r.get("encoder") == encoder and r.get("dataset") == dataset
                and r.get("training") == training and r.get("k") == K):
            return r.get("test_acc")
    return None

def sup(encoder_key):
    for r in sup_rows:
        if r.get("encoder") == encoder_key:
            return r.get("test_acc")
    return None

def pct(v):
    if v is None: return "   —  "
    return f"{v:.1%}"

# ── Encoder registry ───────────────────────────────────────────────────────────

ENCODERS = [
    # display, mnist_ema_exp, fmnist_ema_exp, sup_key
    ("MLP-shallow",  "exp_mlp_ema_1024",         "fmnist_exp_mlp_ema",          "mlp1"),
    ("MLP-deep",     "exp_mlp2_ema_1024",        "fmnist_exp_mlp2_ema",         "mlp2"),
    ("ConvNet-v1",   "exp_conv_ema_1024",        "fmnist_exp_conv_ema",         "conv1"),
    ("ConvNet-v2",   "exp_conv2_ema_1024",       "fmnist_exp_conv2_ema",        "conv2"),
    ("ViT",          "exp_vit_ema_1024",         "fmnist_exp_vit_ema",          "vit"),
    ("ViT-patch",    "exp_vit_patch_ema",        "fmnist_exp_vit_patch_ema",    "vit_patch"),
    ("Gram-dual",    "exp_ema_1024",             "fmnist_exp_ema",              "gram"),
    ("Gram-single",  "exp_gram_single_ema_1024", "fmnist_exp_gram_single_ema",  "gram_single"),
    ("Gram-GLU",     "exp_gram_glu_ema",         "fmnist_exp_gram_glu_ema",     "gram_glu"),
]

print("\n" + "="*110)
print("MNIST — full metric comparison")
print("="*110)
print(f"{'Encoder':<14} {'SupCls':>7} {'LinProbe':>9} {'KNN-1':>7} {'KNN-5':>7} {'KNN-5cos':>9} "
      f"{'KM10':>6} {'KM128':>7} {'KM128p':>8} {'RandKM':>8} {'SigKM':>7}")
print("-"*110)
for name, me, fe, sk in ENCODERS:
    row = [
        pct(sup(sk)),
        pct(ssl_probe(me)),
        pct(knn(me, "mnist", "l2", 1)),
        pct(knn(me, "mnist", "l2", 5)),
        pct(knn(me, "mnist", "cos", 5)),
        pct(km(me,  "mnist", 10,  0)),
        pct(km(me,  "mnist", 128, 0)),
        pct(km(me,  "mnist", 128, 256)),
        pct(cmp(name, "mnist", "random")),
        pct(cmp(name, "mnist", "sigreg")),
    ]
    print(f"{name:<14} " + "  ".join(row))

print("\n" + "="*110)
print("Fashion-MNIST — full metric comparison")
print("="*110)
print(f"{'Encoder':<14} {'LinProbe':>9} {'KNN-1':>7} {'KNN-5':>7} {'KNN-5cos':>9} "
      f"{'KM10':>6} {'KM128':>7} {'KM128p':>8} {'RandKM':>8} {'SigKM':>7}")
print("-"*110)
for name, me, fe, sk in ENCODERS:
    row = [
        pct(ssl_probe(fe)),
        pct(knn(fe, "fashion_mnist", "l2", 1)),
        pct(knn(fe, "fashion_mnist", "l2", 5)),
        pct(knn(fe, "fashion_mnist", "cos", 5)),
        pct(km(fe,  "fashion_mnist", 10,  0)),
        pct(km(fe,  "fashion_mnist", 128, 0)),
        pct(km(fe,  "fashion_mnist", 128, 256)),
        pct(cmp(name, "fashion_mnist", "random")),
        pct(cmp(name, "fashion_mnist", "sigreg")),
    ]
    print(f"{name:<14} " + "  ".join(row))

# Also dump raw sigreg rows for reference
print("\n--- SIGReg rows in comparison JSONL ---")
for r in cmp_rows:
    if r.get("training") == "sigreg":
        print(json.dumps(r))
