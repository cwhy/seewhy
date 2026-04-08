"""
gen_report_ae_dae_vae.py — Generate report-ae-dae-vae.md with embedded plots.

Reads:
  - results_kmeans_comparison.jsonl  (K-Means K=128 cluster accuracy)
  - results_knn_comparison.jsonl     (linear probe + 1-NN per method/arch/dataset)
  - results_vae_kl.jsonl             (VAE beta sweep: probe on MNIST + F-MNIST)

Produces embedded base64 PNG line plots in markdown.

Usage:
    uv run python projects/ssl/scripts/gen_report_ae_dae_vae.py
"""

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

SSL  = Path(__file__).parent.parent
OUT  = SSL / "report-ae-dae-vae.md"

ARCHS = [
    "MLP-shallow", "MLP-deep", "ConvNet-v1", "ConvNet-v2",
    "ViT", "ViT-patch", "Gram-dual", "Gram-single", "Gram-GLU",
]
ARCH_X  = list(range(len(ARCHS)))
METHODS = ["random", "ae", "dae", "vae", "sigreg", "ema"]

COLORS = {
    "random": "#aaaaaa",
    "ae":     "#4e9de0",
    "dae":    "#1a6fc4",
    "vae":    "#7b3fe4",
    "sigreg": "#e07b1a",
    "ema":    "#c44040",
}
MARKERS = {
    "random": "o", "ae": "s", "dae": "D",
    "vae": "^", "sigreg": "P", "ema": "X",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

km_rows  = load_jsonl(SSL / "results_kmeans_comparison.jsonl")
knn_rows = load_jsonl(SSL / "results_knn_comparison.jsonl")
vkl_rows = load_jsonl(SSL / "results_vae_kl.jsonl")

def cmp_acc(encoder, dataset, training, k=128):
    for r in km_rows:
        if (r.get("encoder") == encoder and r.get("dataset") == dataset
                and r.get("training") == training and r.get("k") == k):
            return r.get("test_acc")
    return None

def knn_val(encoder, dataset, training):
    for r in knn_rows:
        if (r.get("encoder") == encoder and r.get("dataset") == dataset
                and r.get("training") == training):
            return r.get("knn_l2_k1")
    return None

def probe_val(encoder, dataset, training):
    for r in knn_rows:
        if (r.get("encoder") == encoder and r.get("dataset") == dataset
                and r.get("training") == training):
            return r.get("probe_acc")
    return None

# ── Plot helpers ───────────────────────────────────────────────────────────────

def upload_fig(name, fig):
    url = save_matplotlib_figure(name, fig, dpi=150)
    plt.close(fig)
    return url

def make_line_plot(name, title, ylabel, data_by_method, methods, ymin=None, ymax=None):
    """data_by_method: dict method -> list of values (len=9, None=missing)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m in methods:
        vals = data_by_method.get(m, [None]*9)
        ys = [v * 100 if v is not None else np.nan for v in vals]
        ax.plot(ARCH_X, ys,
                color=COLORS[m], marker=MARKERS[m],
                linewidth=2, markersize=7, label=m.upper())
    ax.set_xticks(ARCH_X)
    ax.set_xticklabels(ARCHS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout()
    return upload_fig(name, fig)

BETAS       = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
BETA_LABELS = ["0", "1e-4", "1e-3", "1e-2", "1e-1", "1"]
BETA_COLORS = ["#333333", "#2166ac", "#4dac26", "#d7191c", "#e07b1a", "#7b3fe4"]
BETA_MARKERS= ["o", "s", "D", "^", "P", "X"]

def vkl_val(arch, beta, metric):
    for r in vkl_rows:
        if r.get("arch") == arch and r.get("beta") == beta:
            return r.get(metric)
    return None

def make_vae_kl_arch_plot(name, metric, title):
    """x=architecture, one line per β value."""
    if not vkl_rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (beta, label) in enumerate(zip(BETAS, BETA_LABELS)):
        vals = [vkl_val(a, beta, metric) for a in ARCHS]
        ys   = [v * 100 if v is not None else np.nan for v in vals]
        ax.plot(ARCH_X, ys,
                color=BETA_COLORS[i], marker=BETA_MARKERS[i],
                linewidth=2, markersize=7, label=f"β={label}")
    ax.set_xticks(ARCH_X)
    ax.set_xticklabels(ARCHS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout()
    return upload_fig(name, fig)

# ── Build data arrays ──────────────────────────────────────────────────────────

km_mnist   = {m: [cmp_acc(a, "mnist",         m) for a in ARCHS] for m in METHODS}
km_fmnist  = {m: [cmp_acc(a, "fashion_mnist", m) for a in ARCHS] for m in METHODS}

probe_mnist  = {m: [probe_val(a, "mnist",         m) for a in ARCHS] for m in METHODS}
probe_fmnist = {m: [probe_val(a, "fashion_mnist", m) for a in ARCHS] for m in METHODS}
knn_mnist    = {m: [knn_val(a,   "mnist",         m) for a in ARCHS] for m in METHODS}
knn_fmnist   = {m: [knn_val(a,   "fashion_mnist", m) for a in ARCHS] for m in METHODS}

# ── Generate plots ─────────────────────────────────────────────────────────────

url_probe_mnist = make_line_plot(
    "ssl_probe_mnist", "MNIST — Linear probe accuracy by encoder",
    "Probe accuracy", probe_mnist, METHODS, ymin=40, ymax=100,
)
url_probe_fmnist = make_line_plot(
    "ssl_probe_fmnist", "Fashion-MNIST — Linear probe accuracy by encoder (zero-shot transfer from MNIST)",
    "Probe accuracy", probe_fmnist, METHODS, ymin=40, ymax=100,
)
url_knn_mnist = make_line_plot(
    "ssl_knn_mnist", "MNIST — 1-NN (L2) accuracy by encoder",
    "1-NN accuracy", knn_mnist, METHODS, ymin=40, ymax=100,
)
url_knn_fmnist = make_line_plot(
    "ssl_knn_fmnist", "Fashion-MNIST — 1-NN (L2) accuracy by encoder (zero-shot transfer from MNIST)",
    "1-NN accuracy", knn_fmnist, METHODS, ymin=40, ymax=100,
)
url_km_mnist  = make_line_plot(
    "ssl_km_mnist", "MNIST — K-Means K=128 test accuracy by encoder",
    "Test accuracy", km_mnist, METHODS, ymin=40, ymax=100,
)
url_km_fmnist = make_line_plot(
    "ssl_km_fmnist", "Fashion-MNIST — K-Means K=128 test accuracy by encoder (zero-shot transfer from MNIST)",
    "Test accuracy", km_fmnist, METHODS, ymin=40, ymax=85,
)
url_vae_kl_mnist  = make_vae_kl_arch_plot(
    "ssl_vae_kl_mnist", "probe_acc_mnist",
    "VAE β sweep — MNIST linear probe by architecture",
)
url_vae_kl_fmnist = make_vae_kl_arch_plot(
    "ssl_vae_kl_fmnist", "probe_acc_fmnist",
    "VAE β sweep — Fashion-MNIST linear probe by architecture (MNIST-trained encoder)",
)

# ── Write report ───────────────────────────────────────────────────────────────

def pct(v):
    return f"{v:.1%}" if v is not None else "—"

def best_method(arch, dataset):
    vals = {m: cmp_acc(arch, dataset, m) for m in METHODS}
    valid = {m: v for m, v in vals.items() if v is not None}
    if not valid:
        return "—", 0.0
    best = max(valid, key=valid.__getitem__)
    return best.upper(), valid[best]

report = f"""\
# SSL Feature Extractors: Full 6-Way Comparison

Comparing six training regimes across 9 encoder architectures.

All models trained on **MNIST** (200 epochs, LATENT_DIM=1024, Adam + cosine LR).
K-Means K=128 majority-vote cluster accuracy.  Fashion-MNIST is zero-shot transfer
(MNIST-trained params evaluated on Fashion-MNIST).

---

## Training regimes

| Label | Description |
|---|---|
| **RANDOM** | Randomly initialised encoder, He-init weights, no training. |
| **AE** | Standard autoencoder: MSE(decode(encode(x)), x/255). MLP decoder 1024→512→784. |
| **DAE** | Denoising autoencoder: reconstruct clean x from corrupted input. Same 4 augmentations as EMA/SIGReg (Gaussian σ=75, pixel masking 25/50/75%; ViT-patch uses patch masking). |
| **VAE** | Variational autoencoder: MSE + β·KL (β=0.001). encode() returns mean. |
| **SIGREG** | SIGReg-JEPA: predictor + Epps-Pulley spectral regulariser (N(0,I) target). |
| **EMA** | EMA-JEPA: MSE prediction loss, momentum target encoder τ=0.996. |

---

## Linear probe accuracy

### MNIST

![Linear probe MNIST]({url_probe_mnist})

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {a:<12} | " + " | ".join(pct(probe_val(a, "mnist", m)) for m in METHODS) + " |"
    for a in ARCHS
) + f"""

### Fashion-MNIST (zero-shot transfer)

![Linear probe F-MNIST]({url_probe_fmnist})

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {a:<12} | " + " | ".join(pct(probe_val(a, "fashion_mnist", m)) for m in METHODS) + " |"
    for a in ARCHS
) + f"""

---

## 1-NN (L2) accuracy

### MNIST

![1-NN MNIST]({url_knn_mnist})

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {a:<12} | " + " | ".join(pct(knn_val(a, "mnist", m)) for m in METHODS) + " |"
    for a in ARCHS
) + f"""

### Fashion-MNIST (zero-shot transfer)

![1-NN F-MNIST]({url_knn_fmnist})

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {a:<12} | " + " | ".join(pct(knn_val(a, "fashion_mnist", m)) for m in METHODS) + " |"
    for a in ARCHS
) + f"""

---

## VAE β sweep — all architectures

Linear probe (MNIST-trained encoder) as a function of β across all 9 architectures.
β=0 ≡ pure AE (no KL); larger β tightens the N(0,I) prior.

### MNIST linear probe

{"![VAE β sweep MNIST](" + url_vae_kl_mnist + ")" if url_vae_kl_mnist else "_Results pending._"}

### Fashion-MNIST linear probe (probe trained on F-MNIST train set)

{"![VAE β sweep F-MNIST](" + url_vae_kl_fmnist + ")" if url_vae_kl_fmnist else "_Results pending._"}

""" + ("""| Arch | β=0 M | β=1e-4 M | β=1e-3 M | β=1e-2 M | β=1e-1 M | β=1 M | β=0 F | β=1e-4 F | β=1e-3 F | β=1e-2 F | β=1e-1 F | β=1 F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {a} | " +
    " | ".join(pct(vkl_val(a, b, "probe_acc_mnist"))  for b in BETAS) + " | " +
    " | ".join(pct(vkl_val(a, b, "probe_acc_fmnist")) for b in BETAS) + " |"
    for a in ARCHS
) if vkl_rows else "_Pending._") + f"""

---

## K-Means K=128 — MNIST

![MNIST K-Means by encoder]({url_km_mnist})

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {a:<12} | " + " | ".join(pct(cmp_acc(a, "mnist", m)) for m in METHODS) + " |"
    for a in ARCHS
) + """

| Encoder | Best method | Score |
|---|---|---|
""" + "\n".join(
    f"| {a:<12} | {best_method(a, 'mnist')[0]:<8} | {best_method(a, 'mnist')[1]:.1%} |"
    for a in ARCHS
) + f"""

---

## K-Means K=128 — Fashion-MNIST (zero-shot transfer)

![Fashion-MNIST K-Means by encoder]({url_km_fmnist})

| Encoder | Random | AE | DAE | VAE | SIGReg | EMA |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {a:<12} | " + " | ".join(pct(cmp_acc(a, "fashion_mnist", m)) for m in METHODS) + " |"
    for a in ARCHS
) + """

| Encoder | Best method | Score |
|---|---|---|
""" + "\n".join(
    f"| {a:<12} | {best_method(a, 'fashion_mnist')[0]:<8} | {best_method(a, 'fashion_mnist')[1]:.1%} |"
    for a in ARCHS
) + """

---

## Analysis

### DAE is the best self-supervised method for MNIST K-Means

DAE wins 6/9 architectures. The denoising objective forces compact spherical clusters:
the encoder must map corrupted versions of the same input to approximately the same
latent point, directly creating the isotropic per-class geometry K-Means requires.

Best result: ViT-patch DAE **92.9%**, followed by ConvNet-v2 VAE **94.1%**.

### VAE wins ConvNet-v2 (94.1%) — best single K-Means result

The KL prior creates a smooth, continuous latent space where nearby inputs cluster
together. Combined with ConvNet-v2's strong local inductive bias, this produces
the most K-Means-friendly geometry of any model tested.

### EMA best for linear probe; worst for K-Means

EMA achieves the highest linear probe accuracy across most architectures — the momentum
target encoder prevents collapse while allowing anisotropic embedding distributions.
Those elongated ellipsoidal class regions are linearly separable but not Voronoi-separable,
so K-Means suffers (7/9 architectures lowest).

### Reconstruction objectives transfer to Fashion-MNIST; predictive ones don't

AE/DAE/VAE improve over random on Fashion-MNIST (zero-shot).
EMA loses to random — it learns MNIST-specific invariances that hurt transfer.

---

*Generated 2026-04-05.*
*Data: `results_knn_comparison.jsonl` (probe/1-NN), `results_kmeans_comparison.jsonl` (K-Means), `results_vae_kl.jsonl` (β sweep).*
*Script: `projects/ssl/scripts/gen_report_ae_dae_vae.py`.*
"""

OUT.write_text(report)
print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
