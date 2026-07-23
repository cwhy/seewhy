"""
Generate and upload a full visualisation report for exp51
(continuous Gaussian noise conditioning, COND_DIM=64, 300ep).

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python projects/variations/scripts/gen_report_exp51.py
"""

import sys, pickle, json, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

from shared_lib.media import save_media
from shared_lib.datasets import load_supervised_image
from experiments51 import (
    encode, decode,
    LATENT_DIM, COND_DIM, K_NN,
    EVAL_EPS, IMG_FLAT, N_VAR_SAMPLES,
)

PROJECT   = Path(__file__).parent.parent
PLOTS_DIR = PROJECT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
EXP = "exp51"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def flat_to_rgb(x_flat):
    if x_flat.ndim == 1:
        return np.clip(x_flat.reshape(3, 32, 32).transpose(1, 2, 0), 0, 1)
    return np.clip(x_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), 0, 1)


def save_fig(name, fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.read()
    (PLOTS_DIR / f"{name}.png").write_bytes(png_bytes)
    r2_url = save_media(f"{name}.png", io.BytesIO(png_bytes), "image/png")
    return r2_url


def global_knn(x_query, X_db, k):
    sq_q  = (x_query ** 2).sum(-1, keepdims=True)
    sq_db = (X_db ** 2).sum(-1)
    dot   = x_query @ X_db.T
    dists = sq_q + sq_db - 2.0 * dot
    return jnp.argsort(dists, axis=1)[:, :k]


# ── Load artifacts ─────────────────────────────────────────────────────────────

print("Loading params…")
with open(PROJECT / f"params_{EXP}.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

with open(PROJECT / f"history_{EXP}.pkl", "rb") as f:
    history = pickle.load(f)

result = None
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("experiment") == EXP:
        result = r
        break

print("Loading CIFAR-10…")
data     = load_supervised_image("cifar10")
X_test   = np.array(data.X_test).reshape(data.n_test_samples, IMG_FLAT).astype(np.float32)
X_train  = np.array(data.X).reshape(data.n_samples, IMG_FLAT).astype(np.float32)
y_train  = np.array(data.y)
X_tr_jax = jnp.array(X_train)

n_params_val = result["n_params"]
time_s       = result["time_s"]
final_recon  = result["final_recon_mse"]
final_recall = result["final_recall_mse"]
latent_dim   = result["latent_dim"]
n_var        = result["n_var_samples"]
dc_c1        = result["dc_c1"]
dc_c2        = result["dc_c2"]
dc_c3        = result["dc_c3"]
dc_stem      = result["dc_stem"]
enc_c1       = result["enc_c1"]
enc_c2       = result["enc_c2"]
enc_c3       = result["enc_c3"]
enc_fc1      = result["enc_fc1"]

# ── Plot 1: Learning curves ────────────────────────────────────────────────────

epochs      = list(range(len(history["loss"])))
rec_ep, rec_mse = zip(*history["eval_recall_mse"])
recon_ep, recon_mse_ev = zip(*history["eval_recon_mse"])
rank_data   = history["eval_recall_by_rank"]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

ax = axes[0]
ax.plot(epochs, history["loss"],    lw=1.8, color="#2563eb", label="total loss")
ax.plot(epochs, history["L_imle"],  lw=1.4, color="#7c3aed", label="L_imle",  linestyle="--")
ax.plot(epochs, history["L_recon"], lw=1.4, color="#059669", label="L_recon", linestyle=":")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Training loss (Gaussian noise conditioning)")
ax.set_yscale("log")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(recon_ep, recon_mse_ev, "o-",  lw=1.8, label="recon MSE (test, eps=0)", color="#2563eb", ms=4)
ax.plot(rec_ep,   rec_mse,      "s--", lw=1.8, label="recall MSE",              color="#dc2626", ms=4)
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Eval metrics over training")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
n_lines = len(rank_data)
colors  = plt.cm.coolwarm(np.linspace(0, 1, n_lines))
ranks   = np.arange(1, K_NN + 1)
for i, (ep, curve) in enumerate(rank_data):
    ax.plot(ranks, curve, color=colors[i], lw=1.2, alpha=0.85,
            label=f"ep {ep}" if i in (0, n_lines // 2, n_lines - 1) else None)
ax.set_xlabel("neighbour rank (1 = closest)")
ax.set_ylabel("recall MSE")
ax.set_title("Per-rank recall MSE\n(blue = early  →  red = final)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(f"Learning dynamics — {EXP} (continuous Gaussian conditioning, COND_DIM={COND_DIM}, 300ep)", fontsize=13)
plt.tight_layout()
curves_url = save_fig(f"variations_{EXP}_curves", fig, dpi=130)
plt.close(fig)
print(f"curves → {curves_url}")

# ── Plot 2: Reconstruction grid ────────────────────────────────────────────────

NROW, NCOL = 4, 8
indices    = np.arange(0, NROW * NCOL * 10, 10)[:NROW * NCOL]
x_sample   = jnp.array(X_test[indices])
z          = encode(params, x_sample / 255.0)
recon_eps  = jnp.zeros((len(indices), COND_DIM))
recon      = np.array(decode(params, jnp.concatenate([z, recon_eps], axis=-1)))
orig       = np.array(x_sample) / 255.0

fig, axes = plt.subplots(2 * NROW, NCOL, figsize=(NCOL * 1.2, 2 * NROW * 1.2))
for i in range(NROW * NCOL):
    r, c = divmod(i, NCOL)
    axes[2*r,   c].imshow(flat_to_rgb(orig[i]),  vmin=0, vmax=1)
    axes[2*r+1, c].imshow(flat_to_rgb(recon[i]), vmin=0, vmax=1)
    axes[2*r,   c].axis("off")
    axes[2*r+1, c].axis("off")
axes[0, 0].set_title("orig",  fontsize=8, pad=2)
axes[1, 0].set_title("recon\n(eps=0)", fontsize=7, pad=2)
fig.suptitle(f"{EXP} — eps=0 reconstructions on test set (LATENT={latent_dim})", fontsize=11)
plt.tight_layout(pad=0.3)
recon_url = save_fig(f"variations_{EXP}_recon", fig, dpi=150)
plt.close(fig)
print(f"recon  → {recon_url}")

# ── Plot 3: Per-class variation grids ─────────────────────────────────────────

GRID_SIDE = 16   # 16×16 = 256 variations using all EVAL_EPS vectors

eval_conds = EVAL_EPS   # (256, COND_DIM)

for cls in range(10):
    train_idx  = int(np.where(y_train == cls)[0][0])
    x_source   = X_train[train_idx]

    # Global K-NN of source image
    x_q        = jnp.array(x_source[None] / 255.0)
    knn_idx_g  = global_knn(x_q, X_tr_jax / 255.0, K_NN + 1)
    knn_imgs   = np.array(X_tr_jax[knn_idx_g[0, 1:11]])   # 10 neighbours

    # 16×16 mosaic of 256 eps variations
    x_d        = jnp.array(x_source)[None]
    z_d        = encode(params, x_d / 255.0)
    z_d_rep    = jnp.repeat(z_d, 256, axis=0)
    variations = np.array(decode(params, jnp.concatenate([z_d_rep, eval_conds], axis=-1)))

    mosaic = np.zeros((GRID_SIDE * 32, GRID_SIDE * 32, 3))
    for idx in range(256):
        r, c = divmod(idx, GRID_SIDE)
        mosaic[r*32:(r+1)*32, c*32:(c+1)*32] = flat_to_rgb(variations[idx])

    # Find closest variation to each of the 10 global NN
    knn_norm = knn_imgs / 255.0
    sq_knn   = (knn_norm ** 2).sum(-1)
    sq_var   = (variations ** 2).sum(-1)
    dot_kv   = knn_norm @ variations.T
    best_var = np.argmin(sq_knn[:, None] + sq_var[None, :] - 2.0 * dot_kv, axis=1)
    closest  = variations[best_var]

    # eps=0 decode as "reconstruction"
    recon_src = np.array(decode(params, jnp.concatenate(
        [z_d, jnp.zeros((1, COND_DIM))], axis=-1)))[0]

    fig = plt.figure(figsize=(7, 9))
    gs  = fig.add_gridspec(3, 11, height_ratios=[1, 1, 16], hspace=0.06, wspace=0.04)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(flat_to_rgb(x_source / 255.0), vmin=0, vmax=1)
    ax0.axis("off"); ax0.set_title("src", fontsize=6, pad=2)

    for k in range(10):
        axk = fig.add_subplot(gs[0, k + 1])
        axk.imshow(flat_to_rgb(knn_imgs[k] / 255.0), vmin=0, vmax=1)
        axk.axis("off")
        if k == 0: axk.set_title("global NN →", fontsize=6, pad=2, loc="left")

    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.imshow(flat_to_rgb(recon_src), vmin=0, vmax=1)
    ax_r.axis("off"); ax_r.set_title("eps=0\nrecon", fontsize=6, pad=2)

    for k in range(10):
        axc = fig.add_subplot(gs[1, k + 1])
        axc.imshow(flat_to_rgb(closest[k]), vmin=0, vmax=1)
        axc.axis("off")
        if k == 0: axc.set_title("closest var →", fontsize=6, pad=2, loc="left")

    ax_m = fig.add_subplot(gs[2, :])
    ax_m.imshow(mosaic, vmin=0, vmax=1, interpolation="nearest")
    ax_m.axis("off")

    cls_name = CIFAR_CLASSES[cls]
    fig.suptitle(
        f"{EXP} — class {cls} ({cls_name})  |  16×16 mosaic of 256 fixed N(0,I) eps variations\n"
        f"Top: source + global 10-NN  ·  Middle: eps=0 decode + closest variations to global NN",
        fontsize=8
    )
    plt.tight_layout(pad=0.2)
    url = save_fig(f"variations_{EXP}_var_class{cls}", fig, dpi=150)
    plt.close(fig)
    print(f"class {cls} ({cls_name}) → {url}")

print("All plots done.")
