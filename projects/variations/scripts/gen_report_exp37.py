"""
Generate and upload a full visualisation report for exp37
(CIFAR-10, Global IMLE, LATENT=2048, exact GPU NN, no recon loss).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/variations/scripts/gen_report_exp37.py
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
from experiments37 import (
    encode, decode, cats_to_cond,
    LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM, K_NN,
    EVAL_COND_CATS, IMG_FLAT, N_VAR_SAMPLES,
)

PROJECT   = Path(__file__).parent.parent
PLOTS_DIR = PROJECT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
EXP = "exp37"

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
    """Find k nearest neighbours of x_query (N, D) in X_db (M, D). Returns indices (N, k)."""
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
batch_size   = result["batch_size"]
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
ax.plot(epochs, history["loss"], lw=1.8, color="#2563eb", label="IMLE loss")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Training loss (global IMLE)")
ax.set_yscale("log")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(recon_ep, recon_mse_ev, "o-",  lw=1.8, label="recon MSE (test, zero-cat)", color="#2563eb", ms=4)
ax.plot(rec_ep,   rec_mse,      "s--", lw=1.8, label="recall MSE (global K-NN)",   color="#dc2626", ms=4)
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

fig.suptitle(f"Learning dynamics — {EXP} (CIFAR-10, Global IMLE, LATENT={latent_dim})", fontsize=13)
plt.tight_layout()
curves_url = save_fig(f"variations_{EXP}_curves", fig, dpi=130)
plt.close(fig)
print(f"curves → {curves_url}")

# ── Plot 2: Reconstruction grid ────────────────────────────────────────────────

NROW, NCOL = 4, 8
indices    = np.arange(0, NROW * NCOL * 10, 10)[:NROW * NCOL]
x_sample   = jnp.array(X_test[indices])
z          = encode(params, x_sample / 255.0)
cats_zero  = jnp.zeros((1, N_CAT), dtype=jnp.int32)
recon_cond = jnp.broadcast_to(cats_to_cond(params, cats_zero), (len(indices), COND_DIM))
recon      = np.array(decode(params, jnp.concatenate([z, recon_cond], axis=-1)))
orig       = np.array(x_sample) / 255.0

fig, axes = plt.subplots(2 * NROW, NCOL, figsize=(NCOL * 1.2, 2 * NROW * 1.2))
for i in range(NROW * NCOL):
    r, c = divmod(i, NCOL)
    axes[2*r,   c].imshow(flat_to_rgb(orig[i]),  vmin=0, vmax=1)
    axes[2*r+1, c].imshow(flat_to_rgb(recon[i]), vmin=0, vmax=1)
    axes[2*r,   c].axis("off")
    axes[2*r+1, c].axis("off")
axes[0, 0].set_title("orig",  fontsize=8, pad=2)
axes[1, 0].set_title("recon\n(zero-cat)", fontsize=7, pad=2)
fig.suptitle(f"{EXP} — zero-cat reconstructions on test set (LATENT={latent_dim})", fontsize=11)
plt.tight_layout(pad=0.3)
recon_url = save_fig(f"variations_{EXP}_recon", fig, dpi=150)
plt.close(fig)
print(f"recon  → {recon_url}")

# ── Plot 3: Per-class variation mosaics ────────────────────────────────────────

OUTER_VALS = [(0, 0), (4, 0), (0, 4), (4, 4)]
INNER_SIDE = 8
GRID_SIDE  = 16

def build_mosaic_cats():
    cats = np.zeros((GRID_SIDE * GRID_SIDE, N_CAT), dtype=np.int32)
    for row16 in range(GRID_SIDE):
        for col16 in range(GRID_SIDE):
            quad_r = row16 // INNER_SIDE
            quad_c = col16 // INNER_SIDE
            cat3_val, cat4_val = OUTER_VALS[quad_r * 2 + quad_c]
            cats[row16 * GRID_SIDE + col16] = [row16 % INNER_SIDE, col16 % INNER_SIDE, cat3_val, cat4_val]
    return jnp.array(cats, dtype=jnp.int32)

mosaic_cats = build_mosaic_cats()
eval_conds  = cats_to_cond(params, EVAL_COND_CATS)   # (256, COND_DIM)

for cls in range(10):
    train_idx  = int(np.where(y_train == cls)[0][0])
    x_source   = X_train[train_idx]

    # Global K-NN of source image
    x_q        = jnp.array(x_source[None] / 255.0)
    knn_idx_g  = global_knn(x_q, X_tr_jax / 255.0, K_NN + 1)  # +1 because self is rank 0
    knn_imgs   = np.array(X_tr_jax[knn_idx_g[0, 1:11]])  # skip self (rank 0), take 10 neighbors

    # 16×16 mosaic of variations
    x_d        = jnp.array(x_source)[None]
    z_d        = encode(params, x_d / 255.0)
    z_d_rep    = jnp.repeat(z_d, GRID_SIDE ** 2, axis=0)
    mosaic_conds = cats_to_cond(params, mosaic_cats)
    variations = np.array(decode(params, jnp.concatenate([z_d_rep, mosaic_conds], axis=-1)))

    mosaic = np.zeros((GRID_SIDE * 32, GRID_SIDE * 32, 3))
    for idx in range(GRID_SIDE ** 2):
        r, c = divmod(idx, GRID_SIDE)
        mosaic[r*32:(r+1)*32, c*32:(c+1)*32] = flat_to_rgb(variations[idx])

    # 256 eval variations → find closest to each of the 10 global NN
    z_d_rep256 = jnp.repeat(z_d, 256, axis=0)
    eval_vars  = np.array(decode(params, jnp.concatenate([z_d_rep256,
                    jnp.broadcast_to(eval_conds[None], (1, 256, COND_DIM)).reshape(256, COND_DIM)], axis=-1)))

    knn_norm   = knn_imgs / 255.0
    sq_knn     = (knn_norm ** 2).sum(-1)
    sq_var     = (eval_vars ** 2).sum(-1)
    dot_kv     = knn_norm @ eval_vars.T
    best_var   = np.argmin(sq_knn[:, None] + sq_var[None, :] - 2.0 * dot_kv, axis=1)
    closest    = eval_vars[best_var]

    # zero-cat decode as "reconstruction"
    recon_src  = np.array(decode(params, jnp.concatenate([z_d, recon_cond[:1]], axis=-1)))[0]

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
    ax_r.axis("off"); ax_r.set_title("zero\ncat", fontsize=6, pad=2)

    for k in range(10):
        axc = fig.add_subplot(gs[1, k + 1])
        axc.imshow(flat_to_rgb(closest[k]), vmin=0, vmax=1)
        axc.axis("off")
        if k == 0: axc.set_title("closest var →", fontsize=6, pad=2, loc="left")

    ax_m = fig.add_subplot(gs[2, :])
    ax_m.imshow(mosaic, vmin=0, vmax=1, interpolation="nearest")
    ax_m.axis("off")
    ax_m.axhline(y=INNER_SIDE * 32 - 0.5, color="red", lw=0.8, alpha=0.6)
    ax_m.axvline(x=INNER_SIDE * 32 - 0.5, color="red", lw=0.8, alpha=0.6)

    cls_name = CIFAR_CLASSES[cls]
    fig.suptitle(
        f"{EXP} — class {cls} ({cls_name})  |  16×16 variation mosaic (cat1 × cat2, 4 quadrants = cat3×cat4)\n"
        f"Top: source + global 10-NN  ·  Middle: zero-cat decode + closest variations to global NN",
        fontsize=8
    )
    plt.tight_layout(pad=0.2)
    url = save_fig(f"variations_{EXP}_var_class{cls}", fig, dpi=150)
    plt.close(fig)
    print(f"class {cls} ({cls_name}) → {url}")

print("All plots done.")
