"""
Generate and upload a report for variations exp19 (CIFAR-10, feature-space kNN).

Usage:
    uv run python projects/variations/scripts/gen_report_exp19.py
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
from shared_lib.report import save_report_file
from experiments19 import (
    encode, decode, cats_to_cond,
    LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM, K_NN,
    EVAL_COND_CATS, DC_C1, DC_C2, DC_C3, DC_STEM, IMG_FLAT,
    ENC_C1, ENC_C2, ENC_C3, ENC_FC1, KNN_UPDATE_EVERY,
)
from shared_lib.datasets import load_supervised_image

PROJECT   = Path(__file__).parent.parent
PLOTS_DIR = PROJECT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
EXP = "exp19"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def flat_to_rgb(x_flat):
    if x_flat.ndim == 1:
        return x_flat.reshape(3, 32, 32).transpose(1, 2, 0)
    return x_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def save_fig_local_and_r2(name, fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.read()
    local_path = PLOTS_DIR / f"{name}.png"
    local_path.write_bytes(png_bytes)
    r2_url = save_media(f"{name}.png", io.BytesIO(png_bytes), "image/png")
    return f"plots/{name}.png", r2_url


# ── Load artifacts ─────────────────────────────────────────────────────────────

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

data    = load_supervised_image("cifar10")
X_test  = np.array(data.X_test.reshape(data.n_test_samples, IMG_FLAT).astype(np.float32))
X_train = np.array(data.X.reshape(data.n_samples, IMG_FLAT).astype(np.float32))
y_train = np.array(data.y)

knn_idx = np.load(PROJECT / "knn_cifar_k64.npy")
print(f"kNN index loaded (pixel-space, for visualization): {knn_idx.shape}")

# ── Plot 1: Learning curves ────────────────────────────────────────────────────

epochs = list(range(len(history["loss"])))
ev_ep, ev_mse = zip(*history["eval_mse"])
rank_data = history["eval_recall_by_rank"]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

ax = axes[0]
ax.plot(epochs, history["loss"],    label="total loss", lw=1.5)
ax.plot(epochs, history["L_recon"], label="L_recon",    lw=1.5, ls="--")
ax.plot(epochs, history["L_var"],   label="L_var",      lw=1.5, ls=":")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Training losses")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(ev_ep, ev_mse, "o-", lw=1.5, label="eval recon MSE")
if history.get("eval_recall_mse"):
    rec_ep, rec_mse = zip(*history["eval_recall_mse"])
    ax.plot(rec_ep, rec_mse, "s--", lw=1.5, label="recall MSE")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Eval MSE (recon_emb + recall)")
ax.grid(alpha=0.3); ax.legend()

ax = axes[2]
n_lines = len(rank_data)
colors  = plt.cm.coolwarm(np.linspace(0, 1, n_lines))
ranks   = np.arange(1, K_NN + 1)
for i, (ep, curve) in enumerate(rank_data):
    ax.plot(ranks, curve, color=colors[i], lw=1.2, alpha=0.8,
            label=f"ep{ep}" if i in (0, n_lines - 1) else None)
ax.set_xlabel("kNN rank (1 = closest)"); ax.set_ylabel("recall MSE")
ax.set_title("Per-rank recall MSE (blue=early → red=final)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(f"variations {EXP} — learning curves (CIFAR-10, feature-space kNN)", fontsize=13)
plt.tight_layout()
curves_loc, curves_url = save_fig_local_and_r2(f"variations_{EXP}_curves", fig, dpi=130)
plt.close(fig)
print(f"curves → {curves_url}")

# ── Plot 2: Reconstruction grid ────────────────────────────────────────────────

NROW, NCOL = 4, 8
indices    = np.arange(0, NROW * NCOL * 10, 10)[:NROW * NCOL]
x_sample   = jnp.array(X_test[indices])
z          = encode(params, x_sample / 255.0)
cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
recon_cond = jnp.broadcast_to(
    jnp.array(params["recon_emb"]) + cats_to_cond(params, cats_recon), (len(indices), COND_DIM)
)
recon = np.array(decode(params, jnp.concatenate([z, recon_cond], axis=-1)))
orig  = np.array(x_sample) / 255.0

fig, axes = plt.subplots(2 * NROW, NCOL, figsize=(NCOL * 1.2, 2 * NROW * 1.2))
for i in range(NROW * NCOL):
    r, c = divmod(i, NCOL)
    axes[2*r,   c].imshow(flat_to_rgb(orig[i]),  vmin=0, vmax=1)
    axes[2*r+1, c].imshow(flat_to_rgb(recon[i]), vmin=0, vmax=1)
    axes[2*r,   c].axis("off")
    axes[2*r+1, c].axis("off")
axes[0, 0].set_title("orig",  fontsize=8, pad=2)
axes[1, 0].set_title("recon", fontsize=8, pad=2)
fig.suptitle(f"variations {EXP} — reconstructions (recon_emb, CIFAR-10 feature-space kNN)", fontsize=11)
plt.tight_layout(pad=0.3)
recon_loc, recon_url = save_fig_local_and_r2(f"variations_{EXP}_recon", fig, dpi=150)
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

class_locs, class_urls = {}, {}
for cls in range(10):
    train_idx = int(np.where(y_train == cls)[0][0])
    x_source  = X_train[train_idx]
    knn_imgs  = X_train[knn_idx[train_idx][:10]]

    x_d     = jnp.array(x_source)[None]
    z_d     = encode(params, x_d / 255.0)
    z_d_rep = jnp.repeat(z_d, GRID_SIDE ** 2, axis=0)
    cur_mosaic_conds = jnp.array(params["recon_emb"]) + cats_to_cond(params, mosaic_cats)
    variations = np.array(decode(params, jnp.concatenate([z_d_rep, cur_mosaic_conds], axis=-1)))

    mosaic = np.zeros((GRID_SIDE * 32, GRID_SIDE * 32, 3))
    for idx in range(GRID_SIDE ** 2):
        r, c = divmod(idx, GRID_SIDE)
        mosaic[r*32:(r+1)*32, c*32:(c+1)*32] = flat_to_rgb(variations[idx])

    cur_eval_conds = jnp.array(params["recon_emb"]) + cats_to_cond(params, EVAL_COND_CATS)
    z_d_rep256 = jnp.repeat(z_d, 256, axis=0)
    eval_vars  = np.array(decode(params, jnp.concatenate([z_d_rep256, cur_eval_conds], axis=-1)))

    knn_norm = knn_imgs / 255.0
    sq_knn   = (knn_norm ** 2).sum(-1)
    sq_var   = (eval_vars ** 2).sum(-1)
    dot      = knn_norm @ eval_vars.T
    best_var = np.argmin(sq_knn[:, None] + sq_var[None, :] - 2.0 * dot, axis=1)
    closest  = eval_vars[best_var]

    cats_recon_src = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond_src = jnp.array(params["recon_emb"]) + cats_to_cond(params, cats_recon_src)
    recon_src = np.array(decode(params, jnp.concatenate([z_d, recon_cond_src], axis=-1)))[0]

    fig = plt.figure(figsize=(7, 9))
    gs  = fig.add_gridspec(3, 11, height_ratios=[1, 1, 16], hspace=0.06, wspace=0.04)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(flat_to_rgb(x_source / 255.0), vmin=0, vmax=1)
    ax0.axis("off"); ax0.set_title("src", fontsize=6, pad=2)

    for k in range(10):
        axk = fig.add_subplot(gs[0, k + 1])
        axk.imshow(flat_to_rgb(knn_imgs[k] / 255.0), vmin=0, vmax=1)
        axk.axis("off")
        if k == 0: axk.set_title("64-NN neighbors →", fontsize=6, pad=2, loc="left")

    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.imshow(flat_to_rgb(recon_src), vmin=0, vmax=1)
    ax_r.axis("off"); ax_r.set_title("recon", fontsize=6, pad=2)

    for k in range(10):
        axc = fig.add_subplot(gs[1, k + 1])
        axc.imshow(flat_to_rgb(closest[k]), vmin=0, vmax=1)
        axc.axis("off")
        if k == 0: axc.set_title("closest decoded →", fontsize=6, pad=2, loc="left")

    ax_m = fig.add_subplot(gs[2, :])
    ax_m.imshow(mosaic, vmin=0, vmax=1, interpolation="nearest")
    ax_m.axis("off")
    ax_m.axhline(y=INNER_SIDE * 32 - 0.5, color="red", lw=0.8, alpha=0.6)
    ax_m.axvline(x=INNER_SIDE * 32 - 0.5, color="red", lw=0.8, alpha=0.6)

    fig.text(0.07, 0.38, "cat1 (0→7) ↓", ha="center", va="center", fontsize=7, rotation=90)
    fig.text(0.5,  0.34, "cat2 (0→7) →", ha="center", va="bottom",  fontsize=7)
    fig.text(0.25, 0.32, "(cat3=0,cat4=0)", ha="center", va="top",    fontsize=6, color="red")
    fig.text(0.75, 0.32, "(cat3=4,cat4=0)", ha="center", va="top",    fontsize=6, color="red")
    fig.text(0.25, 0.01, "(cat3=0,cat4=4)", ha="center", va="bottom", fontsize=6, color="red")
    fig.text(0.75, 0.01, "(cat3=4,cat4=4)", ha="center", va="bottom", fontsize=6, color="red")

    fig.suptitle(
        f"class {cls} ({CIFAR_CLASSES[cls]}) — kNN | closest decoded | mosaic",
        fontsize=9, y=0.99
    )
    plt.tight_layout(pad=0.2)
    loc, url = save_fig_local_and_r2(f"variations_{EXP}_var_class{cls}", fig, dpi=150)
    plt.close(fig)
    class_locs[cls] = loc
    class_urls[cls] = url
    print(f"class {cls} ({CIFAR_CLASSES[cls]}) → {url}")

# ── Markdown ───────────────────────────────────────────────────────────────────

def img_link(alt, img, href): return f"[![{alt}]({img})]({href})"
def make_table(rows):
    lines = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0: lines.append("| " + " | ".join(":---:" for _ in row) + " |")
    return "\n".join(lines)

n_p     = result.get("n_params", "?")
t_s     = result.get("time_s", "?")
fin_mse = result.get("final_recon_mse", "?")

local_var_table = make_table([
    [f"class {c} ({CIFAR_CLASSES[c]})" for c in range(5)],
    [img_link(CIFAR_CLASSES[c], class_locs[c], class_locs[c]) for c in range(5)],
    [f"class {c} ({CIFAR_CLASSES[c]})" for c in range(5, 10)],
    [img_link(CIFAR_CLASSES[c], class_locs[c], class_locs[c]) for c in range(5, 10)],
])

md_local = f"""# variations — exp19 report (CIFAR-10, feature-space kNN)

**Algorithm**: Wide CNN encoder + wide deconvnet decoder + dynamic encoder-space kNN.

Key change from exp17: pixel-space kNN replaced with encoder-embedding kNN.
kNN recomputed every {KNN_UPDATE_EVERY} epochs using 128-d encoder embeddings (cosine similarity via FAISS).
Bootstrap: initial kNN from pixel space.

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | {LATENT_DIM} |
| Encoder | CNN: Conv({ENC_C1},3×3,s2) → Conv({ENC_C2},3×3,s2) → Conv({ENC_C3},3×3,s2) → FC({ENC_FC1}→{LATENT_DIM}) |
| N_CAT | {N_CAT} |
| VOCAB_SIZE | {VOCAB_SIZE} |
| COND_DIM | {COND_DIM} |
| DC_STEM / DC_C1 / DC_C2 / DC_C3 | {DC_STEM} / {DC_C1} / {DC_C2} / {DC_C3} |
| K_NN | {K_NN} |
| kNN type | dynamic encoder-space (cosine, FAISS), updated every {KNN_UPDATE_EVERY} epochs |
| BATCH_SIZE | 128 |
| N_VAR_SAMPLES | 16 |
| MAX_EPOCHS | 100 |
| n_params | {n_p:,} |
| time | {t_s}s |

## Results

**Final reconstruction MSE**: `{fin_mse}` (vs exp17 wide decoder: TBD, exp16 CNN: 0.002545)

## Learning curves

![Learning curves]({curves_loc})

## Reconstructions

![Reconstruction grid]({recon_loc})

## Variation grids

{local_var_table}
"""

report_path = PROJECT / f"report_{EXP}.md"
report_path.write_text(md_local)
print(f"\nLocal report → {report_path}")

r2_var_table = make_table([
    [f"class {c} ({CIFAR_CLASSES[c]})" for c in range(5)],
    [img_link(CIFAR_CLASSES[c], class_urls[c], class_urls[c]) for c in range(5)],
    [f"class {c} ({CIFAR_CLASSES[c]})" for c in range(5, 10)],
    [img_link(CIFAR_CLASSES[c], class_urls[c], class_urls[c]) for c in range(5, 10)],
])
md_r2 = md_local.replace(curves_loc, curves_url).replace(recon_loc, recon_url).replace(local_var_table, r2_var_table)
report_r2_path = PROJECT / f"report_{EXP}_r2.md"
report_r2_path.write_text(md_r2)
report_url = save_report_file(f"variations_{EXP}_report", report_r2_path)
report_r2_path.unlink()
print(f"Report URL: {report_url}")
