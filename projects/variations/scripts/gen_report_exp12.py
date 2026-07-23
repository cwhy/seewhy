"""
Generate and upload a report for variations exp12.

Produces:
  - Learning curves (training loss + eval MSE + per-rank recall)
  - Reconstruction grid (test originals vs recon_emb outputs)
  - 10 variation mosaics (one per digit):
    top strip = training source + 10 kNN neighbors
    row 2     = closest decoded (using 256 EVAL_COND_CATS slots)
    bottom    = 16×16 grid arranged as 2×2 block of 8×8 sub-grids
                fixing (cat3,cat4) per quadrant, sweeping cat1×cat2
  - Local report: projects/variations/report_exp12.md
  - HTML report uploaded to R2

Usage:
    uv run python projects/variations/scripts/gen_report_exp12.py
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

from shared_lib.media import save_matplotlib_figure, save_media
from shared_lib.report import save_report_file
from experiments12 import (
    encode, decode, cats_to_cond,
    LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM, K_NN,
    EVAL_COND_CATS,
)
from shared_lib.datasets import load_supervised_image

PROJECT   = Path(__file__).parent.parent
PLOTS_DIR = PROJECT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
EXP = "exp12"


def save_fig_local_and_r2(name: str, fig, dpi: int = 150):
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

data    = load_supervised_image("mnist")
X_test  = np.array(data.X_test.reshape(data.n_test_samples, 784).astype(np.float32))
labels  = np.array(data.y_test)
X_train = np.array(data.X.reshape(data.n_samples, 784).astype(np.float32))
y_train = np.array(data.y)

knn_idx = np.load(PROJECT / "knn_mnist_k64.npy")
print(f"kNN index loaded: {knn_idx.shape}")

# ── Eval conditioning: 256 random samples from EVAL_COND_CATS ─────────────────

eval_conds = jnp.array(params["recon_emb"]) + cats_to_cond(params, EVAL_COND_CATS)   # (256, COND_DIM)

# ── Plot 1: Learning curves (3 panels: losses, eval MSE, per-rank recall) ─────

epochs = list(range(len(history["loss"])))
ev_ep, ev_mse = zip(*history["eval_mse"])

has_rank = len(history.get("eval_recall_by_rank", [])) > 0
if has_rank:
    rank_data = history["eval_recall_by_rank"]   # [(epoch, [K_NN floats]), ...]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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

if has_rank:
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

fig.suptitle(f"variations {EXP} — learning curves", fontsize=13)
plt.tight_layout()
curves_loc, curves_url = save_fig_local_and_r2(f"variations_{EXP}_curves", fig, dpi=130)
plt.close(fig)
print(f"curves → {curves_url}")

# ── Plot 2: Reconstruction grid ────────────────────────────────────────────────

NROW, NCOL = 4, 8
indices   = np.arange(0, NROW * NCOL * 10, 10)[:NROW * NCOL]
x_sample  = jnp.array(X_test[indices])
z         = encode(params, x_sample / 255.0)
cats_recon = jnp.zeros((1, N_CAT), dtype=jnp.int32)
recon_cond = jnp.broadcast_to(
    jnp.array(params["recon_emb"]) + cats_to_cond(params, cats_recon), (len(indices), COND_DIM)
)
recon     = np.array(decode(params, jnp.concatenate([z, recon_cond], axis=-1)))
orig      = np.array(x_sample) / 255.0

fig, axes = plt.subplots(2 * NROW, NCOL, figsize=(NCOL * 1.2, 2 * NROW * 1.2))
for i in range(NROW * NCOL):
    r, c = divmod(i, NCOL)
    axes[2*r,   c].imshow(orig[i].reshape(28, 28),  cmap="gray", vmin=0, vmax=1)
    axes[2*r+1, c].imshow(recon[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[2*r,   c].axis("off")
    axes[2*r+1, c].axis("off")
axes[0, 0].set_title("orig",  fontsize=8, pad=2)
axes[1, 0].set_title("recon", fontsize=8, pad=2)
fig.suptitle(f"variations {EXP} — reconstructions (recon_emb)", fontsize=11)
plt.tight_layout(pad=0.3)
recon_loc, recon_url = save_fig_local_and_r2(f"variations_{EXP}_recon", fig, dpi=150)
plt.close(fig)
print(f"recon  → {recon_url}")

# ── Plot 3: One combined figure per digit (kNN strip + closest decoded + mosaic)
# Mosaic: 16×16 grid as 2×2 block of 8×8 sub-grids
#   TL: (cat3=0, cat4=0), TR: (cat3=4, cat4=0)
#   BL: (cat3=0, cat4=4), BR: (cat3=4, cat4=4)
#   Each sub-grid: row=cat1 (0–7), col=cat2 (0–7)

OUTER_VALS = [(0, 0), (4, 0), (0, 4), (4, 4)]  # (cat3, cat4) for TL, TR, BL, BR
INNER_SIDE = 8   # each sub-grid is 8×8
GRID_SIDE  = 16  # full mosaic is 16×16

def build_mosaic_cats():
    """Build (256, N_CAT) array: 4 quadrants of 8×8, each fixing (cat3,cat4)."""
    rows_list = []
    for quad_r in range(2):
        quad_cols = []
        for quad_c in range(2):
            cat3_val, cat4_val = OUTER_VALS[quad_r * 2 + quad_c]
            for r in range(INNER_SIDE):
                for c in range(INNER_SIDE):
                    quad_cols.append([r, c, cat3_val, cat4_val])
        rows_list.extend(quad_cols)
    # Actually build properly: row-major in 16×16 grid
    cats = np.zeros((GRID_SIDE * GRID_SIDE, N_CAT), dtype=np.int32)
    for row16 in range(GRID_SIDE):
        for col16 in range(GRID_SIDE):
            quad_r = row16 // INNER_SIDE
            quad_c = col16 // INNER_SIDE
            cat3_val, cat4_val = OUTER_VALS[quad_r * 2 + quad_c]
            cat1_val = row16 % INNER_SIDE
            cat2_val = col16 % INNER_SIDE
            cats[row16 * GRID_SIDE + col16] = [cat1_val, cat2_val, cat3_val, cat4_val]
    return jnp.array(cats, dtype=jnp.int32)

mosaic_cats  = build_mosaic_cats()   # (256, N_CAT)
mosaic_conds = cats_to_cond(params, mosaic_cats)  # (256, COND_DIM)

digit_locs, digit_urls = {}, {}
for digit in range(10):
    train_idx = int(np.where(y_train == digit)[0][0])
    x_source  = X_train[train_idx]
    knn_imgs  = X_train[knn_idx[train_idx][:10]]

    # Decode mosaic: 256 slots (16×16 grid)
    x_d  = jnp.array(x_source)[None]
    z_d  = encode(params, x_d / 255.0)
    z_d_rep = jnp.repeat(z_d, GRID_SIDE ** 2, axis=0)
    cur_mosaic_conds = jnp.array(params["recon_emb"]) + cats_to_cond(params, mosaic_cats)
    zc        = jnp.concatenate([z_d_rep, cur_mosaic_conds], axis=-1)
    variations = np.array(decode(params, zc))   # (256, 784)

    # Build 16×16 mosaic image
    mosaic = np.zeros((GRID_SIDE * 28, GRID_SIDE * 28))
    for idx in range(GRID_SIDE ** 2):
        r, c = divmod(idx, GRID_SIDE)
        mosaic[r*28:(r+1)*28, c*28:(c+1)*28] = variations[idx].reshape(28, 28)

    # Closest decoded output to each of the 10 kNN neighbors
    # Use EVAL_COND_CATS (256 random slots) for matching — same as eval
    cur_eval_conds = jnp.array(params["recon_emb"]) + cats_to_cond(params, EVAL_COND_CATS)
    z_d_rep256 = jnp.repeat(z_d, 256, axis=0)
    zc_eval    = jnp.concatenate([z_d_rep256, cur_eval_conds], axis=-1)
    eval_vars  = np.array(decode(params, zc_eval))   # (256, 784)

    knn_norm  = knn_imgs / 255.0
    sq_knn    = (knn_norm ** 2).sum(-1)
    sq_var    = (eval_vars ** 2).sum(-1)
    dot       = knn_norm @ eval_vars.T
    dists_kv  = sq_knn[:, None] + sq_var[None, :] - 2.0 * dot
    best_var  = np.argmin(dists_kv, axis=1)
    closest   = eval_vars[best_var]

    # Reconstruction using recon_emb
    cats_recon_src = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond_src = jnp.array(params["recon_emb"]) + cats_to_cond(params, cats_recon_src)
    recon_src = np.array(decode(params, jnp.concatenate([z_d, recon_cond_src], axis=-1)))[0]

    # Combined figure: 3 rows [kNN | closest decoded | mosaic]
    fig = plt.figure(figsize=(7, 9))
    gs  = fig.add_gridspec(3, 11, height_ratios=[1, 1, 16], hspace=0.06, wspace=0.04)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(x_source.reshape(28, 28) / 255.0, cmap="gray", vmin=0, vmax=1)
    ax0.axis("off")
    ax0.set_title("src", fontsize=6, pad=2)

    for k in range(10):
        axk = fig.add_subplot(gs[0, k + 1])
        axk.imshow(knn_imgs[k].reshape(28, 28) / 255.0, cmap="gray", vmin=0, vmax=1)
        axk.axis("off")
        if k == 0:
            axk.set_title("64-NN neighbors →", fontsize=6, pad=2, loc="left")

    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.imshow(recon_src.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ax_r.axis("off")
    ax_r.set_title("recon", fontsize=6, pad=2)

    for k in range(10):
        axc = fig.add_subplot(gs[1, k + 1])
        axc.imshow(closest[k].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axc.axis("off")
        if k == 0:
            axc.set_title("closest decoded →", fontsize=6, pad=2, loc="left")

    ax_m = fig.add_subplot(gs[2, :])
    ax_m.imshow(mosaic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_m.axis("off")

    # Quadrant dividers
    ax_m.axhline(y=INNER_SIDE * 28 - 0.5, color="red", lw=0.8, alpha=0.6)
    ax_m.axvline(x=INNER_SIDE * 28 - 0.5, color="red", lw=0.8, alpha=0.6)

    fig.text(0.07, 0.38, "cat1 (0→7) ↓", ha="center", va="center",
             fontsize=7, rotation=90)
    fig.text(0.5, 0.34, "cat2 (0→7) →", ha="center", va="bottom",
             fontsize=7)
    fig.text(0.25, 0.32, "(cat3=0,cat4=0)", ha="center", va="top", fontsize=6, color="red")
    fig.text(0.75, 0.32, "(cat3=4,cat4=0)", ha="center", va="top", fontsize=6, color="red")
    fig.text(0.25, 0.01, "(cat3=0,cat4=4)", ha="center", va="bottom", fontsize=6, color="red")
    fig.text(0.75, 0.01, "(cat3=4,cat4=4)", ha="center", va="bottom", fontsize=6, color="red")

    fig.suptitle(
        f"digit {digit} — kNN (row 0) | closest decoded (row 1) | 4×8 mosaic 2×2 quadrants (row 2)",
        fontsize=9, y=0.99
    )
    plt.tight_layout(pad=0.2)

    loc, url = save_fig_local_and_r2(f"variations_{EXP}_var_digit{digit}", fig, dpi=150)
    plt.close(fig)
    digit_locs[digit] = loc
    digit_urls[digit] = url
    print(f"digit {digit} → {url}")

# ── Markdown helpers ───────────────────────────────────────────────────────────

def img_link(alt, img, href):
    return f"[![{alt}]({img})]({href})"

def make_table(rows):
    lines = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(":---:" for _ in row) + " |")
    return "\n".join(lines)

# ── Write local markdown ───────────────────────────────────────────────────────

n_p     = result.get("n_params", "?")
t_s     = result.get("time_s", "?")
fin_mse = result.get("final_recon_mse", "?")

local_var_table = make_table([
    [f"digit {d}" for d in range(5)],
    [img_link(f"digit {d}", digit_locs[d], digit_locs[d]) for d in range(5)],
    [f"digit {d}" for d in range(5, 10)],
    [img_link(f"digit {d}", digit_locs[d], digit_locs[d]) for d in range(5, 10)],
])

md_local = f"""# variations — exp12 report

**Algorithm**: exp11 + vectorized decode (performance optimization).

Same hyperparameters as exp11 (N_VAR=32, K_NN=64, B=64). The `decode()` function has been
rewritten to replace the per-pixel vmap (784 sequential-ish ops) with 3 batched matmuls:
outer products → position projection → gram application. Pure XLA-fused matmuls on GPU.
Results should be directly comparable to exp11 (identical model capacity).

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | {LATENT_DIM} |
| N_CAT | {N_CAT} |
| VOCAB_SIZE | {VOCAB_SIZE} |
| E_DIM | 4 |
| COND_DIM | {COND_DIM} (= N_CAT × E_DIM) |
| Total slots | {VOCAB_SIZE**N_CAT} (8⁴) |
| MLP_HIDDEN | 256 |
| DEC_RANK | 16 |
| K_NN | {K_NN} |
| ALPHA | 1.0 |
| BATCH_SIZE | 64 |
| N_VAR_SAMPLES | 32 |
| MAX_EPOCHS | 100 |
| ADAM_LR | 3e-4 |
| decode | vectorized (3 batched matmuls) |
| n_params | {n_p:,} |
| time | {t_s}s |

## Results

**Final reconstruction MSE (recon_emb on test set)**: `{fin_mse}`

## Learning curves

Training losses, eval MSE, and per-rank recall MSE (blue=early epochs → red=final epoch).
The per-rank plot shows recall error by kNN rank; rank 1 is the nearest neighbor.

![Learning curves]({curves_loc})

## Reconstructions (recon_emb)

Top row: original. Bottom row: decoded with the dedicated `recon_emb` conditioning vector.

![Reconstruction grid]({recon_loc})

## Variation grids — 4×8 mosaic (2×2 quadrant layout)

Each image shows:
- **Row 0**: training source + 10 kNN neighbors
- **Row 1**: closest decoded match per kNN (using 256 EVAL_COND_CATS random slots)
- **Row 2**: 16×16 mosaic arranged as 2×2 block of 8×8 sub-grids.
  Each quadrant fixes (cat3, cat4): TL=(0,0), TR=(4,0), BL=(0,4), BR=(4,4).
  Within each sub-grid: **row=cat1** (0→7), **col=cat2** (0→7).
  Red lines separate quadrants.

{local_var_table}
"""

report_path = PROJECT / f"report_{EXP}.md"
report_path.write_text(md_local)
print(f"\nLocal report → {report_path}")

# ── Write R2 markdown ──────────────────────────────────────────────────────────

r2_var_table = make_table([
    [f"digit {d}" for d in range(5)],
    [img_link(f"digit {d}", digit_urls[d], digit_urls[d]) for d in range(5)],
    [f"digit {d}" for d in range(5, 10)],
    [img_link(f"digit {d}", digit_urls[d], digit_urls[d]) for d in range(5, 10)],
])

md_r2 = md_local.replace(curves_loc, curves_url).replace(recon_loc, recon_url)
md_r2 = md_r2.replace(local_var_table, r2_var_table)

report_r2_path = PROJECT / f"report_{EXP}_r2.md"
report_r2_path.write_text(md_r2)

report_url = save_report_file(f"variations_{EXP}_report", report_r2_path)
report_r2_path.unlink()
print(f"Report URL: {report_url}")
