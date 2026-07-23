"""
Generate and upload a report for variations exp5.

Produces:
  - Learning curves
  - Reconstruction grid (test originals vs cat=(0,0) outputs)
  - 10 variation mosaics (one per digit):
    top strip = training source + 10 kNN neighbors
    bottom = 16×16 grid of ALL 256 (cat1, cat2) combinations
             row = cat1 (0–15), col = cat2 (0–15)
  - Local report: projects/variations/report_exp5.md
  - HTML report uploaded to R2

Usage:
    uv run python projects/variations/scripts/gen_report_exp5.py
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
from experiments5 import encode, decode, cats_to_cond, LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM
from shared_lib.datasets import load_supervised_image

PROJECT   = Path(__file__).parent.parent
PLOTS_DIR = PROJECT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
EXP = "exp5"


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

# ── All 256 (cat1, cat2) combinations for the mosaic ──────────────────────────
# Ordered so row=cat1 (0-15), col=cat2 (0-15)

all_cats = jnp.array([[i, j] for i in range(VOCAB_SIZE) for j in range(VOCAB_SIZE)],
                     dtype=jnp.int32)              # (256, 2)
all_conds = cats_to_cond(params, all_cats)         # (256, COND_DIM) — recomputed per digit

# Canonical reconstruction conditioning: (cat1=0, cat2=0)
cat0      = jnp.zeros((1, N_CAT), dtype=jnp.int32)
cond0_vec = cats_to_cond(params, cat0)             # (1, COND_DIM)

# ── Plot 1: Learning curves ────────────────────────────────────────────────────

epochs = list(range(len(history["loss"])))
ev_ep, ev_mse = zip(*history["eval_mse"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(epochs, history["loss"],    label="total loss", lw=1.5)
ax.plot(epochs, history["L_recon"], label="L_recon",    lw=1.5, ls="--")
ax.plot(epochs, history["L_var"],   label="L_var",      lw=1.5, ls=":")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Training losses")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(ev_ep, ev_mse, "o-", lw=1.5, label="eval recon MSE")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Eval reconstruction MSE (cat=(0,0))")
ax.grid(alpha=0.3); ax.legend()

fig.suptitle(f"variations {EXP} — learning curves", fontsize=13)
plt.tight_layout()
curves_loc, curves_url = save_fig_local_and_r2(f"variations_{EXP}_curves", fig, dpi=130)
plt.close(fig)
print(f"curves → {curves_url}")

# ── Plot 2: Reconstruction grid ────────────────────────────────────────────────

NROW, NCOL = 4, 8
indices  = np.arange(0, NROW * NCOL * 10, 10)[:NROW * NCOL]
x_sample = jnp.array(X_test[indices])
z        = encode(params, x_sample / 255.0)
cond0_rep = jnp.broadcast_to(cond0_vec, (len(indices), COND_DIM))
recon    = np.array(decode(params, jnp.concatenate([z, cond0_rep], axis=-1)))
orig     = np.array(x_sample) / 255.0

fig, axes = plt.subplots(2 * NROW, NCOL, figsize=(NCOL * 1.2, 2 * NROW * 1.2))
for i in range(NROW * NCOL):
    r, c = divmod(i, NCOL)
    axes[2*r,   c].imshow(orig[i].reshape(28, 28),  cmap="gray", vmin=0, vmax=1)
    axes[2*r+1, c].imshow(recon[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[2*r,   c].axis("off")
    axes[2*r+1, c].axis("off")
axes[0, 0].set_title("orig",  fontsize=8, pad=2)
axes[1, 0].set_title("recon", fontsize=8, pad=2)
fig.suptitle(f"variations {EXP} — reconstructions (cat=(0,0))", fontsize=11)
plt.tight_layout(pad=0.3)
recon_loc, recon_url = save_fig_local_and_r2(f"variations_{EXP}_recon", fig, dpi=150)
plt.close(fig)
print(f"recon  → {recon_url}")

# ── Plot 3: One combined figure per digit (kNN strip + mosaic) ─────────────────
# Mosaic: all 256 (cat1, cat2) combinations, row=cat1 col=cat2

GRID_SIDE = 16   # 16×16 = 256 slots

digit_locs, digit_urls = {}, {}
for digit in range(10):
    train_idx = int(np.where(y_train == digit)[0][0])
    x_source  = X_train[train_idx]
    knn_imgs  = X_train[knn_idx[train_idx][:10]]

    # Decode all 256 (cat1, cat2) combinations
    x_d  = jnp.array(x_source)[None]
    z_d  = jnp.repeat(encode(params, x_d / 255.0), GRID_SIDE ** 2, axis=0)
    # Recompute cond embeddings from current params (embeddings are learned, must use params)
    cur_conds = cats_to_cond(params, all_cats)                             # (256, COND_DIM)
    zc        = jnp.concatenate([z_d, cur_conds], axis=-1)
    variations = np.array(decode(params, zc))                             # (256, 784)

    # Mosaic: row = cat1 index, col = cat2 index
    mosaic = np.zeros((GRID_SIDE * 28, GRID_SIDE * 28))
    for idx in range(GRID_SIDE ** 2):
        r, c = divmod(idx, GRID_SIDE)
        mosaic[r*28:(r+1)*28, c*28:(c+1)*28] = variations[idx].reshape(28, 28)

    # Closest decoded output to each of the 10 kNN neighbors
    knn_norm = knn_imgs / 255.0
    sq_knn   = (knn_norm ** 2).sum(-1)
    sq_var   = (variations ** 2).sum(-1)
    dot      = knn_norm @ variations.T
    dists_kv = sq_knn[:, None] + sq_var[None, :] - 2.0 * dot
    best_var = np.argmin(dists_kv, axis=1)
    closest  = variations[best_var]

    # Reconstruction for source image at canonical conditioning (cat=(0,0))
    z_src    = encode(params, x_d / 255.0)
    cat0_src = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    cond0_src = cats_to_cond(params, cat0_src)
    recon_src = np.array(decode(params, jnp.concatenate([z_src, cond0_src], axis=-1)))[0]

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

    fig.text(0.07, 0.38, "cat1 (0→15) ↓", ha="center", va="center",
             fontsize=7, rotation=90)

    fig.suptitle(f"digit {digit} — kNN (row 0) | closest decoded (row 1) | all 256 (cat1×cat2) combos (row 2)",
                 fontsize=9, y=0.99)
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

md_local = f"""# variations — exp5 report

**Algorithm**: MLP encoder + gram decoder with 2×16 categorical embedding conditioning
(inverted IMLE: kNN→closest sample, B=256, N_VAR=2, no checkpointing needed).

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | {LATENT_DIM} |
| N_CAT | {N_CAT} |
| VOCAB_SIZE | {VOCAB_SIZE} |
| E_DIM | 4 |
| COND_DIM | {COND_DIM} (= N_CAT × E_DIM) |
| MLP_HIDDEN | 256 |
| DEC_RANK | 16 |
| K_NN | 64 |
| ALPHA | 1.0 |
| BATCH_SIZE | 256 |
| N_VAR_SAMPLES | 2 |
| MAX_EPOCHS | 100 |
| ADAM_LR | 3e-4 |
| n_params | {n_p:,} |
| time | {t_s}s |

## Results

**Final reconstruction MSE (cat=(0,0) on test set)**: `{fin_mse}`

## Learning curves

![Learning curves]({curves_loc})

## Reconstructions (cat=(0,0))

Top row: original. Bottom row: decoded with both categories set to index 0.

![Reconstruction grid]({recon_loc})

## Variation grids — all 256 (cat1 × cat2) combinations

Each image shows: top strip = training source image + its 10 nearest kNN neighbors;
bottom = 16×16 mosaic where **row = cat1 (0→15)** and **col = cat2 (0→15)**.
The structured layout reveals how each embedding dimension controls variation independently.
Click any image to enlarge.

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
