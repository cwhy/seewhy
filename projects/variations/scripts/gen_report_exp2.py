"""
Generate and upload a report for variations exp2.

Produces:
  - Learning curves: total loss, L_recon, L_var (each on its own panel)
  - Reconstruction grid (test originals vs b=0⁸ outputs)
  - 10 separate variation mosaics (one per digit, 16×16 = all 256 bit patterns)
    with a kNN neighbor strip above each mosaic
  - Local report: projects/variations/report_exp2.md  (relative paths, works offline)
  - HTML report uploaded to R2

Usage:
    uv run python projects/variations/scripts/gen_report_exp2.py
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
from experiments2 import encode, decode, LATENT_DIM, N_BITS
from shared_lib.datasets import load_supervised_image

PROJECT   = Path(__file__).parent.parent
PLOTS_DIR = PROJECT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
EXP = "exp2"


def save_fig_local_and_r2(name: str, fig, dpi: int = 150):
    """Save fig to plots/ (local) and upload to R2. Returns (local_rel_path, r2_url)."""
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
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Eval reconstruction MSE (b=0⁸)")
ax.grid(alpha=0.3); ax.legend()

fig.suptitle(f"variations {EXP} — learning curves", fontsize=13)
plt.tight_layout()
curves_loc, curves_url = save_fig_local_and_r2(f"variations_{EXP}_curves", fig, dpi=130)
plt.close(fig)
print(f"curves → {curves_url}")

# ── Plot 2: Reconstruction grid ────────────────────────────────────────────────

NROW, NCOL = 4, 8
indices = np.arange(0, NROW * NCOL * 10, 10)[:NROW * NCOL]
x_sample = jnp.array(X_test[indices])
z        = encode(params, x_sample / 255.0)
zeros    = jnp.zeros((len(indices), N_BITS))
recon    = np.array(decode(params, jnp.concatenate([z, zeros], axis=-1)))
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
fig.suptitle(f"variations {EXP} — reconstructions (b=0⁸)", fontsize=11)
plt.tight_layout(pad=0.3)
recon_loc, recon_url = save_fig_local_and_r2(f"variations_{EXP}_recon", fig, dpi=150)
plt.close(fig)
print(f"recon  → {recon_url}")

# ── Plot 3: One combined figure per digit (kNN strip + mosaic) ─────────────────

all_bits = jnp.array([[int(b) for b in format(i, f"0{N_BITS}b")] for i in range(2**N_BITS)],
                     dtype=jnp.float32)  # (256, 8)
GRID_SIDE = 16

digit_locs, digit_urls = {}, {}
for digit in range(10):
    # Use training image so knn_idx applies directly
    train_idx = int(np.where(y_train == digit)[0][0])
    x_source  = X_train[train_idx]                         # (784,) uint8-like float32
    knn_imgs  = X_train[knn_idx[train_idx][:10]]           # (10, 784)

    # Variation mosaic: all 256 bit patterns
    x_d = jnp.array(x_source)[None]
    z_d = jnp.repeat(encode(params, x_d / 255.0), 2**N_BITS, axis=0)
    zb  = jnp.concatenate([z_d, all_bits], axis=-1)
    variations = np.array(decode(params, zb))              # (256, 784)

    mosaic = np.zeros((GRID_SIDE * 28, GRID_SIDE * 28))
    for i in range(2**N_BITS):
        r, c = divmod(i, GRID_SIDE)
        mosaic[r*28:(r+1)*28, c*28:(c+1)*28] = variations[i].reshape(28, 28)

    # Closest decoded output to each of the 10 kNN neighbors
    knn_norm = knn_imgs / 255.0                                # (10, 784)
    sq_knn   = (knn_norm ** 2).sum(-1)                         # (10,)
    sq_var   = (variations ** 2).sum(-1)                       # (256,)
    dot      = knn_norm @ variations.T                         # (10, 256)
    dists_kv = sq_knn[:, None] + sq_var[None, :] - 2.0 * dot  # (10, 256)
    best_var = np.argmin(dists_kv, axis=1)                     # (10,)
    closest  = variations[best_var]                            # (10, 784)

    # Reconstruction for source image at canonical conditioning (b=0^8)
    z_src     = encode(params, x_d / 255.0)                    # (1, LATENT_DIM)
    zeros_src = jnp.zeros((1, N_BITS))
    recon_src = np.array(decode(params, jnp.concatenate([z_src, zeros_src], axis=-1)))[0]

    # Combined figure: 3 rows [kNN | closest decoded | mosaic]
    fig = plt.figure(figsize=(7, 9))
    gs  = fig.add_gridspec(3, 11, height_ratios=[1, 1, 16], hspace=0.06, wspace=0.04)

    # Row 0: source image + 10 kNN neighbors
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

    # Row 1: reconstruction (col 0) + closest decoded output to each kNN neighbor
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

    # Row 2: mosaic spanning full width
    ax_m = fig.add_subplot(gs[2, :])
    ax_m.imshow(mosaic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_m.axis("off")

    fig.suptitle(f"digit {digit} — kNN (row 0) | closest decoded (row 1) | all 256 bit patterns (row 2)",
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
    """rows: list of lists of cell strings."""
    lines = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(":---:" for _ in row) + " |")
    return "\n".join(lines)

# ── Write local markdown (uses relative paths → works offline) ─────────────────

n_p     = result.get("n_params", "?")
t_s     = result.get("time_s", "?")
fin_mse = result.get("final_recon_mse", "?")

local_var_table = make_table([
    [f"digit {d}" for d in range(5)],
    [img_link(f"digit {d}", digit_locs[d], digit_locs[d]) for d in range(5)],
    [f"digit {d}" for d in range(5, 10)],
    [img_link(f"digit {d}", digit_locs[d], digit_locs[d]) for d in range(5, 10)],
])

md_local = f"""# variations — exp2 report

**Algorithm**: MLP encoder + gram decoder with 8-bit binary conditioning (inverted IMLE: kNN→closest sample, B=64, N_VAR=16).

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | {LATENT_DIM} |
| N_BITS | {N_BITS} |
| MLP_HIDDEN | 256 |
| DEC_RANK | 16 |
| K_NN | 64 |
| ALPHA | 1.0 |
| BATCH_SIZE | 256 |
| MAX_EPOCHS | 100 |
| ADAM_LR | 3e-4 |
| n_params | {n_p:,} |
| time | {t_s}s |

## Results

**Final reconstruction MSE (b=0⁸ on test set)**: `{fin_mse}`

## Learning curves

![Learning curves]({curves_loc})

## Reconstructions (b=0⁸)

Top row: original. Bottom row: decoded with all bits zero.

![Reconstruction grid]({recon_loc})

## Variation grids — all 256 bit patterns

Each image shows: top strip = training source image + its 10 nearest kNN neighbors;
bottom = 16×16 mosaic of all 256 possible 8-bit patterns decoded for that source.
Click any image to enlarge.

{local_var_table}
"""

report_path = PROJECT / f"report_{EXP}.md"
report_path.write_text(md_local)
print(f"\nLocal report → {report_path}")

# ── Write R2 markdown (uses R2 URLs → works from anywhere) ────────────────────

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

# ── Upload HTML report ─────────────────────────────────────────────────────────

report_url = save_report_file(f"variations_{EXP}_report", report_r2_path)
report_r2_path.unlink()   # temp file, not needed on disk
print(f"Report URL: {report_url}")
