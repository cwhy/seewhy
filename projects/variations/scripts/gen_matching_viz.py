"""
Visualize batch matching for a variations experiment.

For a sample batch of training images, shows:
  - Source image
  - Decoded slot 0 and slot 1 (the 2 sampled conditioning vectors)
  - kNN neighbors color-coded by which slot they matched
  - Match counts per slot per image
  - Aggregate histogram of slot assignments

This reveals whether all images collapse to the same slot.

Usage:
    uv run python projects/variations/scripts/gen_matching_viz.py [exp6|exp7|exp8]
"""

import sys, pickle, io, base64, json
from pathlib import Path

ROOT_FOR_SHARED = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_FOR_SHARED))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_media

EXP = sys.argv[1] if len(sys.argv) > 1 else "exp6"

if EXP == "exp6":
    from experiments6 import encode, decode, cats_to_cond, LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM, K_NN, EVAL_COND_CATS
elif EXP == "exp7":
    from experiments7 import encode, decode, cats_to_cond, LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM, K_NN, EVAL_COND_CATS
elif EXP == "exp8":
    from experiments8 import encode, decode, cats_to_cond, LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM, K_NN, EVAL_COND_CATS
elif EXP == "exp9":
    from experiments9 import encode, decode, cats_to_cond, LATENT_DIM, N_CAT, VOCAB_SIZE, COND_DIM, K_NN, EVAL_COND_CATS
else:
    raise ValueError(f"Unknown exp: {EXP}")

PROJECT = Path(__file__).parent.parent
N_VAR = 2       # matching candidates (as used in training)
N_SHOW = 12     # images to show
N_KNN_SHOW = 8  # kNN neighbors to show per image
SEED = 99

# ── Load ───────────────────────────────────────────────────────────────────────

with open(PROJECT / f"params_{EXP}.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

data    = load_supervised_image("mnist")
X_train = np.array(data.X.reshape(data.n_samples, 784).astype(np.float32))
knn_idx = np.load(PROJECT / "knn_mnist_k64.npy")

rng = np.random.RandomState(SEED)
batch_idx = rng.choice(len(X_train), N_SHOW, replace=False)
x_batch   = X_train[batch_idx]           # (N_SHOW, 784)
knn_batch = knn_idx[batch_idx]           # (N_SHOW, K_NN)

# ── Run matching (shared cats across all images — current training behavior) ───

rng_jax = jax.random.PRNGKey(SEED)
cats    = jax.random.randint(rng_jax, (N_VAR, N_CAT), 0, VOCAB_SIZE)   # (N_VAR, N_CAT)
cond_var = params["recon_emb"][None] + cats_to_cond(params, cats)       # (N_VAR, COND_DIM)

x_norm = jnp.array(x_batch / 255.0)
z      = encode(params, x_norm)                                          # (N_SHOW, LATENT_DIM)

B      = N_SHOW
z_rep  = jnp.repeat(z[:, None, :], N_VAR, axis=1).reshape(B * N_VAR, LATENT_DIM)
c_rep  = jnp.broadcast_to(cond_var[None], (B, N_VAR, COND_DIM)).reshape(B * N_VAR, COND_DIM)
Y_all  = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_VAR, 784)
Y_all  = np.array(Y_all)                                                 # (N_SHOW, N_VAR, 784)

# For each image, compute which slot each kNN neighbor matched to
knn_norm   = X_train[knn_batch] / 255.0                                  # (N_SHOW, K_NN, 784)
sq_knn     = (knn_norm ** 2).sum(-1)                                     # (N_SHOW, K_NN)
sq_Y       = (Y_all ** 2).sum(-1)                                        # (N_SHOW, N_VAR)
dot        = np.einsum("bkd,bvd->bkv", knn_norm, Y_all)
dists      = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot         # (N_SHOW, K_NN, N_VAR)
best_slot  = np.argmin(dists, axis=-1)                                   # (N_SHOW, K_NN)
count0     = (best_slot == 0).sum(axis=1)                                # (N_SHOW,)
count1     = (best_slot == 1).sum(axis=1)                                # (N_SHOW,)

# ── Helper: image → base64 PNG ────────────────────────────────────────────────

COLORS = ["#2196F3", "#F44336"]   # blue=slot0, red=slot1

def img_to_b64(arr_784, border_color=None, size=56):
    fig, ax = plt.subplots(figsize=(size/72, size/72), dpi=72)
    ax.imshow(arr_784.reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    if border_color:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)
    fig.patch.set_facecolor(border_color if border_color else "white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def img_tag(b64, title="", size=56):
    return f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" title="{title}" style="display:inline-block;margin:1px">'

# ── Build aggregate stats plot ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(count0, bins=range(K_NN + 2), color=COLORS[0], alpha=0.7, label="slot 0")
axes[0].hist(count1, bins=range(K_NN + 2), color=COLORS[1], alpha=0.5, label="slot 1")
axes[0].set_xlabel("# kNN matched to slot"); axes[0].set_ylabel("# images")
axes[0].set_title(f"Match distribution across {N_SHOW} images (K_NN={K_NN})")
axes[0].legend()

axes[1].bar(["slot 0", "slot 1"], [count0.sum(), count1.sum()], color=COLORS)
axes[1].set_ylabel("total kNN assignments"); axes[1].set_title("Total assignments per slot")
for i, v in enumerate([count0.sum(), count1.sum()]):
    axes[1].text(i, v + 1, str(v), ha="center", fontsize=11)

plt.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
plt.close(fig)
stats_b64 = base64.b64encode(buf.getvalue()).decode()

# ── Build HTML ────────────────────────────────────────────────────────────────

slot_cats_str = [f"cats={cats[v].tolist()}" for v in range(N_VAR)]

rows_html = []
for b in range(N_SHOW):
    src_b64  = img_to_b64(x_batch[b] / 255.0)
    slot0_b64 = img_to_b64(Y_all[b, 0], border_color=COLORS[0])
    slot1_b64 = img_to_b64(Y_all[b, 1], border_color=COLORS[1])

    knn_html = ""
    for k in range(N_KNN_SHOW):
        slot = best_slot[b, k]
        knn_img = img_to_b64(knn_norm[b, k], border_color=COLORS[slot], size=42)
        knn_html += img_tag(knn_img, f"kNN {k}, slot {slot}", size=42)

    rows_html.append(f"""
    <tr>
      <td style="padding:4px;text-align:center">
        {img_tag(src_b64, "source")}<br>
        <small>source</small>
      </td>
      <td style="padding:4px;text-align:center;border-left:2px solid {COLORS[0]}">
        {img_tag(slot0_b64, slot_cats_str[0], size=64)}<br>
        <small style="color:{COLORS[0]}"><b>slot 0: {count0[b]}/{K_NN}</b></small><br>
        <small>{slot_cats_str[0]}</small>
      </td>
      <td style="padding:4px;text-align:center;border-left:2px solid {COLORS[1]}">
        {img_tag(slot1_b64, slot_cats_str[1], size=64)}<br>
        <small style="color:{COLORS[1]}"><b>slot 1: {count1[b]}/{K_NN}</b></small><br>
        <small>{slot_cats_str[1]}</small>
      </td>
      <td style="padding:4px;border-left:2px solid #ccc">{knn_html}<br>
        <small>first {N_KNN_SHOW} kNN (border = matched slot)</small>
      </td>
    </tr>""")

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Batch matching viz — {EXP}</title>
<style>
  body {{ font-family: monospace; background: #f8f8f8; padding: 20px; }}
  h1 {{ font-size: 1.2em; }}
  table {{ border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  th {{ background: #333; color: white; padding: 8px 12px; }}
  .stat {{ background: white; padding: 10px; border-radius: 4px; margin-bottom: 16px; display: inline-block; }}
</style>
</head>
<body>
<h1>Batch matching — {EXP} — shared cats across all images</h1>
<p>
  <b>N_VAR={N_VAR}</b> candidate slots sampled once, shared across all {N_SHOW} images.<br>
  Each kNN neighbor is assigned to the closer decoded slot.<br>
  If one slot dominates (e.g. 60/64 matches), collapse is occurring.
</p>

<div class="stat">
  <b>Slot 0</b> total: {int(count0.sum())} / {N_SHOW * K_NN} assignments
  ({100*count0.sum()/(N_SHOW*K_NN):.1f}%) &nbsp;|&nbsp;
  <b>Slot 1</b> total: {int(count1.sum())} / {N_SHOW * K_NN} assignments
  ({100*count1.sum()/(N_SHOW*K_NN):.1f}%)<br>
  min/max slot0 per image: {int(count0.min())}–{int(count0.max())}
</div>

<br>
<img src="data:image/png;base64,{stats_b64}" style="max-width:700px;margin-bottom:16px;display:block">

<table>
<thead>
  <tr>
    <th>source</th>
    <th style="color:{COLORS[0]}">slot 0</th>
    <th style="color:{COLORS[1]}">slot 1</th>
    <th>first {N_KNN_SHOW} kNN neighbors</th>
  </tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>

</body>
</html>
"""

html_bytes = html.encode("utf-8")
url = save_media(f"matching_viz_{EXP}.html", io.BytesIO(html_bytes), "text/html")
print(f"Matching viz → {url}")
