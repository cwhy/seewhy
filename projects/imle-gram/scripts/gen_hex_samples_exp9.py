"""
Generate a sample grid with hex-encoded discrete codes as titles (exp9).
Slot 0 is always 0; remaining 15 slots shown as hex digits.
Usage: uv run python projects/imle-gram/scripts/gen_hex_samples_exp9.py
"""
import pickle, sys, numpy as np, jax, jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments9 import decode, N_VARS, N_CODES, BIN_CENTERS, hard_lookup, sample_indices
from shared_lib.media import save_matplotlib_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(Path(__file__).parent.parent / "params_exp9.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

rows, cols = 5, 8
key = jax.random.PRNGKey(0)
idx = sample_indices(key, rows * cols)   # slot 0 always 0
z   = hard_lookup(params, idx)
logits  = decode(params, z)
probs   = jax.nn.softmax(logits, axis=-1)
imgs    = np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)).reshape(rows * cols, 28, 28)
idx_np  = np.array(idx)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.8))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    # slot 0 shown as '0', rest as hex
    hex_str = "".join(f"{v:x}" for v in idx_np[i])
    ax.set_title(hex_str, fontsize=6, fontfamily="monospace")
    ax.axis("off")
fig.tight_layout()

url = save_matplotlib_figure("exp9_samples_hex", fig, dpi=120)
print(f"url   {url}")
