"""Measure code-length statistics for exp20c on MNIST: cb_used, per-position entropy, label entropy."""
from __future__ import annotations

import pickle, sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))
sys.path.insert(0, str(PROJ_DIR))

from shared_lib.datasets import load_supervised_image  # noqa: E402
import experiments20c as exp                            # noqa: E402

with open(PROJ_DIR / "params_exp20c_mnist.pkl", "rb") as f:
    p_np = pickle.load(f)
p = {k: jnp.array(v) for k, v in p_np.items()}

data = load_supervised_image("mnist")
X_train = np.array(data.X.reshape(data.n_samples, 784)) / 255.0
y_train = np.array(data.y, dtype=np.int32)

ids_all = []
for i in range(0, 60_000, 1024):
    bx = jnp.array(X_train[i:i + 1024])
    ids = exp.extract_ids_one(p, bx)            # (B, K_SAMPLE)
    ids_all.append(np.array(ids))
ids_all = np.concatenate(ids_all, axis=0)       # (60000, 16)

# overall cb_used
unique_overall = np.unique(ids_all)
print(f"image-token cb_used (overall, all positions): {len(unique_overall)} / 1013 content slots")

# per-position cb_used
print("per-image-position cb_used:")
for t in range(16):
    u = len(np.unique(ids_all[:, t]))
    cnt = Counter(ids_all[:, t].tolist())
    total = sum(cnt.values())
    H = -sum((c / total) * np.log2(c / total) for c in cnt.values())
    print(f"  pos {t:2d}: {u:4d} unique  entropy {H:5.2f} bits")

# label: 10 digit IDs, count their frequencies (uniform-ish from MNIST class balance)
cnt_y = Counter(y_train.tolist())
total = sum(cnt_y.values())
H_y = -sum((c / total) * np.log2(c / total) for c in cnt_y.values())
print(f"\nlabel cb_used: 10 / 10 (always all digits)  entropy {H_y:.3f} bits  (max log2(10)={np.log2(10):.3f})")

# average bits per image (sum of per-position entropies, ignoring correlations)
total_H = 0
for t in range(16):
    cnt = Counter(ids_all[:, t].tolist())
    total = sum(cnt.values())
    H = -sum((c / total) * np.log2(c / total) for c in cnt.values())
    total_H += H
print(f"\nimage total per-position entropy (independence assumption): {total_H:.2f} bits / image")
print(f"label entropy:                                                {H_y:.2f} bits / label")
print(f"Raw MNIST pixels: 784 × 8 = 6272 bits / image")
print(f"Compression vs raw: {6272 / total_H:.1f}×")
