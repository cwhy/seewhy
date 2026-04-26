"""
Profile FAISS search speed at various IVFFlat configs.
Standalone — no exp37 imports needed.

Usage:
    uv run python scripts/tmp/profile_exp37.py 2>&1 | tee /tmp/profile_exp37_out.txt
"""

import sys, time
import numpy as np
import faiss

import os
os.environ["JAX_PLATFORMS"] = "cpu"   # force CPU to avoid GPU OOM on data load

sys.path.insert(0, "/home/newuser/Projects/seewhy")
from shared_lib.datasets import load_supervised_image

data    = load_supervised_image("cifar10")
X_train = np.array(data.X).reshape(50000, 3072).astype(np.float32)

BATCH_SIZE    = 64
STEPS_PER_EP  = 781
N_REPS        = 5
rng           = np.random.RandomState(42)

def bench(idx, n_queries, n_reps=N_REPS, label=""):
    queries = rng.uniform(0, 255, (n_queries, 3072)).astype(np.float32)
    idx.search(queries[:16], 1)  # warm-up
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        idx.search(queries, 1)
        times.append(time.perf_counter() - t0)
    ms = np.mean(times) * 1e3
    epoch_s = ms / 1e3 * STEPS_PER_EP
    print(f"  {label:40s}  {ms:6.0f}ms/call  epoch≈{epoch_s:.0f}s ({epoch_s/60:.1f}min)  200ep≈{epoch_s*200/3600:.1f}h", flush=True)
    return ms

print(f"FAISS search benchmark: 50k × 3072 float32 training set", flush=True)
print(f"Simulating {BATCH_SIZE}×N_VAR queries per step, {STEPS_PER_EP} steps/epoch", flush=True)

# ── N_VAR variants ─────────────────────────────────────────────────────────────
print("\n── N_VAR sweep (IVFFlat nlist=100 nprobe=5) ─────────────────────────", flush=True)
q   = faiss.IndexFlatL2(3072)
idx = faiss.IndexIVFFlat(q, 3072, 100, faiss.METRIC_L2)
idx.train(X_train)
idx.add(X_train)
idx.nprobe = 5
for n_var in [4, 8, 16, 32, 64]:
    bench(idx, BATCH_SIZE * n_var, label=f"N_VAR={n_var} → {BATCH_SIZE*n_var} queries")
del idx

# ── nprobe sweep at N_VAR=16 ───────────────────────────────────────────────────
print("\n── nprobe sweep (IVFFlat nlist=100, N_VAR=16 → 1024 queries) ────────", flush=True)
for nlist, nprobe in [(100, 1), (100, 5), (100, 10), (200, 5), (500, 5), (500, 20)]:
    q   = faiss.IndexFlatL2(3072)
    idx = faiss.IndexIVFFlat(q, 3072, nlist, faiss.METRIC_L2)
    t0  = time.perf_counter()
    idx.train(X_train)
    idx.add(X_train)
    bt  = time.perf_counter() - t0
    idx.nprobe = nprobe
    bench(idx, BATCH_SIZE * 16, label=f"nlist={nlist} nprobe={nprobe} (build {bt:.0f}s)")
    del idx

# ── GPU→CPU transfer ───────────────────────────────────────────────────────────
print("\n── GPU→CPU transfer cost ────────────────────────────────────────────", flush=True)
for n_var in [4, 8, 16, 64]:
    n = BATCH_SIZE * n_var
    mb = n * 3072 * 4 / 1e6
    # PCIe 4.0 theoretical: ~32 GB/s → n*3072*4 bytes
    est_ms = mb / 32000 * 1e3  # 32 GB/s
    print(f"  N_VAR={n_var} ({n}×3072 = {mb:.1f}MB): estimated ~{est_ms:.1f}ms @ PCIe 4.0", flush=True)

print("\nDone.", flush=True)
