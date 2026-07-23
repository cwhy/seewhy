"""Benchmark single forward pass speed to diagnose slow generation."""
import sys, pickle, time
from pathlib import Path
import jax, jax.numpy as jnp

SMALL_LM = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SMALL_LM))

import importlib.util
spec = importlib.util.spec_from_file_location("exp", SMALL_LM / "exp_kdyck_rope.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

with open(SMALL_LM / "pkl" / "params_exp_kdyck_rope_phase1.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

print(f"Devices: {jax.devices()}")

x = jnp.ones((1, 64), dtype=jnp.int32)

# Warm up JIT
mod.forward(params, x).block_until_ready()

N = 64
t = time.perf_counter()
for _ in range(N):
    mod.forward(params, x).block_until_ready()
elapsed = time.perf_counter() - t
print(f"{N} forward passes: {elapsed:.2f}s  ({elapsed/N*1000:.1f}ms each)")
print(f"Estimated time for 16 answers x 64 tokens: {16 * 64 * elapsed/N:.1f}s")
