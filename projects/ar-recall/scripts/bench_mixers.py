"""What sequence length is affordable, for each mixer, at a realistic model size.

At 28x28 with half the image revealed, one label costs 392 x 3 = 1176 tokens, so
the number of labels an episode can hold is decided by this table rather than by
the task. Measures a 4-layer stack (mixer + FFN) through a forward and backward
pass: peak memory and time per optimiser step.

Each configuration runs in its own process — `peak_bytes_in_use` is a
process-wide high-water mark that never falls, so measuring several in one
process reports the largest so far for all of them.

    .venv/bin/python projects/ar-recall/scripts/bench_mixers.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

import jax
import jax.numpy as jnp

from lib.mixers import MIXERS

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rg_core", PROJECT.parents[0] / "recall-gen" / "lib" / "core.py")
rg_core = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rg_core)
Cfg, ln = rg_core.Cfg, rg_core.ln

D, H, DK, LAYERS, FF = 512, 8, 64, 4, 4
# (batch, tokens, what that is in task terms)
CONFIGS = [(8, 768, "8 labels x 32 pos"),
           (8, 1536, "8 labels x 64 pos"),
           (8, 2352, "2 labels x half of 28x28"),
           (8, 4704, "4 labels x half of 28x28"),
           (8, 9408, "8 labels x half of 28x28")]


def stack(key):
    ks = jax.random.split(key, LAYERS * 8)
    lin = lambda k, s: jax.random.normal(k, s) * (1.0 / s[0] ** 0.5)
    return [dict(ln1_g=jnp.ones(D), ln1_b=jnp.zeros(D),
                 Wq=lin(ks[i * 8], (D, D)), Wk=lin(ks[i * 8 + 1], (D, D)),
                 Wv=lin(ks[i * 8 + 2], (D, D)), Wo=lin(ks[i * 8 + 3], (D, D)),
                 Wa=lin(ks[i * 8 + 4], (D, D)) * 0.1, ba=jnp.zeros(D),
                 Wb=lin(ks[i * 8 + 5], (D, H)), bb=jnp.zeros(H),
                 ln2_g=jnp.ones(D), ln2_b=jnp.zeros(D),
                 W1=lin(ks[i * 8 + 6], (D, FF * D)), b1=jnp.zeros(FF * D),
                 W2=lin(ks[i * 8 + 7], (FF * D, D)), b2=jnp.zeros(D))
            for i in range(LAYERS)]


def forward(p, x, cfg, mixer, chunk):
    mix = MIXERS[mixer]
    for Lp in p:
        x = x + mix(ln(x, Lp["ln1_g"], Lp["ln1_b"]), Lp, cfg, chunk=chunk)
        x = x + (jax.nn.gelu(ln(x, Lp["ln2_g"], Lp["ln2_b"]) @ Lp["W1"] + Lp["b1"])
                 @ Lp["W2"] + Lp["b2"])
    return jnp.sum(x ** 2)


def measure(B, T, mixer, chunk):
    cfg = Cfg(d_model=D, n_layers=LAYERS, dk=DK, n_heads=H)
    p = stack(jax.random.key(0))
    x = jax.random.normal(jax.random.key(1), (B, T, D))
    g = jax.jit(jax.grad(lambda pp: forward(pp, x, cfg, mixer, chunk)))
    jax.block_until_ready(g(p))
    t0 = time.perf_counter()
    for _ in range(5):
        r = g(p)
    jax.block_until_ready(r)
    dt = (time.perf_counter() - t0) / 5
    peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30
    print(f"{peak:.2f} {dt * 1000:.0f}")


def main():
    print(f"{'batch':>5} {'tokens':>7}  {'kda scan':>21}  {'kda chunkwise':>21}  "
          f"{'attention':>21}   task")
    for B, T, what in CONFIGS:
        cells = []
        for mixer, chunk in (("kda", "64"), ("kda_wy", "64"), ("attn", "none")):
            r = subprocess.run([sys.executable, __file__, "--one", str(B), str(T),
                                mixer, chunk], capture_output=True, text=True)
            if r.returncode != 0:
                cells.append(f"{'OOM':>21}")
            else:
                peak, ms = r.stdout.strip().splitlines()[-1].split()
                per48k = float(ms) * 48000 / 1000 / 3600
                cells.append(f"{peak:>5}G {ms:>6}ms {per48k:>5.1f}h")
        print(f"{B:>5} {T:>7}  {cells[0]}  {cells[1]}  {cells[2]}   {what}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        measure(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
                None if sys.argv[5] == "none" else int(sys.argv[5]))
    else:
        main()
