"""Checks on the causal chunked delta rule. Correctness first, then memory.

1. chunked == unchunked, for several chunk sizes including ones that do not
   divide the sequence length
2. the causal kernel reproduces recall-gen's own kernel on the tokens where the
   two are supposed to agree — recall-gen's query tokens do not write, so the
   final state it reads IS the state as of those positions
3. gradients match, not just outputs
4. peak memory against sequence length, which is the thing the chunking is for

    .venv/bin/python projects/ar-recall/scripts/test_mixers.py
"""
import os
import sys
from pathlib import Path

# XLA preallocates most of the card on first use. The memory measurements below
# run in subprocesses, and a parent holding a preallocated pool would starve
# them into spurious OOMs — which is exactly what the first version of this
# script reported. Allocate on demand instead; the env is inherited by children.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

import numpy as np
import jax
import jax.numpy as jnp

from lib.mixers import kda_causal, kda_chunkwise, attn_causal

# recall-gen's kernel is loaded BY PATH: `lib` is already this project's package,
# so putting recall-gen on sys.path would not rebind it. Its core.py has no
# relative imports, so a direct file load is enough.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rg_core", PROJECT.parents[0] / "recall-gen" / "lib" / "core.py")
rg_core = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rg_core)
Cfg, kda_recallgen, decay_bias = rg_core.Cfg, rg_core.kda, rg_core._decay_bias


def layer(key, D, H, DK, horizon=160.0):
    """`ba` carries the project's own decay-bias initialisation. It is not a
    detail: with ba = 0 the gate sits at a ~ 0.5 and the chunkwise form's
    within-chunk cumulative decay underflows, which is the precondition test 5
    measures rather than hides."""
    ks = jax.random.split(key, 6)
    lin = lambda k, s: jax.random.normal(k, s) * (1.0 / s[0] ** 0.5)
    return dict(Wq=lin(ks[0], (D, D)), Wk=lin(ks[1], (D, D)), Wv=lin(ks[2], (D, D)),
                Wa=lin(ks[3], (D, D)) * 0.1, ba=jnp.full(D, decay_bias(horizon)),
                Wb=lin(ks[4], (D, H)), bb=jnp.zeros(H),
                Wo=lin(ks[5], (D, D)))


def main():
    D, H, DK, B, N = 128, 4, 32, 3, 37          # N deliberately not round
    cfg = Cfg(d_model=D, n_layers=1, dk=DK, n_heads=H)
    k0, k1 = jax.random.split(jax.random.key(0))
    Lp = layer(k0, D, H, DK)
    x = jax.random.normal(k1, (B, N, D))

    ref = kda_causal(x, Lp, cfg, chunk=None)
    print("1. chunked vs unchunked")
    for c in (1, 4, 8, 16, 32, 64):
        got = kda_causal(x, Lp, cfg, chunk=c)
        d = float(jnp.abs(got - ref).max())
        print(f"   chunk={c:<3d} max|diff| = {d:.3e}  {'ok' if d < 2e-4 else 'MISMATCH'}")

    print("2. causal kernel vs recall-gen's, on non-writing query tokens")
    # recall-gen reads the FINAL state for every token, and its query tokens do
    # not write. So for those positions the two kernels must agree exactly.
    M = 30
    is_ctx = jnp.concatenate([jnp.ones((B, M)), jnp.zeros((B, N - M))], axis=1)
    rg = kda_recallgen(x, Lp, is_ctx, cfg)
    mine = kda_causal(x, Lp, cfg, write=is_ctx, chunk=8)
    d = float(jnp.abs(rg[:, M:] - mine[:, M:]).max())
    scale = float(jnp.abs(rg[:, M:]).max())
    print(f"   query tokens max|diff| = {d:.3e}  (values up to {scale:.3f})  "
          f"{'ok' if d < 2e-4 else 'MISMATCH'}")
    dc = float(jnp.abs(rg[:, :M] - mine[:, :M]).max())
    print(f"   context tokens differ by {dc:.3e} — expected, recall-gen is not "
          f"causal there")

    print("3. chunkwise (matmul form) vs the scan reference")
    for c in (8, 16, 32, 64):
        got = kda_chunkwise(x, Lp, cfg, chunk=c)
        rel = float(jnp.abs(got - ref).max() / jnp.abs(ref).max())
        print(f"   chunk={c:<3d} max|diff|/scale = {rel:.2e}  "
              f"{'ok (fp32 noise)' if rel < 3e-3 else 'MISMATCH'}")
    gw = kda_chunkwise(x, Lp, cfg, write=is_ctx, chunk=16)
    rel = float(jnp.abs(gw[:, M:] - rg[:, M:]).max() / jnp.abs(rg[:, M:]).max())
    print(f"   with a write mask, vs recall-gen on query tokens: {rel:.2e}  "
          f"{'ok (fp32 noise)' if rel < 3e-3 else 'MISMATCH'}")
    print("   in float64 the same comparison is machine epsilon — run")
    print("     JAX_ENABLE_X64=1 ... --f64   (the exactness claim lives there)")

    print("4. chunkwise gradients")
    lw = lambda p: jnp.sum(kda_chunkwise(x, p, cfg, chunk=16) ** 2)
    ls = lambda p: jnp.sum(kda_causal(x, p, cfg, chunk=None) ** 2)
    gw_, gs_ = jax.grad(lw)(Lp), jax.grad(ls)(Lp)
    worst = max(float(jnp.abs(gw_[k] - gs_[k]).max()
                      / (jnp.abs(gs_[k]).max() + 1e-9)) for k in gs_)
    print(f"   max relative grad diff = {worst:.2e}  "
          f"{'ok' if worst < 1e-3 else 'MISMATCH'}")

    print("5. how far the decay can stray before the chunkwise form breaks")
    print("   (a is the per-channel decay; the project initialises it near 0.994)")
    for horizon, lab in ((160.0, "project init"), (16.0, "10x faster forgetting"),
                         (4.0, "very fast"), (1.0, "extreme")):
        Lh = layer(k0, D, H, DK, horizon=horizon)
        a = float(jax.nn.sigmoid(jnp.mean(x @ Lh["Wa"] + Lh["ba"])))
        r = kda_causal(x, Lh, cfg, chunk=None)
        for c in (16, 64):
            g = kda_chunkwise(x, Lh, cfg, chunk=c)
            rel = float(jnp.abs(g - r).max() / (jnp.abs(r).max() + 1e-12))
            print(f"   a~{a:.4f} ({lab:<22}) chunk={c:<3d} rel err {rel:.2e}")

    print("6. gradients")
    lossfn = lambda p, c: jnp.sum(kda_causal(x, p, cfg, chunk=c) ** 2)
    g_ref = jax.grad(lossfn)(Lp, None)
    g_ch = jax.grad(lossfn)(Lp, 8)
    worst = max(float(jnp.abs(g_ref[k] - g_ch[k]).max()) for k in g_ref)
    print(f"   max|grad diff| = {worst:.3e}  {'ok' if worst < 1e-3 else 'MISMATCH'}")

    print("4. peak memory against sequence length (batch 32, d=512, dk=64)")
    # `peak_bytes_in_use` is a process-wide high-water mark and never falls, so
    # measuring several configurations in one process reports the largest so far
    # for every one of them. Each measurement gets a fresh process.
    import subprocess
    for T in (64, 256, 768):
        row = [f"   T={T:<5d}"]
        for name, c in (("naive", "none"), ("chunk=32", "32")):
            r = subprocess.run([sys.executable, __file__, "--mem", str(T), c],
                               capture_output=True, text=True)
            row.append(f"{name} {r.stdout.strip().splitlines()[-1] if r.returncode == 0 else 'OOM'}")
        print("  ".join(row))


def measure(T, chunk):
    D, H, DK, B = 512, 8, 64, 32
    cfg = Cfg(d_model=D, n_layers=1, dk=DK, n_heads=H)
    Lp = layer(jax.random.key(2), D, H, DK)
    xx = jax.random.normal(jax.random.key(3), (B, T, D))
    f = jax.jit(lambda p, y: jnp.sum(kda_causal(y, p, cfg, chunk=chunk) ** 2))
    jax.block_until_ready(jax.grad(f)(Lp, xx))
    print(f"peak {jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 2**30:5.2f} GiB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mem":
        measure(int(sys.argv[2]),
                None if sys.argv[3] == "none" else int(sys.argv[3]))
    else:
        main()
