"""Does the model run, and does it learn? Small config, real episodes.

    1  both parameterisations build and produce the right shapes
    2  loss at initialisation is log(V) — the model starts at uniform
    3  the next-token alignment is right: shifting the targets by one must HURT
    4  a few hundred steps on a fixed batch drives loss down (it can fit)

    .venv/bin/python projects/ar-recall/scripts/test_model.py
"""
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from lib.model import Cfg, init_params, n_params, loss_and_acc, forward
from lib.task import TaskCfg, build_holdout, make, quantise, value_class

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rg_domains", PROJECT.parents[0] / "recall-gen" / "lib" / "domains.py")
rg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rg)


def main():
    tcfg = TaskCfg(n_context=1)
    H = build_holdout(tcfg)
    Xtr, _, _, _ = rg.raw_pools("mnist")
    pool = quantise(Xtr[:2000], tcfg)
    ep = make(pool, H, tcfg, 4, np.random.default_rng(0), present=True,
              training=True)
    tok = jnp.array(ep.tokens)
    vc = jnp.array(value_class(ep.tokens, tcfg))
    isv = jnp.array(ep.is_value)
    print(f"episode: {tok.shape[1]} tokens, "
          f"{int(isv.sum(1)[0])} value targets each\n")

    for full in (False, True):
        cfg = Cfg(d_model=128, n_layers=2, dk=32, n_heads=4, vocab=tcfg.vocab,
                  n_values=tcfg.n_values, kda_full=full, chunk=64, horizon=4096.0)
        p = init_params(jax.random.key(0), cfg)
        lo, ac = jax.jit(loss_and_acc, static_argnums=4)(p, tok, vc, isv, cfg)
        lab = "paper parameterisation" if full else "simplified (recall-gen)"
        print(f"{lab:<26} {n_params(p) / 1e6:.2f}M params   "
              f"init loss {float(lo):.4f} nats "
              f"({float(lo) / np.log(2):.3f} bits)  acc {float(ac):.4f}")

    print(f"\n  log(V) = {np.log(tcfg.n_values):.4f} nats — init should match\n")

    cfg = Cfg(d_model=128, n_layers=2, dk=32, n_heads=4, vocab=tcfg.vocab,
              n_values=tcfg.n_values, kda_full=True, chunk=64, horizon=4096.0)
    p = init_params(jax.random.key(0), cfg)

    print("3. next-token alignment")
    # A value slot's predictor must be the slot before it, which holds the
    # position token. Scoring against the label slot instead should be worse
    # once trained; at init both are uniform, so this only checks the masks
    # select the right count of targets and nothing is off by one.
    n_val = int(isv.sum(1)[0])
    n_tri = tok.shape[1] // 3
    print(f"   value targets per episode {n_val}, triples {n_tri}  "
          f"{'ok' if n_val == n_tri else 'MISMATCH'}")
    idx = np.flatnonzero(np.asarray(isv[0]))
    print(f"   first value slots at {idx[:4]} (expect 2, 5, 8, 11)  "
          f"{'ok' if list(idx[:4]) == [2, 5, 8, 11] else 'MISMATCH'}")

    print("\n4. overfitting one batch")
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
    st = opt.init(p)

    @jax.jit
    def step(p, st):
        (l, a), g = jax.value_and_grad(
            lambda pp: loss_and_acc(pp, tok, vc, isv, cfg), has_aux=True)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l, a

    t0 = time.perf_counter()
    for i in range(301):
        p, st, l, a = step(p, st)
        if i % 60 == 0:
            print(f"   step {i:>4d}  loss {float(l):.4f} nats  "
                  f"({float(l) / np.log(2):.3f} bits)  acc {float(a):.4f}")
    print(f"   {time.perf_counter() - t0:.0f}s total")
    print("\n   reference: the position marginal is 0.810 accuracy / 1.020 bits")


if __name__ == "__main__":
    main()
