"""
Is KDA's result real, or is it reading the token it predicts?

KDA at H=32 solves every sparsity from s=3 to s=8 in a median of 53 steps, with t* identical
across difficulties. Insensitivity to s is exactly what a leak looks like — and it is also
what the paper's own argument predicts for an architecture that selects positions by linear
readout instead of by search. The two hypotheses are distinguished by a control:

    replace x1 with FRESH RANDOM BITS, independent of x0.

Nothing can predict that. Any causal model must sit at ln 2 = 0.693 on the second half. A
model that can see the token it is predicting will drive it to 0.

Run on KDA and, as a reference, the transformer.

Usage: uv run --no-sync python projects/sparse-attn-emergence/scripts/check_kda_leak.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from lib.models import (Config, forward, forward_kda, init_kda_params,      # noqa: E402
                        init_params)
from lib.tasks import linear_map_matrix                                     # noqa: E402

S, C = 16, 2
STEPS, BATCH, N_SEEDS, LR = 1500, 256, 4, 1e-3
PLATEAU = float(np.log(2))


def random_second_half(key, A, batch):
    """concat(x0, r) with r independent of x0 — unpredictable by construction."""
    k1, k2 = jax.random.split(key)
    x0 = jax.random.bernoulli(k1, 0.5, (batch, S)).astype(jnp.int32)
    r = jax.random.bernoulli(k2, 0.5, (batch, S)).astype(jnp.int32)
    return jnp.concatenate([x0, r], axis=1)


def run(arch, heads):
    cfg = Config(1, 128, 512, heads, 128 // heads, C, S * 2)
    init_fn = init_kda_params if arch == "kda" else init_params
    fwd = forward_kda if arch == "kda" else forward

    keys = jax.random.split(jax.random.key(0), N_SEEDS)
    params = jax.vmap(lambda k: init_fn(k, cfg))(keys)
    A = jax.vmap(lambda k: linear_map_matrix(k, S, 3))(
        jax.random.split(jax.random.fold_in(jax.random.key(0), 99), N_SEEDS))
    opt = optax.adamw(optax.join_schedules(
        [optax.linear_schedule(0.0, LR, 200), optax.constant_schedule(LR)], [200]),
        weight_decay=0.01)
    opt_state = jax.vmap(opt.init)(params)

    def loss_fn(p, b):
        lg = fwd(p, b, cfg)[:, S - 1 : 2 * S - 1, :]
        ls = jax.nn.log_softmax(lg, -1)
        return -jnp.take_along_axis(ls, b[:, S:][..., None], -1).squeeze(-1).mean()

    def chunk(p, st, ks, A):
        def body(carry, key):
            p, st = carry
            g = jax.grad(loss_fn)(p, random_second_half(key, A, BATCH))
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), 0.0

        (p, st), _ = jax.lax.scan(body, (p, st), ks)
        return p, st

    step = jax.jit(jax.vmap(chunk))
    base = jax.random.fold_in(jax.random.key(0), 1)
    for c in range(STEPS // 100):
        ks = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * 100)
        params, opt_state = step(params, opt_state, ks.reshape(N_SEEDS, 100), A)

    ev = jax.vmap(lambda k, a: random_second_half(k, a, 1024))(
        jax.random.split(jax.random.key(7), N_SEEDS), A)
    loss = np.asarray(jax.jit(jax.vmap(loss_fn))(params, ev))
    acc = np.asarray(jax.jit(jax.vmap(lambda p, b: (
        fwd(p, b, cfg)[:, S - 1 : 2 * S - 1, :].argmax(-1) == b[:, S:]).mean()))(params, ev))
    return loss, acc


print(f"CONTROL: second half is random noise. Any causal model must stay at "
      f"ln 2 = {PLATEAU:.4f}, accuracy 0.50.\n")
for arch, heads in (("kda", 32), ("kda", 8), ("transformer", 8)):
    loss, acc = run(arch, heads)
    verdict = "LEAK" if loss.mean() < 0.6 else "clean"
    print(f"  {arch:<12} H={heads:<3} loss {loss.mean():.4f}  acc {acc.mean():.3f}   {verdict}")
