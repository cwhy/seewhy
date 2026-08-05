"""
Which rows does KDA actually learn, and is the decay horizon the reason it stops?

exp9 ended at final loss 0.4332 at s = 6, 7 and 8 alike — identical to four decimals across
three sparsities. ln2 x 10/16 = 0.4332, i.e. six of sixteen rows learned and ten at chance.
An exact fraction that does not move with difficulty is a structural limit, not a difficulty
effect.

The natural suspect is decay. Query S-1+i must reach back into x0, and the further into the
second half the query sits, the older that evidence is. So this prints per-POSITION accuracy
for the default horizon (= sequence length) against a horizon 100x longer, which makes the
per-channel decay ~1 (memory that does not fade).

Usage: uv run --no-sync python projects/sparse-attn-emergence/scripts/check_kda_positions.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from lib.models import Config, forward_kda, init_kda_params      # noqa: E402
from lib.tasks import linear_map_batch, linear_map_matrix        # noqa: E402

S, SPARSITY, C = 16, 3, 2
STEPS, BATCH, N_SEEDS, LR = 3000, 256, 4, 1e-3
CFG = Config(1, 128, 512, 8, 16, C, S * 2)


def train(horizon, cfg=None):
    global CFG
    if cfg is not None:
        CFG = cfg
    seed_keys = jax.random.split(jax.random.key(0), N_SEEDS)
    a_keys = jax.random.split(jax.random.fold_in(jax.random.key(0), 99), N_SEEDS)
    params = jax.vmap(lambda k: init_kda_params(k, CFG, horizon))(seed_keys)
    A_all = jax.vmap(lambda k: linear_map_matrix(k, S, SPARSITY))(a_keys)
    opt = optax.adamw(optax.join_schedules(
        [optax.linear_schedule(0.0, LR, 200), optax.constant_schedule(LR)], [200]),
        weight_decay=0.01)
    opt_state = jax.vmap(opt.init)(params)

    def loss_fn(p, b):
        lg = forward_kda(p, b, CFG)[:, S - 1 : 2 * S - 1, :]
        tgt = b[:, S:]
        ls = jax.nn.log_softmax(lg, -1)
        return -jnp.take_along_axis(ls, tgt[..., None], -1).squeeze(-1).mean()

    def chunk(p, st, keys, A):
        def body(carry, key):
            p, st = carry
            g = jax.grad(loss_fn)(p, linear_map_batch(key, A, BATCH))
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), 0.0

        (p, st), _ = jax.lax.scan(body, (p, st), keys)
        return p, st

    run = jax.jit(jax.vmap(chunk))
    base = jax.random.fold_in(jax.random.key(0), 1)
    for c in range(STEPS // 100):
        ks = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * 100)
        params, opt_state = run(params, opt_state, ks.reshape(N_SEEDS, 100), A_all)

    def per_pos(p, A):
        b = linear_map_batch(jax.random.key(999), A, 1024)
        lg = forward_kda(p, b, CFG)[:, S - 1 : 2 * S - 1, :]
        return (lg.argmax(-1) == b[:, S:]).mean(0)

    return np.asarray(jax.jit(jax.vmap(per_pos))(params, A_all))


def report(label, acc):
    m = acc.mean(0)
    per_seed = [(a > 0.99).sum() for a in acc]
    print(f"\n{label}")
    print("  output row i:  " + " ".join(f"{i:>4}" for i in range(S)))
    print("  accuracy:      " + " ".join(f"{v:4.2f}" for v in m))
    print(f"  rows solved by every seed: {int((m > 0.99).sum())}/{S}   "
          f"per-seed rows solved: {per_seed}   mean acc {m.mean():.3f}")


# 1. is it decay?
for label, horizon in (("decay: default (horizon = 32 = seq len)", None),
                       ("decay: long (horizon = 3200, decay ≈ 1)", 3200.0)):
    report(label, train(horizon, Config(1, 128, 512, 8, 16, C, S * 2)))

# 2. is it memory capacity? State is d_head x d_head per head; a linear associative memory
#    holds roughly d_head distinguishable keys, and this task needs S = 16 of them.
for dh in (64, 32, 16, 8, 4):
    report(f"heads: H = {128 // dh:>2}, d_head = {dh:>2} (state {dh}x{dh} per head)",
           train(None, Config(1, 128, 512, 128 // dh, dh, C, S * 2)))
