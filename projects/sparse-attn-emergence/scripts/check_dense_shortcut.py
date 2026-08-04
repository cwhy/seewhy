"""Is the s=S cell solved by COPYING rather than by learning the map?

At s=S every row of A is all-ones, so all S second-half tokens equal parity(x0) — one
value repeated. A model that computes nothing can emit position S at chance and copy it
for the remaining S-1, scoring 1 - 0.5/S accuracy and leaving exactly ln2/S loss.

exp2 measured final loss2 = 0.0433 at S=16 and 0.0217 at S=32; ln2/16 = 0.04332 and
ln2/32 = 0.02166. This checks the mechanism directly: per-POSITION accuracy should be
~0.5 at the first second-half position and ~1.0 at every later one.

Usage: uv run --no-sync python projects/sparse-attn-emergence/scripts/tmp/check_dense_shortcut.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from lib.models import Config, forward, init_params      # noqa: E402
from lib.tasks import linear_map_batch, linear_map_matrix  # noqa: E402

S, C, T = 16, 2, 2
STEPS, BATCH, N_SEEDS = 600, 256, 4
CFG = Config(1, 128, 512, 8, 16, C, S * T)

A = linear_map_matrix(jax.random.key(0), S, S)            # all ones by construction
assert int(A.sum()) == S * S, "s=S should force an all-ones matrix"

keys = jax.random.split(jax.random.key(0), N_SEEDS)
params = jax.vmap(lambda k: init_params(k, CFG))(keys)
opt = optax.adamw(optax.join_schedules(
    [optax.linear_schedule(0.0, 3e-4, 200), optax.constant_schedule(3e-4)], [200]),
    weight_decay=0.01)
opt_state = jax.vmap(opt.init)(params)


def loss_fn(p, b):
    lg = forward(p, b, CFG)[:, S - 1 : 2 * S - 1, :]
    tgt = b[:, S:]
    ls = jax.nn.log_softmax(lg, -1)
    return -jnp.take_along_axis(ls, tgt[..., None], -1).squeeze(-1).mean()


def chunk(p, st, ks):
    def body(carry, key):
        p, st = carry
        b = linear_map_batch(key, A, BATCH)
        g = jax.grad(loss_fn)(p, b)
        upd, st = opt.update(g, st, p)
        return (optax.apply_updates(p, upd), st), 0.0

    (p, st), _ = jax.lax.scan(body, (p, st), ks)
    return p, st


run = jax.jit(jax.vmap(chunk))
ks = jax.random.split(jax.random.key(1), N_SEEDS * STEPS).reshape(N_SEEDS, STEPS)
params, opt_state = run(params, opt_state, ks)


def per_position(p, b):
    lg = forward(p, b, CFG)[:, S - 1 : 2 * S - 1, :]
    tgt = b[:, S:]
    acc = (lg.argmax(-1) == tgt).mean(0)                                   # (S,)
    ls = -jnp.take_along_axis(jax.nn.log_softmax(lg, -1), tgt[..., None], -1).squeeze(-1)
    return acc, ls.mean(0)


eval_b = linear_map_batch(jax.random.key(999), A, 2048)
acc, ls = (np.asarray(x) for x in jax.jit(jax.vmap(per_position, in_axes=(0, None)))(params, eval_b))

print(f"S={S}, s=S (all-ones A), {N_SEEDS} seeds, {STEPS} steps\n")
print("position:      " + " ".join(f"{S+i:>5}" for i in range(S)))
for i in range(N_SEEDS):
    print(f"seed {i} acc:    " + " ".join(f"{a:5.2f}" for a in acc[i]))
print("\nmean acc:      " + " ".join(f"{a:5.2f}" for a in acc.mean(0)))
print("mean loss:     " + " ".join(f"{v:5.2f}" for v in ls.mean(0)))
print(f"\nfirst position  acc {acc.mean(0)[0]:.3f}   (copying predicts ~0.50)")
print(f"later positions acc {acc.mean(0)[1:].mean():.3f}   (copying predicts ~1.00)")
print(f"overall         acc {acc.mean():.4f}   (copying predicts {1 - 0.5 / S:.4f})")
print(f"overall        loss {ls.mean():.4f}   (copying predicts ln2/S = {np.log(2)/S:.4f})")
