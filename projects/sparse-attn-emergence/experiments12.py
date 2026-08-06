"""
Sparse-attn-emergence — exp12: is the CA task in-context learning, or memorisation?

The paper draws a pool of N=256 rules ONCE PER RUN and samples one per training example. The
model therefore sees the same 256 lookup tables for all 10,000 steps — 256 x 64 entries x 2
bits is about 4 KB, trivially storable in an 800k-parameter model. What has to be inferred
from context is then only WHICH stored rule is active: an index, not a function.

That is in-context *selection*, not in-context learning, and it matters for how far the
paper's synthetic results reach. Its motivating capabilities (IOI, induction, copying) work on
tokens never seen in that arrangement before.

One parameter separates the two. Sweep the pool size:

    N = 1        no inference at all — pure memorisation of a single rule
    N = 16       tiny pool
    N = 256      the paper's setting
    N = 4096     large pool, memorisation getting expensive
    N = fresh    a new table per sequence, drawn from 4^64 — memorisation impossible

Everything else is held fixed at exp5's only learnable depth (k=1, span 3). Two diagnostics:

  * does it learn at all (final-state loss, solve rate) — if performance collapses only when
    the pool is removed, the paper's synthetic evidence is about memorisation
  * the PER-STATE loss profile — its slope across a sequence measures how much in-context
    inference is actually happening. Flat means nothing is being inferred; steep means the
    model is identifying the rule as evidence arrives.

Usage:
    SHARD=0 uv run --no-sync python .../scripts/run_experiments.py --bg --gpu 0 exp12
"""

import json
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib.metrics import time_to_emergence
from lib.models import Config, forward, init_params, n_params
from lib.tasks import ca_batch, ca_fresh_batch, ca_rule_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
SHARD = os.environ.get("SHARD")

S, T, C, W, K = 16, 16, 4, 3, 1          # k=1 — exp5's only learnable depth
POOLS = (1, 16, 256, 4096, 0)            # 0 = fresh rule per sequence
LRS = (3e-4, 1e-3)
if SMOKE:
    POOLS, LRS, T = (256, 0), (1e-3,), 4
elif SHARD is not None:
    POOLS = tuple(p for i, p in enumerate(POOLS) if i % 2 == int(SHARD))

N_LAYERS, D_MODEL, D_MLP, N_HEADS = 4, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
SEQ_LEN = S * T
BATCH = max(8, 8192 // SEQ_LEN)
WARMUP, WD = 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 100)
N_SEEDS = 2 if SMOKE else 8
SEED = 0
MAIN_THRESH = 0.95
CURVE_EVERY, CURVE_ROUND = 100, 5
PLATEAU = float(np.log(C))
CFG = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, SEQ_LEN)


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def run(pool: int, lr: float) -> dict:
    fresh = pool == 0
    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    params = jax.vmap(lambda k: init_params(k, CFG))(seed_keys)
    pools = None if fresh else jax.vmap(
        lambda k: ca_rule_pool(k, pool, C, W))(
            jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS))

    opt = optax.adamw(optax.join_schedules(
        [optax.linear_schedule(0.0, lr, WARMUP), optax.constant_schedule(lr)], [WARMUP]),
        weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def losses(p, b):
        logits = forward(p, b, CFG)[:, S - 1 : -1, :]
        tgt = b[:, S:]
        ls = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), tgt[..., None], -1).squeeze(-1)
        per_state = ls.reshape(ls.shape[0], T - 1, S).mean(axis=(0, 2))
        acc_last = (logits[:, -S:].argmax(-1) == tgt[:, -S:]).mean()
        return ls.mean(), (per_state[-1], acc_last)

    def batch_of(key, rules):
        return (ca_fresh_batch(key, BATCH, S, T, K, C, W) if fresh
                else ca_batch(key, rules, BATCH, S, T, K, C))

    def chunk_one(p, st, keys, rules):
        def body(carry, key):
            p, st = carry
            (loss, aux), g = jax.value_and_grad(losses, has_aux=True)(p, batch_of(key, rules))
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, *aux)

        (p, st), out = jax.lax.scan(body, (p, st), keys)
        return p, st, *out

    # a dummy pool keeps the vmap signature uniform across both branches
    rules_arg = pools if not fresh else jnp.zeros((N_SEEDS, 1), jnp.int32)
    chunk_fn = jax.jit(jax.vmap(chunk_one))
    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, last_c, acc_c = [], [], []
    t0 = time.perf_counter()
    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, last, acc = chunk_fn(
            params, opt_state, keys.reshape(N_SEEDS, CHUNK), rules_arg)
        loss_c.append(np.asarray(loss))
        last_c.append(np.asarray(last))
        acc_c.append(np.asarray(acc))

    loss_all = np.concatenate(loss_c, axis=1)
    loss_last = np.concatenate(last_c, axis=1)
    acc_last = np.concatenate(acc_c, axis=1)

    def per_state_loss(p, b):
        """(T-1,) mean CE for each predicted state — the in-context learning curve."""
        logits = forward(p, b, CFG)[:, S - 1 : -1, :]
        ls = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1),
                                  b[:, S:][..., None], -1).squeeze(-1)
        return ls.reshape(b.shape[0], T - 1, S).mean(axis=(0, 2))

    ek = jax.random.split(jax.random.key(12345), N_SEEDS)
    eb = jax.vmap(batch_of)(ek, rules_arg)
    per_state = np.asarray(jax.jit(jax.vmap(per_state_loss))(params, eb))

    tstar = [time_to_emergence(acc_last[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)
    slope = float(np.mean(per_state[:, 0] - per_state[:, -1]))     # in-context gain

    label = "fresh" if fresh else str(pool)
    logging.info(
        f"  N={label:>5} lr={lr:<7.0e} solve {len(emerged):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"loss_last med {np.median(loss_last[:, -1]):.4f} / plateau {PLATEAU:.3f}  "
        f"per-state {per_state.mean(0)[0]:.3f}→{per_state.mean(0)[-1]:.3f} "
        f"(gain {slope:.3f})  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp12_N{label}_lr{lr:.0e}",
        "name": f"CA pool N={label}, lr={lr:.0e}, k={K}, {N_SEEDS} seeds",
        "task": "cellular_automata", "arch": "transformer",
        "S": S, "T": T, "C": C, "W": W, "k": K, "span": 2 * K + 1,
        "n_rules": None if fresh else pool, "fresh_rule_per_sequence": fresh,
        "memorisable_bits": None if fresh else pool * (C ** W) * 2,
        "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "lr": lr, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": BATCH,
        "n_params": n_params(init_params(jax.random.key(0), CFG)),
        "time_s": round(elapsed, 1), "plateau": PLATEAU, "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss_last": np.round(loss_last[:, -1], 6).tolist(),
        "final_acc_last": np.round(acc_last[:, -1], 5).tolist(),
        "per_state_loss": np.round(per_state, 5).tolist(),
        "in_context_gain": round(slope, 5),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_loss_last": np.round(loss_last[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass

    logging.info(f"exp12 shard={SHARD}: pools {POOLS}, lrs {LRS}, k={K}")
    for pool in POOLS:
        for lr in LRS:
            label = "fresh" if pool == 0 else str(pool)
            name = f"{'smoke_' if SMOKE else ''}exp12_N{label}_lr{lr:.0e}"
            if name in done:
                logging.info(f"  {name} already done — skipping")
                continue
            try:
                append_result(run(pool, lr))
            except Exception as e:
                logging.error(f"  {name} FAILED, continuing: {type(e).__name__}: "
                              f"{str(e).splitlines()[0][:200]}")
    logging.info("exp12 shard finished")
