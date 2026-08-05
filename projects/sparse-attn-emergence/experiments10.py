"""
Sparse-attn-emergence — exp10: do more worked examples per sequence help?

The paper fixes T=2 for the linear map ("We always use C=2 and T=2") and only sweeps
trajectory length on the cellular automata task, where it reports that longer trajectories
LENGTHEN the loss plateau, non-monotonically. So for the linear map this axis is untested.

The question is sharper here than "more data". Training samples are already unlimited — fresh
sequences every step, ~2.5M per run — and hard cells sit at exactly ln 2 regardless, so the
task is not sample-limited. What T changes is how many worked examples of the SAME matrix
appear inside one sequence:

    T=2   x0 -> x1                      one transition to learn from
    T=4   x0 -> x1 -> x2 -> x3          three, in the same sequence
    T=8   ...                           seven

Two opposing effects, which is what makes it worth running:

  + more supervision per sequence, and the same row support has to be found only once to
    pay off at every later state
  - a longer sequence means more candidate positions, and the correct pattern is now at a
    SHIFTING offset (predicting x_t[i] means attending to row i's support in the previous
    state, whose absolute positions depend on t)

Tokens per step are held fixed (batch = BATCH_TOKENS / (S*T)) so every run sees the same
number of tokens — this compares sequence SHAPE, not compute.

Sharded: SHARD=0 -> s in {3, 6}, SHARD=1 -> s in {4}.

Usage:
    SHARD=0 uv run --no-sync python .../scripts/run_experiments.py --bg --gpu 0 exp10
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
from lib.tasks import linear_map_matrix, linear_map_traj_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
SHARD = os.environ.get("SHARD")

S, C = 16, 2
SPARSITIES = (3, 4, 6)
TRAJ = (2, 4, 8)
LRS = (3e-4, 1e-3)
if SMOKE:
    SPARSITIES, TRAJ, LRS = (4,), (4,), (1e-3,)
elif SHARD is not None:
    SPARSITIES = tuple(x for i, x in enumerate(SPARSITIES) if i % 2 == int(SHARD))

N_LAYERS, D_MODEL, D_MLP, N_HEADS = 1, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
BATCH_TOKENS = 8192
WARMUP, WD = 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 100)
N_SEEDS = 4 if SMOKE else 16
SEED = 0
MAIN_THRESH, EXACT = 0.95, 0.01
CURVE_EVERY, CURVE_ROUND = 100, 5
PLATEAU = float(np.log(C))


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def run(s: int, T: int, lr: float) -> dict:
    seq_len = S * T
    batch = BATCH_TOKENS // seq_len
    cfg = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, seq_len)

    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    a_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda k: init_params(k, cfg))(seed_keys)
    A_all = jax.vmap(lambda k: linear_map_matrix(k, S, s))(a_keys)

    opt = optax.adamw(optax.join_schedules(
        [optax.linear_schedule(0.0, lr, WARMUP), optax.constant_schedule(lr)], [WARMUP]),
        weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def losses(p, b):
        """Score every predicted state (positions S .. S*T-1); the first state is noise.
        Also reports accuracy on the LAST state alone, comparable across T."""
        logits = forward(p, b, cfg)[:, S - 1 : -1, :]
        tgt = b[:, S:]
        ls = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), tgt[..., None], -1).squeeze(-1)
        acc_all = (logits.argmax(-1) == tgt).mean()
        acc_last = (logits[:, -S:].argmax(-1) == tgt[:, -S:]).mean()
        return ls.mean(), (acc_all, acc_last)

    def chunk_one(p, st, keys, A):
        def body(carry, key):
            p, st = carry
            b = linear_map_traj_batch(key, A, batch, T)
            (loss, aux), g = jax.value_and_grad(losses, has_aux=True)(p, b)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, *aux)

        (p, st), out = jax.lax.scan(body, (p, st), keys)
        return p, st, *out

    chunk_fn = jax.jit(jax.vmap(chunk_one))
    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, acc_c, last_c = [], [], []
    t0 = time.perf_counter()
    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, acc, last = chunk_fn(
            params, opt_state, keys.reshape(N_SEEDS, CHUNK), A_all)
        loss_c.append(np.asarray(loss))
        acc_c.append(np.asarray(acc))
        last_c.append(np.asarray(last))

    loss2 = np.concatenate(loss_c, axis=1)
    acc2 = np.concatenate(acc_c, axis=1)
    acc_last = np.concatenate(last_c, axis=1)
    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    solved = loss2[:, -1] < EXACT
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  s={s} T={T} lr={lr:<7.0e} batch={batch:>4}  solve {len(emerged):>2}/{N_SEEDS}  "
        f"exact {int(solved.sum()):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"loss med {np.median(loss2[:, -1]):.4f}  "
        f"acc_last med {np.median(acc_last[:, -1]):.3f}  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp10_s{s}_T{T}_lr{lr:.0e}",
        "name": f"trajectory length T={T}, s={s}, lr={lr:.0e} @ S={S}, {N_SEEDS} seeds",
        "arch": "transformer", "task": "linear_map_traj",
        "S": S, "s": s, "T": T, "C": C, "per_seed_matrix": True,
        "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "lr": lr, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": batch, "batch_tokens": BATCH_TOKENS,
        "seq_len": seq_len,
        "n_params": n_params(init_params(jax.random.key(0), cfg)),
        "time_s": round(elapsed, 1), "plateau": PLATEAU, "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "exact_rate": float(solved.mean()),
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss2": np.round(loss2[:, -1], 6).tolist(),
        "final_acc_all": np.round(acc2[:, -1], 5).tolist(),
        "final_acc_last": np.round(acc_last[:, -1], 5).tolist(),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_loss2": np.round(loss2[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass

    logging.info(f"exp10 shard={SHARD}: s in {SPARSITIES}, T in {TRAJ}, lrs {LRS}")
    for s in SPARSITIES:
        for T in TRAJ:
            for lr in LRS:
                name = f"{'smoke_' if SMOKE else ''}exp10_s{s}_T{T}_lr{lr:.0e}"
                if name in done:
                    logging.info(f"  {name} already done — skipping")
                    continue
                try:
                    append_result(run(s, T, lr))
                except Exception as e:
                    logging.error(f"  {name} FAILED, continuing: {type(e).__name__}: "
                                  f"{str(e).splitlines()[0][:200]}")
    logging.info("exp10 shard finished")
