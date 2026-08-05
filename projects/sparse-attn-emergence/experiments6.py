"""
Sparse-attn-emergence — exp6: does a non-attention mixer learn what attention cannot? (H5)

The paper's most pointed claim: "MLP-Mixer learns the linear map task significantly faster
than a transformer". If sparse attention patterns are hard to FIND, then an architecture
that does not have to search — one whose position-mixing weights are learned directly by
gradient descent, with no softmax competition — should not suffer the plateau at all.

Run on the config where the transformer does WORST in exp2, so there is a gap to close.
Both arms share seeds, matrices, data order, optimiser and token budget; only the mixing
mechanism differs.

The transformer arm duplicates an exp2 cell on purpose — same seeds and matrices, so it
doubles as a reproducibility check on the sweep.

Capacity note: the causal mixer has L*L = 1024 mixing parameters against the transformer's
~65k of QKVO (see lib/models.init_mixer_params). The comparison is generous to the
transformer, so a mixer win is the stronger result.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/run_experiments.py --bg exp6
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
from lib.models import (Config, forward, forward_mixer, init_mixer_params, init_params,
                        n_params)
from lib.tasks import linear_map_batch, linear_map_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
ARMS = ("transformer", "mixer")

T, C = 2, 2
N_LAYERS, D_MODEL, D_MLP, N_HEADS = 1, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
BATCH_TOKENS = 8192
LR, WARMUP, WD = 3e-4, 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 100)
N_SEEDS = 4 if SMOKE else 16
SEED = 0
MAIN_THRESH = 0.95
CURVE_EVERY, CURVE_ROUND = 100, 5


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_rows() -> list[dict]:
    if not JSONL.exists():
        return []
    out, seen = [], set()
    for line in JSONL.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("experiment") not in seen:
            seen.add(r.get("experiment"))
            out.append(r)
    return out


def pick_configs() -> list[tuple[int, int, str]]:
    """Three regimes from exp2, because H5 is a claim about SPEED, not just feasibility.

    A first pass ran only the transformer's worst cell — where the transformer scores 0/16 —
    and the mixer scored 0/16 too. That comparison had no headroom: it could only show
    "also impossible" or "does the impossible", and neither speaks to "learns it faster".

      solvable   slowest cell the transformer still solves 16/16 — the speed test
      partial    cell nearest solve_rate 0.5 — headroom in both directions
      impossible transformer's worst cell — does the mixer break the wall at all?
    """
    rows = [r for r in read_rows()
            if str(r.get("experiment", "")).startswith("exp2_S") and r.get("steps", 0) >= 10_000
            and r["s"] < r["S"]]                     # s=S is degenerate (copying) — exclude
    if not rows:
        raise SystemExit("exp6 needs exp2 results in results.jsonl — run exp2 first")

    out = []
    solved = [r for r in rows if r["solve_rate"] == 1.0 and r.get("median_t_star")]
    if solved:
        r = max(solved, key=lambda r: r["median_t_star"])
        out.append((r["S"], r["s"], f"solvable, transformer median t*={r['median_t_star']:.0f}"))
    partial = [r for r in rows if 0.0 < r["solve_rate"] < 1.0]
    if partial:
        r = min(partial, key=lambda r: (abs(r["solve_rate"] - 0.5), -r["S"]))
        out.append((r["S"], r["s"], f"partial, transformer solve_rate={r['solve_rate']:.2f}"))
    zero = [r for r in rows if r["solve_rate"] == 0.0]
    if zero:
        r = min(zero, key=lambda r: -r["S"])
        out.append((r["S"], r["s"], "impossible for the transformer"))
    return out


def run_arm(S: int, s: int, arm: str) -> dict:
    seq_len = S * T
    batch = BATCH_TOKENS // seq_len
    cfg = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, seq_len)
    init_fn = init_params if arm == "transformer" else init_mixer_params
    fwd = forward if arm == "transformer" else forward_mixer

    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    a_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda k: init_fn(k, cfg))(seed_keys)
    A_all = jax.vmap(lambda k: linear_map_matrix(k, S, s))(a_keys)

    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, LR, WARMUP), optax.constant_schedule(LR)], [WARMUP])
    opt = optax.adamw(sched, weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def loss_fn(p, b):
        logits = fwd(p, b, cfg)[:, S - 1 : 2 * S - 1, :]
        tgt = b[:, S:]
        ls = jax.nn.log_softmax(logits, -1)
        return (-jnp.take_along_axis(ls, tgt[..., None], -1).squeeze(-1).mean(),
                (logits.argmax(-1) == tgt).mean())

    def chunk_one(p, st, keys, A):
        def body(carry, key):
            p, st = carry
            b = linear_map_batch(key, A, batch)
            (loss, acc), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, acc)

        (p, st), (loss, acc) = jax.lax.scan(body, (p, st), keys)
        return p, st, loss, acc

    chunk_fn = jax.jit(jax.vmap(chunk_one))
    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, acc_c = [], []
    t0 = time.perf_counter()

    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, acc = chunk_fn(
            params, opt_state, keys.reshape(N_SEEDS, CHUNK), A_all)
        loss_c.append(np.asarray(loss))
        acc_c.append(np.asarray(acc))

    loss2 = np.concatenate(loss_c, axis=1)
    acc2 = np.concatenate(acc_c, axis=1)
    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)
    n_p = n_params(init_fn(jax.random.key(0), cfg))

    logging.info(
        f"  [{arm:>11}] {n_p:>7,} params  solve {len(emerged):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"final loss2 med {np.median(loss2[:, -1]):.4f}  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp6_{arm}_S{S}_s{s}",
        "name": f"{arm} @ S={S} s={s}, {N_SEEDS} seeds, per-seed A",
        "arch": arm, "task": "linear_map", "S": S, "s": s, "T": T, "C": C,
        "per_seed_matrix": True, "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS if arm == "transformer" else None,
        "d_head": D_HEAD if arm == "transformer" else None,
        "lr": LR, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": batch, "batch_tokens": BATCH_TOKENS,
        "n_params": n_p, "time_s": round(elapsed, 1), "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss2": np.round(loss2[:, -1], 6).tolist(),
        "final_acc2": np.round(acc2[:, -1], 5).tolist(),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_loss2": np.round(loss2[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = {r.get("experiment") for r in read_rows()}
    for S, s, why in pick_configs():
        logging.info(f"exp6 @ S={S} s={s}  ({why})")
        for arm in ARMS:
            name = f"{'smoke_' if SMOKE else ''}exp6_{arm}_S{S}_s{s}"
            if name in done:
                logging.info(f"  {name} already done — skipping")
                continue
            append_result(run_arm(S, s, arm))
