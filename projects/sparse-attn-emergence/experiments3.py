"""
Sparse-attn-emergence — exp3: do more heads help, and does head dimension saturate? (H4)

The paper varies H at fixed width D=128 (head dim 128/H) and reports that more heads
consistently lower final loss — H=128 with head dim 1 still solves the linear map. But
that sweep moves two things at once: the number of independent chances to find the sparse
pattern, and the capacity of each. So this runs both legs:

  heads    D=128 fixed, H in {1..128}, d_head = 128/H   (paper's sweep: search width up,
                                                         per-head capacity down)
  headdim  H=8 fixed, d_head in {1..64}                 (capacity alone; search width fixed)

If head COUNT is what matters, leg 1 improves monotonically while leg 2 flattens after a
small d_head. If capacity were the driver, leg 1 would be non-monotone.

The config comes from exp2: the hardest cell that is not impossible (solve_rate strictly
between 0 and 1), so there is room to move in both directions. A cell at 0.0 or 1.0 would
censor the effect.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/run_experiments.py --bg exp3
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

from lib.metrics import support_iou, time_to_emergence
from lib.models import Config, forward, init_params, n_params
from lib.tasks import linear_map_batch, linear_map_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"

HEAD_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)      # leg 1: D=128 fixed, d_head = 128/H
HEAD_DIMS = (1, 2, 4, 8, 16, 32, 64)             # leg 2: H=8 fixed
if SMOKE:
    HEAD_COUNTS, HEAD_DIMS = (1, 128), (1,)

T, C = 2, 2
N_LAYERS, D_MODEL, D_MLP = 1, 128, 512
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
        if r.get("experiment") not in seen:      # dedupe per workflow
            seen.add(r.get("experiment"))
            out.append(r)
    return out


def pick_config() -> tuple[int, int, str]:
    """Hardest exp2 cell that is not impossible: solve_rate in (0,1), closest to 0.5,
    tie-broken toward larger S. Falls back to the slowest fully-solved cell."""
    rows = [r for r in read_rows()
            if str(r.get("experiment", "")).startswith("exp2_S") and r.get("steps", 0) >= 10_000]
    if not rows:
        raise SystemExit("exp3 needs exp2 results in results.jsonl — run exp2 first")

    partial = [r for r in rows if 0.0 < r["solve_rate"] < 1.0]
    if partial:
        best = min(partial, key=lambda r: (abs(r["solve_rate"] - 0.5), -r["S"]))
        return best["S"], best["s"], f"partial cell, solve_rate={best['solve_rate']:.2f}"

    solved = [r for r in rows if r.get("median_t_star")]
    if solved:
        best = max(solved, key=lambda r: r["median_t_star"])
        return best["S"], best["s"], f"no partial cell; slowest solved, median t*={best['median_t_star']:.0f}"
    best = max(rows, key=lambda r: r["S"])
    return best["S"], best["s"], "no cell emerged at all; largest S"


def run_config(S: int, s: int, n_heads: int, d_head: int, leg: str) -> dict:
    seq_len = S * T
    batch = BATCH_TOKENS // seq_len
    # H=128 first OOMed instantiating a CUDA graph for a 100-step scan, then wedged for
    # 2h in XLA compilation with command buffers disabled. A shorter scan keeps the graph
    # small enough for either path. Data order differs from the other configs as a result,
    # which is harmless — every config is an independent sample.
    chunk = 25 if n_heads >= 128 else CHUNK
    cfg = Config(N_LAYERS, D_MODEL, D_MLP, n_heads, d_head, C, seq_len)

    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    a_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda k: init_params(k, cfg))(seed_keys)
    A_all = jax.vmap(lambda k: linear_map_matrix(k, S, s))(a_keys)

    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, LR, WARMUP), optax.constant_schedule(LR)], [WARMUP])
    opt = optax.adamw(sched, weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def loss_fn(p, b):
        logits = forward(p, b, cfg)[:, S - 1 : 2 * S - 1, :]
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
    diag_fn = jax.jit(jax.vmap(lambda p, A, b: support_iou(
        forward(p, b, cfg, return_attn=True)[1][0].mean(0), A, s, S, n_heads)))

    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, acc_c = [], []
    t0 = time.perf_counter()
    for c in range(STEPS // chunk):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * chunk)
        params, opt_state, loss, acc = chunk_fn(
            params, opt_state, keys.reshape(N_SEEDS, chunk), A_all)
        loss_c.append(np.asarray(loss))
        acc_c.append(np.asarray(acc))

    loss2 = np.concatenate(loss_c, axis=1)
    acc2 = np.concatenate(acc_c, axis=1)
    eval_keys = jax.random.split(jax.random.key(12345), N_SEEDS)
    eval_b = jax.vmap(lambda k, A: linear_map_batch(k, A, batch))(eval_keys, A_all)
    iou_head, iou_row = (np.asarray(x) for x in diag_fn(params, A_all, eval_b))

    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  [{leg:>7}] H={n_heads:>3} d_head={d_head:>3}  solve {len(emerged):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"final loss2 med {np.median(loss2[:, -1]):.4f}  iou_row {iou_row.mean():.2f}  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp3_{leg}_H{n_heads}_dh{d_head}",
        "name": f"{leg} sweep H={n_heads} d_head={d_head} @ S={S} s={s}, {N_SEEDS} seeds",
        "leg": leg, "task": "linear_map", "S": S, "s": s, "T": T, "C": C,
        "per_seed_matrix": True, "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": n_heads, "d_head": d_head, "attn_width": n_heads * d_head,
        "lr": LR, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": batch, "batch_tokens": BATCH_TOKENS,
        "n_params": n_params(init_params(jax.random.key(0), cfg)),
        "time_s": round(elapsed, 1), "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss2": np.round(loss2[:, -1], 6).tolist(),
        "final_acc2": np.round(acc2[:, -1], 5).tolist(),
        "final_iou_rowbest": np.round(iou_row, 4).tolist(),
        "final_iou_headbest": np.round(iou_head, 4).tolist(),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_loss2": np.round(loss2[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = {r.get("experiment") for r in read_rows()}
    S, s, why = pick_config()
    logging.info(f"exp3 config from exp2: S={S} s={s}  ({why})")

    configs = ([("heads", h, D_MODEL // h) for h in HEAD_COUNTS]
               + [("headdim", 8, dh) for dh in HEAD_DIMS])
    logging.info(f"exp3: {len(configs)} configs x {N_SEEDS} seeds x {STEPS} steps")
    t_all = time.perf_counter()

    for leg, h, dh in configs:
        name = f"{'smoke_' if SMOKE else ''}exp3_{leg}_H{h}_dh{dh}"
        if name in done:
            logging.info(f"  {name} already done — skipping")
            continue
        # One config must not take the sweep down with it. H=128 OOMed on the first pass
        # (16 seeds x 128 heads is a 2 GB attention tensor) and killed the seven
        # head-dim configs queued behind it.
        try:
            append_result(run_config(S, s, h, dh, leg))
        except Exception as e:
            logging.error(f"  {name} FAILED, continuing: {type(e).__name__}: "
                          f"{str(e).splitlines()[0][:200]}")

    logging.info(f"exp3 finished in {time.perf_counter() - t_all:.0f}s")
