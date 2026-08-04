"""
Sparse-attn-emergence — exp4: is the loss jump the attention pattern being found? (H3)

exp1 showed alignment rising as loss falls. That is a correlation, and a weak one there
because its metric picked a single head. Two upgrades here:

1. DENSE, CORRECT diagnostics — every 50 steps, both aggregations (iou_row: per row take
   the best head, then average; iou_head: exp1's best-single-head) plus per-head entropy.
   iou_row is the honest one: heads specialise by row, so best-single-head understates a
   solved model (exp1 finished at 0.49-0.97 with loss already ~0).

2. A CAUSAL ABLATION — after training, zero the output projection block of the
   best-aligned head and re-measure loss. If the found pattern is what carries the
   capability, loss should collapse back toward the ln 2 plateau. Ablating the
   WORST-aligned head is the control: if that hurts just as much, the alignment metric
   is not identifying the mechanism at all.

Same config as exp1 (S=16, s=3, fixed A across seeds) so the two are directly comparable.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/run_experiments.py --bg exp4
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib.metrics import attn_entropy, support_iou, time_to_emergence
from lib.models import Config, forward, init_params, n_params
from lib.tasks import linear_map_batch, linear_map_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
EXP_NAME = "smoke_exp4" if SMOKE else "exp4"

S, SPARSITY, T, C = 16, 3, 2, 2
SEQ_LEN = S * T
MATRIX_SEED = 0                      # same A as exp1
N_LAYERS, D_MODEL, D_MLP, N_HEADS = 1, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
BATCH_TOKENS = 8192
BATCH = BATCH_TOKENS // SEQ_LEN
LR, WARMUP, WD = 3e-4, 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 50)
N_SEEDS = 4 if SMOKE else 16
SEED = 0
SNAP_EARLY = 100 if SMOKE else 200   # pre-jump for every exp1 seed (min t* was 469)
PLATEAU = float(np.log(C))
MAIN_THRESH = 0.95
CURVE_EVERY, CURVE_ROUND = 100, 5

CFG = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, SEQ_LEN)


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def loss_fn(p, batch):
    logits = forward(p, batch, CFG)[:, S - 1 : 2 * S - 1, :]
    tgt = batch[:, S:]
    ls = jax.nn.log_softmax(logits, -1)
    return (-jnp.take_along_axis(ls, tgt[..., None], -1).squeeze(-1).mean(),
            (logits.argmax(-1) == tgt).mean())


def ablate_head(p: dict, head: int) -> dict:
    """Zero the rows of Wo belonging to `head` — removes that head's contribution to the
    residual stream while leaving every other weight untouched."""
    mask = ((jnp.arange(N_HEADS * D_HEAD) // D_HEAD) != head).astype(p["l0_Wo"].dtype)
    return {**p, "l0_Wo": p["l0_Wo"] * mask[:, None]}


def main():
    A = linear_map_matrix(jax.random.key(MATRIX_SEED), S, SPARSITY)
    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    params = jax.vmap(lambda k: init_params(k, CFG))(seed_keys)
    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, LR, WARMUP), optax.constant_schedule(LR)], [WARMUP])
    opt = optax.adamw(sched, weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)
    eval_batch = linear_map_batch(jax.random.key(12345), A, BATCH)

    def chunk_one(p, st, keys):
        def body(carry, key):
            p, st = carry
            b = linear_map_batch(key, A, BATCH)
            (loss, acc), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, acc)

        (p, st), (loss, acc) = jax.lax.scan(body, (p, st), keys)
        return p, st, loss, acc

    def diag_one(p, b):
        a = forward(p, b, CFG, return_attn=True)[1][0].mean(0)          # (H, L, L)
        iou_head, iou_row = support_iou(a, A, SPARSITY, S, N_HEADS)
        ent = attn_entropy(a, S)
        return iou_head, iou_row, ent.min(), ent.mean()

    chunk_fn = jax.jit(jax.vmap(chunk_one))
    diag_fn = jax.jit(jax.vmap(diag_one, in_axes=(0, None)))
    attn_fn = jax.jit(jax.vmap(lambda p, b: forward(p, b, CFG, return_attn=True)[1][0].mean(0),
                               in_axes=(0, None)))
    # per-head IoU, for choosing ablation targets
    head_iou_fn = jax.jit(jax.vmap(lambda p, b: jnp.stack([
        support_iou(forward(p, b, CFG, return_attn=True)[1][0].mean(0)[h][None],
                    A, SPARSITY, S, 1)[0] for h in range(N_HEADS)]), in_axes=(0, None)))
    loss_at = jax.jit(jax.vmap(lambda p, b: loss_fn(p, b), in_axes=(0, None)))

    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, acc_c = [], []
    diag = {"step": [], "iou_head": [], "iou_row": [], "ent_min": [], "ent_mean": []}
    snap_early = None
    t0 = time.perf_counter()

    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, acc = chunk_fn(params, opt_state, keys.reshape(N_SEEDS, CHUNK))
        loss_c.append(np.asarray(loss))
        acc_c.append(np.asarray(acc))
        step = (c + 1) * CHUNK

        d = [np.asarray(x) for x in diag_fn(params, eval_batch)]
        diag["step"].append(step)
        for k, v in zip(("iou_head", "iou_row", "ent_min", "ent_mean"), d):
            diag[k].append(v)
        if step == SNAP_EARLY:
            snap_early = np.asarray(attn_fn(params, eval_batch))

        if step % (CHUNK * 20) == 0 or step == STEPS:
            logging.info(
                f"step {step:>6}/{STEPS}  loss2 med {np.median(loss_c[-1][:, -1]):.4f}  "
                f"iou_row {d[1].mean():.2f} (max {d[1].max():.2f})  "
                f"ent_min {d[2].min():.2f}  ({time.perf_counter() - t0:.0f}s)")

    loss2 = np.concatenate(loss_c, axis=1)
    acc2 = np.concatenate(acc_c, axis=1)
    snap_final = np.asarray(attn_fn(params, eval_batch))

    # ── causal ablation ──
    per_head = np.asarray(head_iou_fn(params, eval_batch))          # (N_SEEDS, H)
    best = per_head.argmax(1)
    worst = per_head.argmin(1)
    base_loss = np.asarray(loss_at(params, eval_batch)[0])
    abl_best = np.asarray(loss_at(jax.vmap(ablate_head)(params, jnp.array(best)), eval_batch)[0])
    abl_worst = np.asarray(loss_at(jax.vmap(ablate_head)(params, jnp.array(worst)), eval_batch)[0])

    logging.info(
        f"ablation: baseline loss2 {base_loss.mean():.4f} | "
        f"best-aligned head removed {abl_best.mean():.4f} | "
        f"worst-aligned head removed {abl_worst.mean():.4f} | plateau {PLATEAU:.4f}")
    logging.info(f"  recovered fraction of plateau: best {abl_best.mean() / PLATEAU:.2f}  "
                 f"worst {abl_worst.mean() / PLATEAU:.2f}")

    hist = {"loss2": loss2, "acc2": acc2,
            "diag_step": np.array(diag["step"]),
            **{k: np.stack(diag[k], axis=1) for k in ("iou_head", "iou_row", "ent_min", "ent_mean")},
            "snap_early": snap_early, "snap_final": snap_final,
            "A": np.asarray(A), "per_head_iou": per_head,
            "abl_best": abl_best, "abl_worst": abl_worst, "base_loss": base_loss,
            "best_head": best, "worst_head": worst}
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(hist, f)

    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    append_result({
        "experiment": EXP_NAME,
        "name": f"mechanism + ablation, S={S} s={SPARSITY}, {N_SEEDS} seeds, fixed A",
        "task": "linear_map", "S": S, "s": SPARSITY, "T": T, "C": C,
        "matrix_seed": MATRIX_SEED, "per_seed_matrix": False,
        "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "lr": LR, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": BATCH, "batch_tokens": BATCH_TOKENS,
        "n_params": n_params(init_params(jax.random.key(0), CFG)),
        "time_s": round(elapsed, 1), "plateau": PLATEAU, "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS, "t_star": tstar,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "final_loss2": np.round(loss2[:, -1], 6).tolist(),
        "ablation_baseline": np.round(base_loss, 5).tolist(),
        "ablation_best_head": np.round(abl_best, 5).tolist(),
        "ablation_worst_head": np.round(abl_worst, 5).tolist(),
        "best_head_idx": best.tolist(), "worst_head_idx": worst.tolist(),
        "per_head_iou": np.round(per_head, 4).tolist(),
        "diag_step": np.array(diag["step"]).tolist(),
        "diag_iou_row": np.round(np.stack(diag["iou_row"], axis=1), CURVE_ROUND).tolist(),
        "diag_iou_head": np.round(np.stack(diag["iou_head"], axis=1), CURVE_ROUND).tolist(),
        "diag_ent_min": np.round(np.stack(diag["ent_min"], axis=1), CURVE_ROUND).tolist(),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_loss2": np.round(loss2[:, sl], CURVE_ROUND).tolist(),
    })
    logging.info(f"{EXP_NAME} done in {elapsed:.0f}s")


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping")
        raise SystemExit(0)
    main()
