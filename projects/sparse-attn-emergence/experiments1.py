"""
Sparse-attn-emergence — exp1: is emergence abrupt, and is its timing seed-random? (H1)

Paper defaults for the linear map task: S=16, s=3, 1 layer, D=128, MLP 512, H=8,
10k steps. The claim under test is not "the model learns it" but the shape of HOW:
a plateau at ln 2 broken abruptly, at a step that varies wildly across seeds.

Design choice: A is FIXED across seeds (MATRIX_SEED), so difficulty is identical and
the only thing varying is init + data order. That makes any spread in time-to-emergence
attributable to the search, not to having drawn an easier matrix. exp2 varies A.

16 seeds, not the paper's 3 — H1 is a claim about a DISTRIBUTION over seeds, and a
mean curve hides it. Affordable because all seeds train simultaneously under one
jax.vmap over a leading param axis; the model is ~200k params and the sequence is
32 tokens, so one seed does not come close to filling a 4090.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/run_experiments.py --bg exp1
    SMOKE=1 uv run --no-sync python projects/sparse-attn-emergence/experiments1.py
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

from lib.models import Config, forward, init_params, n_params
from lib.tasks import linear_map_batch, linear_map_matrix
from lib.viz import save_mechanism_panel, save_seed_curves, save_tstar_hist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
EXP_NAME = "exp1_smoke" if SMOKE else "exp1"

# ── task ──
S, SPARSITY, T, C = 16, 3, 2, 2
SEQ_LEN = S * T
MATRIX_SEED = 0

# ── model (paper defaults) ──
N_LAYERS, D_MODEL, D_MLP, N_HEADS = 1, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS

# ── optimisation ──
# Batch is set from a fixed token budget so S*T*B is constant across S (paper protocol),
# which keeps exp2's sweep over S comparable at equal tokens/step.
BATCH_TOKENS = 8192
BATCH = BATCH_TOKENS // SEQ_LEN
LR, WARMUP, WD = 3e-4, 200, 0.01          # LR/warmup unspecified in the paper — ours
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 100)
N_SEEDS = 4 if SMOKE else 16
SEED = 0

# ── metrics ──
PLATEAU = float(np.log(C))                # ln 2: uniform-prediction loss = total failure
THRESHOLDS = (0.90, 0.95, 0.99)           # t* reported at three, to defuse threshold choice
MAIN_THRESH = 0.95
SMOOTH_WIN = 10                           # trailing-mean window for t* (raw acc2 is noisy)
CURVE_EVERY = 25                          # JSONL curve downsampling; pkl keeps full res
CURVE_ROUND = 5                           # float32 .tolist() is 17 digits of noise otherwise

CFG = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, SEQ_LEN)


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def loss_fn(p, batch):
    """Second-half CE only: position t predicts token t+1, targets are batch[:, S:]."""
    logits = forward(p, batch, CFG)[:, S - 1 : 2 * S - 1, :]
    tgt = batch[:, S:]
    ls = jax.nn.log_softmax(logits, -1)
    loss = -jnp.take_along_axis(ls, tgt[..., None], -1).squeeze(-1).mean()
    acc = (logits.argmax(-1) == tgt).mean()
    return loss, acc


def make_chunk_fn(opt, A):
    """jit(vmap(scan)) — CHUNK steps for every seed at once. Data is generated inside
    the scan from the step key, so nothing large is closed over or stored."""

    def step(carry, key):
        params, opt_state = carry
        batch = linear_map_batch(key, A, BATCH)
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), (loss, acc)

    def chunk_one(params, opt_state, keys):
        (params, opt_state), (loss, acc) = jax.lax.scan(step, (params, opt_state), keys)
        return params, opt_state, loss, acc

    return jax.jit(jax.vmap(chunk_one))


def make_diag_fn(A):
    """Attention diagnostics on a fixed eval batch: how close is any head to the
    ground-truth support of A, and how peaked is the most-peaked head."""
    supp = A.astype(jnp.float32)
    qpos = S - 1 + jnp.arange(S)           # query position predicting token S+i

    def diag_one(params, batch):
        _, attn = forward(params, batch, CFG, return_attn=True)
        a = attn[0].mean(0)                                    # (H, L, L), mean over batch
        aq = a[:, qpos, :]                                     # (H, S, L)
        ent = -(aq * jnp.log(aq + 1e-12)).sum(-1).mean(-1)     # (H,)
        top = jnp.argsort(-aq[:, :, :S], -1)[:, :, :SPARSITY]  # (H, S, s)
        sel = jnp.take_along_axis(jnp.broadcast_to(supp, (N_HEADS, S, S)), top, -1)
        inter = sel.sum(-1)                                    # |top-s ∩ true support|
        iou = (inter / (2 * SPARSITY - inter)).mean(-1)        # (H,) mean over rows
        return iou.max(), iou.mean(), ent.min(), ent.mean()

    return jax.jit(jax.vmap(diag_one, in_axes=(0, None)))


def time_to_emergence(acc2, thresh, win=SMOOTH_WIN):
    """First step whose trailing-mean acc2 exceeds thresh; None if it never does."""
    sm = np.convolve(acc2, np.ones(win) / win, mode="valid")
    hit = np.nonzero(sm > thresh)[0]
    return int(hit[0] + win - 1) if hit.size else None


def train():
    A = linear_map_matrix(jax.random.key(MATRIX_SEED), S, SPARSITY)
    logging.info(f"A row sums (must all be {SPARSITY}): {np.array(A.sum(1)).tolist()}")

    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    params = jax.vmap(lambda k: init_params(k, CFG))(seed_keys)
    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, LR, WARMUP), optax.constant_schedule(LR)], [WARMUP]
    )
    opt = optax.adamw(sched, weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    chunk_fn = make_chunk_fn(opt, A)
    diag_fn = make_diag_fn(A)
    eval_batch = linear_map_batch(jax.random.key(12345), A, BATCH)

    loss_c, acc_c, diag = [], [], {"step": [], "iou_max": [], "iou_mean": [],
                                   "ent_min": [], "ent_mean": []}
    base = jax.random.fold_in(jax.random.key(SEED), 1)
    t0 = time.perf_counter()

    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, acc = chunk_fn(params, opt_state, keys.reshape(N_SEEDS, CHUNK))
        loss_c.append(np.asarray(loss))     # (N_SEEDS, CHUNK)
        acc_c.append(np.asarray(acc))

        iou_max, iou_mean, ent_min, ent_mean = (np.asarray(x) for x in diag_fn(params, eval_batch))
        step = (c + 1) * CHUNK
        diag["step"].append(step)
        diag["iou_max"].append(iou_max)
        diag["iou_mean"].append(iou_mean)
        diag["ent_min"].append(ent_min)
        diag["ent_mean"].append(ent_mean)

        tail_loss, tail_acc = loss_c[-1][:, -1], acc_c[-1][:, -1]
        logging.info(
            f"step {step:>6}/{STEPS}  loss2 med {np.median(tail_loss):.4f} "
            f"min {tail_loss.min():.4f}  solved {int((tail_acc > MAIN_THRESH).sum())}/{N_SEEDS}  "
            f"iou_max {iou_max.max():.2f}  ent_min {ent_min.min():.2f}  "
            f"({time.perf_counter() - t0:.0f}s)"
        )

    hist = {
        "loss2": np.concatenate(loss_c, axis=1),        # (N_SEEDS, STEPS)
        "acc2": np.concatenate(acc_c, axis=1),
        "diag_step": np.array(diag["step"]),
        **{k: np.stack(diag[k], axis=1) for k in ("iou_max", "iou_mean", "ent_min", "ent_mean")},
    }
    return params, hist, time.perf_counter() - t0


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line).get("experiment"))
                except Exception:
                    pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping")
        raise SystemExit(0)

    logging.info(f"{EXP_NAME}: S={S} s={SPARSITY} seq={SEQ_LEN} batch={BATCH} "
                 f"seeds={N_SEEDS} steps={STEPS}")
    n_p = n_params(init_params(jax.random.key(0), CFG))
    logging.info(f"n_params (per seed) = {n_p:,}")

    params, hist, elapsed = train()

    steps_full = np.arange(1, STEPS + 1)
    tstars = {
        f"{th}": [time_to_emergence(hist["acc2"][i], th) for i in range(N_SEEDS)]
        for th in THRESHOLDS
    }
    main_t = tstars[f"{MAIN_THRESH}"]
    emerged = [t for t in main_t if t is not None]
    solve_rate = len(emerged) / N_SEEDS
    logging.info(f"solve_rate @ {MAIN_THRESH} = {solve_rate:.2f} "
                 f"({len(emerged)}/{N_SEEDS});  t* = {sorted(emerged)}")

    d = Path(__file__).parent
    with open(d / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(d / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(hist, f)

    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)
    urls = {
        "curves": save_seed_curves(
            f"sparse_attn_emergence_{EXP_NAME}_seed_curves", steps_full[sl],
            hist["loss2"][:, sl], hist["acc2"][:, sl], PLATEAU, MAIN_THRESH),
        "tstar": save_tstar_hist(
            f"sparse_attn_emergence_{EXP_NAME}_tstar", main_t, STEPS, MAIN_THRESH),
        "mechanism": save_mechanism_panel(
            f"sparse_attn_emergence_{EXP_NAME}_mechanism", hist["diag_step"],
            hist["iou_max"], hist["ent_min"],
            hist["loss2"][:, hist["diag_step"] - 1], SPARSITY),
    }
    for k, v in urls.items():
        logging.info(f"  {k} → {v}")

    append_result({
        "experiment": EXP_NAME,
        "name": f"linear map S={S} s={SPARSITY}, {N_SEEDS} seeds, fixed A",
        "task": "linear_map", "S": S, "s": SPARSITY, "T": T, "C": C,
        "matrix_seed": MATRIX_SEED, "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "lr": LR, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": BATCH, "batch_tokens": BATCH_TOKENS,
        "n_params": n_p, "time_s": round(elapsed, 1),
        "plateau": PLATEAU, "main_thresh": MAIN_THRESH,
        "solve_rate": solve_rate,
        "t_star": tstars,
        "final_loss2": hist["loss2"][:, -1].tolist(),
        "final_acc2": hist["acc2"][:, -1].tolist(),
        "final_iou_max": hist["iou_max"][:, -1].tolist(),
        "curve_step": steps_full[sl].tolist(),
        "curve_loss2": np.round(hist["loss2"][:, sl], CURVE_ROUND).tolist(),
        "curve_acc2": np.round(hist["acc2"][:, sl], CURVE_ROUND).tolist(),
        "diag_step": hist["diag_step"].tolist(),
        "diag_iou_max": np.round(hist["iou_max"], CURVE_ROUND).tolist(),
        "diag_ent_min": np.round(hist["ent_min"], CURVE_ROUND).tolist(),
        "urls": urls,
    })
    logging.info(f"{EXP_NAME} done in {elapsed:.0f}s")
