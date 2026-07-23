"""
Universal Transformer with Memory on Sudoku-Extreme (exp13).
EGGROLL: gradient-free ES with low-rank perturbations (Jordan / ES-MNIST style).
No backprop → no activation memory for K=18 backward passes → larger effective batch.
Perturbations: low-rank (outer product) for 2D+ weights, isotropic for 1D.
Pseudo-gradient fed into Adam for parameter updates.
Usage: uv run python projects/utm/scripts/run_experiments.py exp13
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

# ─── Hyperparameters ──────────────────────────────────────────────────────────
# fmt: off
D_MODEL        = 512
N_HEADS        = 8
HEAD_DIM       = D_MODEL // N_HEADS          # 64
FFN_DIM        = int(D_MODEL * 8 / 3)        # 1365
N_MEM          = 8
K_PONDER       = 18
VOCAB_SIZE     = 10
N_CELLS        = 81
SEQ_LEN        = N_CELLS + N_MEM            # 89
LR             = 1e-3                        # Adam LR for pseudo-gradient updates
N_EPOCHS       = 20                          # ES needs more epochs; each is fast
N_POP          = 32                          # perturbation population size
SIGMA          = 0.001                       # perturbation scale
RANK           = 8                           # low-rank noise rank for 2D weights
BATCH_SIZE     = 64                          # per-step mini-batch (n_pop × bs = 2048 effective)
EMA_DECAY      = 0.999
HALT_BIAS      = -3.0
N_TRAIN        = 200_000                     # subset: ES is slower per-sample than grad
N_TEST         = 20_000
SEED           = 42
EXP_NAME       = "exp13"
# fmt: on

PROJ_DIR = Path(__file__).parent
JSONL    = PROJ_DIR / "results.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

_d      = HEAD_DIM // 2
_theta  = 10000.0 ** (-jnp.arange(_d, dtype=jnp.float32) * 2.0 / HEAD_DIM)
_pos    = jnp.arange(SEQ_LEN, dtype=jnp.float32)
_COS    = jnp.cos(jnp.outer(_pos, _theta))
_SIN    = jnp.sin(jnp.outer(_pos, _theta))


# ─── Utilities ────────────────────────────────────────────────────────────────

def n_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


# ─── Model ────────────────────────────────────────────────────────────────────

def rmsnorm(x, scale, eps=1e-6):
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


def _rope(x):
    x1, x2 = x[..., :_d], x[..., _d:]
    c = _COS[:, None, :]
    s = _SIN[:, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def mha(params, x):
    Q = jnp.einsum("sd,dnh->snh", x, params["W_q"])
    K = jnp.einsum("sd,dnh->snh", x, params["W_k"])
    V = jnp.einsum("sd,dnh->snh", x, params["W_v"])
    Q = _rope(Q); K = _rope(K)
    Q = Q / (jnp.linalg.norm(Q, axis=-1, keepdims=True) + 1e-6)
    K = K / (jnp.linalg.norm(K, axis=-1, keepdims=True) + 1e-6)
    scores  = jnp.einsum("inh,jnh->nij", Q, K) / (HEAD_DIM ** 0.5)
    weights = jax.nn.softmax(scores, axis=-1)
    out     = jnp.einsum("nij,jnh->inh", weights, V).reshape(SEQ_LEN, N_HEADS * HEAD_DIM)
    return out @ params["W_o"]


def ffn(params, x):
    return (jax.nn.silu(x @ params["W_gate"]) * (x @ params["W_val"])) @ params["W_out"]


def utm_block(params, x):
    x = x + mha(params["attn"], rmsnorm(x, params["norm1"]))
    x = x + ffn(params["ffn"],  rmsnorm(x, params["norm2"]))
    return x


def halt_router(params, x):
    mem_mean = jnp.mean(x[N_CELLS:], axis=0)
    return jax.nn.sigmoid(mem_mean @ params["halt"]["W"] + params["halt"]["b"])


def forward(params, puzzle):
    x = jnp.concatenate([params["embed"][puzzle], params["mem_tokens"]], axis=0)
    halt_probs = []
    grid_outs  = []
    for _ in range(K_PONDER):
        x = utm_block(params, x)
        halt_probs.append(halt_router(params, x))
        grid_outs.append(x[:N_CELLS])
    hp       = jnp.stack(halt_probs)
    survival = jnp.concatenate([jnp.ones(1), jnp.cumprod(1 - hp[:-1])])
    w        = survival * hp
    w        = w.at[-1].set(jnp.maximum(1.0 - jnp.sum(w[:-1]), 0.0))
    out      = jnp.einsum("k,knd->nd", w, jnp.stack(grid_outs))
    out      = rmsnorm(out, params["norm_out"])
    return out @ params["head"]   # (N_CELLS, 9) — no ponder cost needed for ES


def batch_xe(params, Xb, yb):
    """Mean XE loss over a batch. No ponder penalty for ES."""
    logits = jax.vmap(forward, in_axes=(None, 0))(params, Xb)   # (B, 81, 9)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, yb - 1))


# ─── EGGROLL ES step ──────────────────────────────────────────────────────────

def make_es_step(optimizer, sigma, n_pop, rank):
    """
    Single-sided ES with low-rank perturbations.
    For 2D+ leaves: eps = (ab[:m] @ ab[m:].T).reshape(shape) * sigma  (rank-r outer product)
    For 1D leaves:  eps = normal * sigma
    Pseudo-gradient: sum_i fitness_i * eps_i / (n_pop * rank * sigma^2)
    """
    leaves_example = None  # set at first call via closure

    def sample_and_eval(pop_key, params, Xb, yb):
        leaves, treedef = jax.tree.flatten(params)
        eps_leaves = []
        for idx, v in enumerate(leaves):
            subkey = random.fold_in(pop_key, idx)
            if v.ndim >= 2:
                shape = v.shape
                m = shape[0]
                n = int(np.prod(shape[1:]))
                ab = random.normal(subkey, (m + n, rank))
                eps_leaves.append((ab[:m] @ ab[m:].T).reshape(shape) * sigma)
            else:
                eps_leaves.append(random.normal(subkey, v.shape) * sigma)
        eps = treedef.unflatten(eps_leaves)
        perturbed = jax.tree.map(lambda p, e: p + e, params, eps)
        loss = batch_xe(perturbed, Xb, yb)
        return loss, eps

    @jax.jit
    def step(params, opt_state, Xb, yb, pop_keys):
        # vmap over population
        losses, epses = jax.vmap(
            sample_and_eval, in_axes=(0, None, None, None)
        )(pop_keys, params, Xb, yb)
        # mean-baseline fitness
        fitnesses = losses - jnp.mean(losses)
        # pseudo-gradient: ES gradient estimator
        pseudo_grad = jax.tree.map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (n_pop * rank * sigma ** 2),
            epses,
        )
        updates, new_opt_state = optimizer.update(pseudo_grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, jnp.mean(jnp.abs(fitnesses))

    return step


# ─── Eval ─────────────────────────────────────────────────────────────────────

@jax.jit
def _predict_batch(params, Xb):
    logits = jax.vmap(forward, in_axes=(None, 0))(params, Xb)
    return jnp.argmax(logits, axis=-1) + 1


def eval_metrics(params, X, y, bs=256):
    n = X.shape[0]
    n_correct = 0
    n_counted = 0
    for i in range(0, n - bs + 1, bs):
        preds = _predict_batch(params, X[i : i + bs])
        n_correct += int(jnp.sum(jnp.all(preds == y[i : i + bs], axis=-1)))
        n_counted += bs
    return n_correct / n_counted if n_counted > 0 else 0.0


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_params(key):
    ks = random.split(key, 16)

    def w(k, *shape):
        return random.normal(k, shape) * 0.02

    return {
        "embed":      w(ks[0], VOCAB_SIZE, D_MODEL),
        "mem_tokens": w(ks[1], N_MEM, D_MODEL),
        "attn": {
            "W_q": w(ks[2], D_MODEL, N_HEADS, HEAD_DIM),
            "W_k": w(ks[3], D_MODEL, N_HEADS, HEAD_DIM),
            "W_v": w(ks[4], D_MODEL, N_HEADS, HEAD_DIM),
            "W_o": w(ks[5], N_HEADS * HEAD_DIM, D_MODEL),
        },
        "ffn": {
            "W_gate": w(ks[6], D_MODEL, FFN_DIM),
            "W_val":  w(ks[7], D_MODEL, FFN_DIM),
            "W_out":  w(ks[8], FFN_DIM, D_MODEL),
        },
        "norm1":    jnp.ones(D_MODEL),
        "norm2":    jnp.ones(D_MODEL),
        "norm_out": jnp.ones(D_MODEL),
        "halt": {
            "W": w(ks[9], D_MODEL),
            "b": jnp.array(HALT_BIAS),
        },
        "head": w(ks[10], D_MODEL, 9),
    }


# ─── Train ────────────────────────────────────────────────────────────────────

def train(params, X_train, y_train, X_test, y_test):
    n         = X_train.shape[0]
    n_batches = n // BATCH_SIZE
    total_steps = N_EPOCHS * n_batches

    sched     = optax.cosine_decay_schedule(LR, total_steps, alpha=3e-5 / LR)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(params)
    ema_params = params
    es_step   = make_es_step(optimizer, SIGMA, N_POP, RANK)
    key       = random.PRNGKey(SEED)

    history = {"loss": [], "exact_acc": [], "fitness": []}
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        key, sk = random.split(key)
        perm = random.permutation(sk, n)
        Xs = X_train[perm]
        ys = y_train[perm]

        acc_fit = jnp.zeros(())
        for b in range(n_batches):
            Xb = Xs[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            yb = ys[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            pop_keys = random.split(random.fold_in(key, b), N_POP)
            params, opt_state, fit = es_step(params, opt_state, Xb, yb, pop_keys)
            # EMA
            ema_params = jax.tree.map(
                lambda e, p: EMA_DECAY * e + (1.0 - EMA_DECAY) * p,
                ema_params, params,
            )
            acc_fit = acc_fit + fit

        loss_approx = float(batch_xe(ema_params, Xs[:512], ys[:512]))
        fit_m       = float(acc_fit) / n_batches
        acc         = eval_metrics(ema_params, X_test, y_test)
        elapsed     = time.time() - t0

        history["loss"].append(loss_approx)
        history["exact_acc"].append(acc)
        history["fitness"].append(fit_m)

        logging.info(
            f"Epoch {epoch+1}/{N_EPOCHS} | loss={loss_approx:.4f} fit={fit_m:.6f} "
            f"| exact_acc={acc:.4f} | {elapsed:.0f}s"
        )

    return params, ema_params, history, time.time() - t0


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(PROJ_DIR.parent.parent))
    from shared_lib.datasets import load_sudoku_extreme

    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping")
        sys.exit(0)

    logging.info(f"Loading dataset (n_tr={N_TRAIN}, n_tst={N_TEST})")
    ds = load_sudoku_extreme(n_tr=N_TRAIN, n_tst=N_TEST)
    logging.info(f"Dataset ready: {ds.n_train} train, {ds.n_test} test")

    params = init_params(random.PRNGKey(SEED))
    logging.info(f"Model params: {n_params(params):,}")

    params, ema_params, history, elapsed = train(
        params, ds.X_train, ds.y_train, ds.X_test, ds.y_test
    )

    final_acc = history["exact_acc"][-1]
    logging.info(f"Done. Final exact_acc={final_acc:.4f} in {elapsed:.0f}s")

    with open(PROJ_DIR / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(jax.tree.map(np.array, ema_params), f)
    with open(PROJ_DIR / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    append_result({
        "experiment":  EXP_NAME,
        "name":        f"UTM EGGROLL n_pop={N_POP} sigma={SIGMA} rank={RANK} bs={BATCH_SIZE} n_tr={N_TRAIN}",
        "d_model":     D_MODEL,
        "n_mem":       N_MEM,
        "k_ponder":    K_PONDER,
        "n_pop":       N_POP,
        "sigma":       SIGMA,
        "rank":        RANK,
        "lr":          LR,
        "batch_size":  BATCH_SIZE,
        "n_epochs":    N_EPOCHS,
        "n_train":     N_TRAIN,
        "n_params":    n_params(params),
        "exact_acc":   final_acc,
        "history":     history,
        "time_s":      elapsed,
    })
