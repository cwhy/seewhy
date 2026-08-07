"""
Omniglot AR — exp6: exp3 at coarse resolution, fully observed.

exp3 fixes label routing; exp5 fixes what gets voted on. This attacks the third
quantity: how much each scored token's gradient is diluted on the way back.

The query's label token pools votes over its own C pixel tokens, so each token
receives roughly 1/C of the signal from the one scored token that depends on it.
At C = 196, with N x Q = 5 scored tokens per episode and 16 episodes per step,
the circuit has to be discovered from a very thin gradient.

Here the image is 10x10 and ALL 100 positions are observed. That cuts C by half
and, separately, removes partial observation as a confound: every drawing shows
its whole content, so nothing is hidden from the match. 1010 tokens per episode
rather than 1970, so it also runs about twice as fast.

Both changes make the task easier, deliberately — this is the "as easy as the
formulation allows" run. Watch the 1-NN floor it reports: 10x10 discards real
detail, so the floor moves and only this run's own floor is the right
comparison.

Usage:
    uv run python projects/omniglot-ar/scripts/run_experiments.py --bg --gpu 1 exp6
"""

import json
import logging
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib.baselines import nearest_neighbour
from lib.models import forward, init_params, n_params, open_accuracy, slot_accuracy
from lib.tasks import Spec, build_batch, class_index, observed_pixels
from shared_lib.datasets import load_omniglot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
EXP_NAME = "exp6"
JSONL = Path(__file__).parent / "results.jsonl"

# ── hyperparameters ──────────────────────────────────────────────────────────
SPEC = Spec(n_way=5, k_shot=1, n_query=1, n_ctx=100, img_size=10, n_bins=8, v_refs=64,
            label_field=True)
D_MODEL, N_LAYERS, HEAD_DIM = 256, 4, 32
MICRO_BATCH, ACCUM = 8, 2          # effective batch = 16 episodes
LR, SEED = 3e-4, 0
NUM_STEPS, EVAL_EVERY = 12000, 250
EVAL_BATCHES = 8                   # eval episodes = EVAL_BATCHES * MICRO_BATCH
WARMUP = 200


def loss_fn(p, pos, val, ref, tgt, is_query, lab):
    logits = forward(p, pos, val, ref, lab, SPEC, HEAD_DIM)
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, jnp.clip(tgt, 0, SPEC.n_content - 1)
    )
    loss = (ce * is_query).sum() / (is_query.sum() + 1e-6)
    return loss, (slot_accuracy(logits, tgt, is_query, SPEC),)


@partial(jax.jit, static_argnums=(0,))
def train_step(opt, p, st, pos, val, ref, tgt, is_query, lab):
    """Gradient accumulation: the leading axis of each arg is ACCUM micro-batches."""
    def micro(_, xs):
        (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(p, *xs)
        return None, (loss, aux, grad)

    _, (losses, auxes, grads) = jax.lax.scan(
        micro, None, (pos, val, ref, tgt, is_query, lab)
    )
    grad = jax.tree_util.tree_map(lambda x: x.mean(0), grads)
    updates, st = opt.update(grad, st, p)
    return optax.apply_updates(p, updates), st, losses.mean(), tuple(a.mean() for a in auxes)


@jax.jit
def eval_step(p, pos, val, ref, tgt, is_query, lab):
    logits = forward(p, pos, val, ref, lab, SPEC, HEAD_DIM)
    return (
        slot_accuracy(logits, tgt, is_query, SPEC),
        open_accuracy(logits, tgt, is_query),
    )


def evaluate(p, X, cls_idx, seed: int) -> tuple[float, float]:
    """Mean N-way and open-vocabulary accuracy over fresh episodes."""
    rng = np.random.default_rng(seed)
    slot, open_ = 0.0, 0.0
    for _ in range(EVAL_BATCHES):
        b = build_batch(rng, X, cls_idx, SPEC, MICRO_BATCH)
        s, o = eval_step(p, *(jnp.asarray(a) for a in b))
        slot += float(s)
        open_ += float(o)
    return slot / EVAL_BATCHES, open_ / EVAL_BATCHES


def nn_baseline(X, cls_idx, seed: int) -> float:
    rng = np.random.default_rng(seed)
    accs = [
        nearest_neighbour(*observed_pixels(rng, X, cls_idx, SPEC, MICRO_BATCH))
        for _ in range(EVAL_BATCHES)
    ]
    return float(np.mean(accs))


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

    logging.info("Loading Omniglot…")
    data = load_omniglot(size=SPEC.img_size, invert=True)
    X_bg = np.asarray(data.X_bg).reshape(len(data.X_bg), -1).astype(np.uint8)
    X_ev = np.asarray(data.X_ev).reshape(len(data.X_ev), -1).astype(np.uint8)
    y_bg = np.asarray(data.y_bg)
    y_ev = np.asarray(data.y_ev)
    idx_bg = class_index(y_bg, data.n_char_bg)
    idx_ev = class_index(y_ev, data.n_char_ev)

    ink_bg = float((X_bg > 0).mean())
    logging.info(
        f"background {X_bg.shape} / {data.n_char_bg} chars   "
        f"evaluation {X_ev.shape} / {data.n_char_ev} chars   ink={ink_bg:.3f}"
    )
    logging.info(
        f"{SPEC.n_way}-way {SPEC.k_shot}-shot   n_ctx={SPEC.n_ctx}/{SPEC.img_size**2} px   "
        f"tokens/episode={SPEC.n_tokens}   eff_batch={MICRO_BATCH * ACCUM}   "
        f"chance={1 / SPEC.n_way:.3f}"
    )

    nn_ev = nn_baseline(X_ev, idx_ev, 101)
    nn_bg = nn_baseline(X_bg, idx_bg, 102)
    logging.info(f"pixel 1-NN baseline — evaluation {nn_ev:.3f}   background {nn_bg:.3f}")

    p = init_params(jax.random.PRNGKey(SEED), SPEC, D_MODEL, N_LAYERS)
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, WARMUP, NUM_STEPS)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4)
    )
    st = opt.init(p)
    logging.info(f"n_params={n_params(p):,}")

    rng = np.random.default_rng(SEED)
    keys = ("step", "loss", "train_acc", "acc_ev", "acc_bg", "open_ev")
    hist: dict[str, list] = {k: [] for k in keys}
    t0 = time.perf_counter()

    for step in range(1, NUM_STEPS + 1):
        mics = [build_batch(rng, X_bg, idx_bg, SPEC, MICRO_BATCH) for _ in range(ACCUM)]
        stacked = tuple(
            jnp.asarray(np.stack([m[i] for m in mics])) for i in range(len(mics[0]))
        )
        p, st, loss, (tr_acc,) = train_step(opt, p, st, *stacked)

        if step % EVAL_EVERY == 0 or step == 1:
            acc_ev, open_ev = evaluate(p, X_ev, idx_ev, 1)   # UNSEEN characters
            acc_bg, _ = evaluate(p, X_bg, idx_bg, 2)         # seen characters
            for k, v in zip(
                keys,
                (step, float(loss), float(tr_acc), acc_ev, acc_bg, open_ev),
            ):
                hist[k].append(v)
            logging.info(
                f"step {step:5d}  loss {float(loss):.3f}  train {float(tr_acc):.3f}  |  "
                f"UNSEEN {acc_ev:.3f} (nn {nn_ev:.3f}, chance {1/SPEC.n_way:.2f})  "
                f"seen {acc_bg:.3f}  open {open_ev:.3f}  "
                f"({time.perf_counter() - t0:.0f}s)"
            )

    elapsed = time.perf_counter() - t0
    final = {k: hist[k][-1] for k in ("acc_ev", "acc_bg", "open_ev", "train_acc", "loss")}
    logging.info(f"DONE {elapsed:.0f}s  {final}")

    row = {
        "experiment": EXP_NAME,
        "name": f"label-as-field, coarse+complete: {SPEC.n_way}-way {SPEC.k_shot}-shot, "
                f"{SPEC.img_size}x{SPEC.img_size} fully observed (exp3 + coarse)",
        "time_s": elapsed,
        "n_params": n_params(p),
        **{f"spec_{k}": v for k, v in SPEC._asdict().items()},
        "n_tokens": SPEC.n_tokens,
        "chance": 1 / SPEC.n_way,
        "nn_ev": nn_ev,
        "nn_bg": nn_bg,
        "ink_frac": ink_bg,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "head_dim": HEAD_DIM,
        "eff_batch": MICRO_BATCH * ACCUM,
        "lr": LR,
        "steps": NUM_STEPS,
        **final,
        "history": hist,
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
