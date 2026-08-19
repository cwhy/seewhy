"""One training loop, shared by every experiment.

The only thing that varies across the paper's conditions is *which parameters
receive gradients*, so that is the one knob :func:`train_task` exposes as
``mode`` — everything else (task, width, depth, seed, budget) is a plain
hyperparameter. Keeping a single loop means the "random" and "normal" numbers
in a table are never separated by an accidental difference in the harness.

Steps run inside a ``lax.scan`` chunk of length ``eval_every`` so XLA fuses the
whole chunk; only the evaluation crosses back into Python.
"""

from __future__ import annotations

import functools
import logging
import time

import jax
import jax.numpy as jnp
import optax

from .model import TRAINABLE, accuracy, forward, init_params, loss_fn, n_params, split_params

DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-3
GRAD_CLIP = 1.0


def make_optimizer(lr: float, wd: float, warmup: int = 0, steps: int | None = None):
    """AdamW with gradient clipping, behind a linear warmup and cosine decay.

    The paper's Appendix D.3 gives only "AdamW ... learning rate 1e-3 and weight
    decay 1e-3 ... clip all gradient norms at 1". Both the warmup and the decay
    are absent from it, and both turn out to be decisive for *fully trained*
    models:

        no warmup, constant lr    needle/full/1024 plateaus at 0.14 forever
        warmup, constant lr       reaches 0.92, then destabilises
        warmup + cosine decay     solves the task

    Embedding-only training succeeds under all three, which is why the omission
    is invisible from the paper's own results. The authors' released code sets
    `warmup_steps = 500` and `lr_scheduler_type = "cosine"`; we match it. See
    reports/exp1-budget.md.
    """
    if warmup and steps:
        sched = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr, warmup_steps=warmup,
            decay_steps=steps, end_value=0.0)
    elif warmup:
        sched = optax.linear_schedule(0.0, lr, warmup)
    else:
        sched = lr
    return optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                       optax.adamw(sched, weight_decay=wd))


@functools.lru_cache(maxsize=None)
def _acc_fn(n_layer: int, n_head: int):
    """Cached across calls on purpose.

    `jax.jit` keys its compilation cache on the wrapped function object, so
    building the closure inside `evaluate` would recompile the whole forward
    pass at every evaluation — which costs far more than the evaluation.
    """
    return jax.jit(lambda p, t, m: accuracy(p, t, m, n_layer=n_layer, n_head=n_head))


def evaluate(params, toks, mask, *, n_layer, n_head, batch=1024):
    """Sequence and token accuracy over a test set, in fixed-size chunks."""
    acc_fn = _acc_fn(n_layer, n_head)
    n = toks.shape[0]
    seq_hits = tok_sum = 0.0
    nb = 0
    for i in range(0, n - n % batch if n >= batch else n, batch):
        s, t = acc_fn(params, toks[i:i + batch], mask[i:i + batch])
        seq_hits += float(s); tok_sum += float(t); nb += 1
    return seq_hits / nb, tok_sum / nb


def train_task(
    task,
    *,
    mode: str,
    d: int,
    n_layer: int = 2,
    n_head: int = 8,
    seed: int = 0,
    steps: int = 10_000,
    batch: int = 1000,
    lr: float = DEFAULT_LR,
    wd: float = DEFAULT_WD,
    warmup: int = 0,
    eval_every: int = 500,
    log: bool = True,
) -> dict:
    """Train one model and return a results row (history included)."""
    assert mode in TRAINABLE, f"unknown mode {mode!r}"
    key = jax.random.key(seed)
    k_init, k_train = jax.random.split(key)

    params = init_params(k_init, vocab=task.vocab, n_ctx=task.n_ctx,
                         d=d, n_layer=n_layer, n_head=n_head)
    trainable, frozen = split_params(params, mode)
    n_train_p, n_total_p = n_params(trainable), n_params(params)

    opt = make_optimizer(lr, wd, warmup=warmup, steps=steps)
    opt_state = opt.init(trainable)

    grad_fn = jax.value_and_grad(loss_fn)

    def step(carry, k):
        tr, os_ = carry
        toks, mask = task.sample(k, batch)
        loss, g = grad_fn(tr, frozen, toks, mask, n_layer=n_layer, n_head=n_head)
        upd, os_ = opt.update(g, os_, tr)
        return (optax.apply_updates(tr, upd), os_), loss

    @jax.jit
    def run_chunk(tr, os_, k):
        (tr, os_), losses = jax.lax.scan(step, (tr, os_), jax.random.split(k, eval_every))
        return tr, os_, losses.mean()

    test_toks, test_mask = task.test_set()
    history = {"step": [], "loss": [], "test_seq_acc": [], "test_tok_acc": []}
    t0 = time.time()

    for chunk in range(steps // eval_every):
        k_train, k = jax.random.split(k_train)
        trainable, opt_state, loss = run_chunk(trainable, opt_state, k)
        seq_acc, tok_acc = evaluate({**trainable, **frozen}, test_toks, test_mask,
                                    n_layer=n_layer, n_head=n_head)
        done = (chunk + 1) * eval_every
        history["step"].append(done)
        history["loss"].append(float(loss))
        history["test_seq_acc"].append(seq_acc)
        history["test_tok_acc"].append(tok_acc)
        if log:
            logging.info(f"    step {done:>6}/{steps}  loss {float(loss):.4f}  "
                         f"seq_acc {seq_acc:.4f}  tok_acc {tok_acc:.4f}")

    elapsed = time.time() - t0
    final = {**trainable, **frozen}
    return {
        "task": task.name, "mode": mode, "d": d, "n_layer": n_layer, "n_head": n_head,
        "seed": seed, "steps": steps, "batch": batch, "lr": lr, "wd": wd,
        "warmup": warmup,
        "vocab": task.vocab, "n_ctx": task.n_ctx, "seq_len": task.seq_len,
        "chance": task.chance,
        "n_trainable_params": n_train_p, "n_params": n_total_p,
        "test_seq_acc": history["test_seq_acc"][-1],
        "test_tok_acc": history["test_tok_acc"][-1],
        "final_loss": history["loss"][-1],
        "time_s": elapsed,
        "history": history,
        "_params": final,          # stripped before the row is written to JSONL
    }


def strip_params(row: dict) -> dict:
    """Drop the parameter dict so a row can be JSON-serialised."""
    return {k: v for k, v in row.items() if not k.startswith("_")}
