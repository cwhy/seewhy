"""A textbook LSTM, for the paper's recurrent baseline.

"Random transformers beat a fully trained LSTM on decimal addition" is one of
the paper's headline comparisons, so the baseline has to exist. It is a single
LSTM layer with an embedding in front and a projection out, fully trained —
deliberately plain, matching the paper's description ("textbook long short-term
memory with an encoding layer and an unembedding layer added").

The paper uses a higher learning rate (5e-3) for the LSTM "for faster
convergence"; that is kept.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import optax

from .train import GRAD_CLIP

LSTM_LR = 5e-3
LSTM_WD = 1e-3


def init_lstm(key, *, vocab: int, d: int):
    ks = jax.random.split(key, 4)
    glorot = lambda k, shape: jax.random.normal(k, shape) * (2.0 / sum(shape)) ** 0.5
    return {
        "E": jax.random.normal(ks[0], (vocab, d)) * 0.02,
        "W_x": glorot(ks[1], (d, 4 * d)),       # input -> (input, forget, cell, output)
        "W_h": glorot(ks[2], (d, 4 * d)),
        "b": jnp.concatenate([jnp.zeros(d), jnp.ones(d), jnp.zeros(2 * d)]),   # forget bias 1
        "U": jax.random.normal(ks[3], (vocab, d)) * 0.02,
    }


def lstm_forward(p, toks):
    """Logits at every position. Sequential in time, so `scan` over positions."""
    x = p["E"][toks]                                  # (B, L, d)
    d = p["W_h"].shape[0]
    B = toks.shape[0]

    def step(carry, x_t):
        h, c = carry
        gates = x_t @ p["W_x"] + h @ p["W_h"] + p["b"]
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), h

    init = (jnp.zeros((B, d)), jnp.zeros((B, d)))
    _, hs = jax.lax.scan(step, init, jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(hs, 0, 1) @ p["U"].T


def lstm_loss(p, toks, mask):
    logits = lstm_forward(p, toks)
    logp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    picked = jnp.take_along_axis(logp, toks[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, :-1]
    return -(picked * m).sum() / jnp.maximum(m.sum(), 1)


@jax.jit
def lstm_accuracy(p, toks, mask):
    pred = lstm_forward(p, toks)[:, :-1].argmax(-1)
    hit = (pred == toks[:, 1:]) * mask[:, :-1]
    m = mask[:, :-1]
    return (hit.sum(-1) == m.sum(-1)).mean(), hit.sum() / jnp.maximum(m.sum(), 1)


def train_lstm(task, *, d: int, seed: int, steps: int, batch: int, eval_every: int = 250,
               lr: float = LSTM_LR, wd: float = LSTM_WD) -> dict:
    key = jax.random.key(10_000 + seed)
    k_init, k_train = jax.random.split(key)
    p = init_lstm(k_init, vocab=task.vocab, d=d)
    opt = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adamw(lr, weight_decay=wd))
    opt_state = opt.init(p)
    grad_fn = jax.value_and_grad(lstm_loss)

    def step_fn(carry, k):
        p_, os_ = carry
        toks, mask = task.sample(k, batch)
        loss, g = grad_fn(p_, toks, mask)
        upd, os_ = opt.update(g, os_, p_)
        return (optax.apply_updates(p_, upd), os_), loss

    @jax.jit
    def run_chunk(p_, os_, k):
        (p_, os_), losses = jax.lax.scan(step_fn, (p_, os_), jax.random.split(k, eval_every))
        return p_, os_, losses.mean()

    test_toks, test_mask = task.test_set()
    history = {"step": [], "loss": [], "test_seq_acc": [], "test_tok_acc": []}
    t0 = time.time()

    for chunk in range(steps // eval_every):
        k_train, k = jax.random.split(k_train)
        p, opt_state, loss = run_chunk(p, opt_state, k)
        accs = [lstm_accuracy(p, test_toks[i:i + 1024], test_mask[i:i + 1024])
                for i in range(0, max(1024, test_toks.shape[0] - test_toks.shape[0] % 1024), 1024)]
        seq = float(sum(float(a[0]) for a in accs) / len(accs))
        tok = float(sum(float(a[1]) for a in accs) / len(accs))
        history["step"].append((chunk + 1) * eval_every)
        history["loss"].append(float(loss))
        history["test_seq_acc"].append(seq)
        history["test_tok_acc"].append(tok)

    return {
        "task": task.name, "mode": "lstm", "d": d, "n_layer": 1, "seed": seed,
        "steps": steps, "batch": batch, "lr": lr, "wd": wd,
        "vocab": task.vocab, "n_ctx": task.n_ctx, "seq_len": task.seq_len,
        "chance": task.chance,
        "n_trainable_params": int(sum(v.size for v in p.values())),
        "n_params": int(sum(v.size for v in p.values())),
        "test_seq_acc": history["test_seq_acc"][-1],
        "test_tok_acc": history["test_tok_acc"][-1],
        "final_loss": history["loss"][-1],
        "time_s": time.time() - t0,
        "history": history,
    }
