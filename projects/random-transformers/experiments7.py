"""exp7 — language modeling on TinyStories (H5).

Hypothesis (paper §5.2): a random transformer can do real language modeling,
badly but not trivially. The paper's headline is that a width-512 random model
lands near a width-32 fully trained one — a ~15x width gap — and that its
samples stay grammatical even though they wander semantically.

This is the task where subspace selection should hurt most: language modeling
needs the arbitrary word-level associations that §5.1 showed random models are
bad at storing, on top of the structure they are good at.

Sweep: widths 32-512 x {2, 4} layers x {random, full}, every run on the same
fixed token stream so the curve compares architectures rather than data.

Usage:
    uv run python projects/random-transformers/scripts/run_experiments.py --bg exp7
"""

import logging
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib.lm_data import CTX, VOCAB, load_tokens, to_contexts
from lib.model import forward, init_params, n_params, split_params
from lib.results_io import append_result, read_rows
from lib.train import GRAD_CLIP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

EXP = "exp7"
WIDTHS = (32, 64, 128, 256, 512)
DEPTHS = (2, 4)
MODES = ("random", "full")
SEEDS = (0,)
N_HEAD = 8
BATCH = 32
LR, WD = 6e-4, 0.1               # the paper's language-modeling optimiser settings
EVAL_EVERY = 500
EVAL_BATCHES = 40

PROJECT = Path(__file__).parent
JSONL = PROJECT / "results.jsonl"

PROMPT = ("Once upon a time, there was a little boy named Max. Max loved to play with "
          "his toy blocks. He had blocks of all colors and shapes. Max liked to build "
          "tall towers and big castles. One day, Max was playing with his blocks in his "
          "room. He built the tallest tower he had ever made. It was so tall that it "
          "almost touched the ceiling! Max was very proud of his tower. Suddenly,")


def lm_loss(trainable, frozen, batch, *, n_layer, n_head):
    logits = forward({**trainable, **frozen}, batch, n_layer=n_layer, n_head=n_head)
    logp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), -1)
    picked = jnp.take_along_axis(logp, batch[:, 1:, None], -1)[..., 0]
    return -picked.mean()


def generate(params, tok, prompt, *, n_layer, n_head, n_new=200, temperature=1.0, seed=0):
    """Sample a continuation. Recomputes the full prefix each step — no KV cache,
    which is fine for a couple of hundred tokens and keeps the model code simple."""
    ids = list(tok.encode(prompt).ids)[-(CTX - n_new - 1):]
    key = jax.random.key(seed)
    fwd = jax.jit(lambda p, t: forward(p, t, n_layer=n_layer, n_head=n_head)[:, -1])
    for _ in range(n_new):
        key, k = jax.random.split(key)
        logits = fwd(params, jnp.asarray(ids, dtype=jnp.int32)[None])[0] / temperature
        ids.append(int(jax.random.categorical(k, logits)))
    return tok.decode(ids)


def main():
    done = {r["cell"] for r in read_rows(JSONL) if r.get("experiment") == EXP}

    train_stream, eval_stream, tok = load_tokens()
    train_ctx = to_contexts(train_stream)
    eval_ctx = jnp.asarray(to_contexts(eval_stream)[: EVAL_BATCHES * BATCH])
    steps = train_ctx.shape[0] // BATCH
    logging.info(f"{train_ctx.shape[0]:,} train contexts -> {steps:,} steps of batch {BATCH} "
                 f"({train_ctx.shape[0] * CTX:,} tokens, one pass)")
    train_ctx = jnp.asarray(train_ctx)

    only = os.environ.get("RT_DEPTHS")
    depths = tuple(int(x) for x in only.split(",")) if only else DEPTHS

    for n_layer in depths:
        for width in WIDTHS:
            for mode in MODES:
                for seed in SEEDS:
                    cell = f"lm/{mode}/{width}/{n_layer}L/{seed}"
                    if cell in done:
                        continue
                    logging.info(f"  {cell}")
                    params = init_params(jax.random.key(seed), vocab=VOCAB, n_ctx=CTX,
                                         d=width, n_layer=n_layer, n_head=N_HEAD)
                    trainable, frozen = split_params(params, mode)
                    opt = optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                                      optax.adamw(LR, weight_decay=WD))
                    opt_state = opt.init(trainable)
                    grad_fn = jax.value_and_grad(lm_loss)

                    def step(carry, idx):
                        tr, os_ = carry
                        loss, g = grad_fn(tr, frozen, train_ctx[idx],
                                          n_layer=n_layer, n_head=N_HEAD)
                        upd, os_ = opt.update(g, os_, tr)
                        return (optax.apply_updates(tr, upd), os_), loss

                    @jax.jit
                    def run_chunk(tr, os_, idxs):
                        (tr, os_), losses = jax.lax.scan(step, (tr, os_), idxs)
                        return tr, os_, losses.mean()

                    @jax.jit
                    def eval_loss(tr):
                        ls = [lm_loss(tr, frozen, eval_ctx[i * BATCH:(i + 1) * BATCH],
                                      n_layer=n_layer, n_head=N_HEAD)
                              for i in range(EVAL_BATCHES)]
                        return jnp.mean(jnp.stack(ls))

                    order = np.random.default_rng(seed).permutation(steps * BATCH)[:steps * BATCH]
                    order = jnp.asarray(order.reshape(steps, BATCH))

                    history = {"step": [], "train_loss": [], "eval_loss": []}
                    t0 = time.time()
                    for c in range(steps // EVAL_EVERY):
                        idxs = order[c * EVAL_EVERY:(c + 1) * EVAL_EVERY]
                        trainable, opt_state, loss = run_chunk(trainable, opt_state, idxs)
                        ev = float(eval_loss(trainable))
                        history["step"].append((c + 1) * EVAL_EVERY)
                        history["train_loss"].append(float(loss))
                        history["eval_loss"].append(ev)
                        logging.info(f"    step {(c + 1) * EVAL_EVERY:>6}/{steps}  "
                                     f"train {float(loss):.4f}  eval {ev:.4f}")

                    final = {**trainable, **frozen}
                    sample = generate(final, tok, PROMPT, n_layer=n_layer, n_head=N_HEAD,
                                      n_new=180, seed=seed)

                    if width == 512:
                        with open(PROJECT / f"params_{EXP}_{mode}_{width}_{n_layer}L.pkl", "wb") as f:
                            pickle.dump(final, f)

                    append_result(JSONL, {
                        "experiment": EXP, "cell": cell,
                        "name": f"TinyStories LM — {mode} width {width} {n_layer}L",
                        "task": "language_modeling", "mode": mode, "d": width,
                        "n_layer": n_layer, "n_head": N_HEAD, "seed": seed,
                        "vocab": VOCAB, "n_ctx": CTX, "batch": BATCH, "steps": steps,
                        "lr": LR, "wd": WD,
                        "train_tokens": int(train_ctx.shape[0] * CTX),
                        "n_trainable_params": n_params(trainable), "n_params": n_params(params),
                        "eval_loss": history["eval_loss"][-1],
                        "eval_ppl": float(np.exp(history["eval_loss"][-1])),
                        "final_loss": history["train_loss"][-1],
                        "sample": sample,
                        "time_s": time.time() - t0, "history": history,
                    })
                    logging.info(f"  {cell}  eval CE={history['eval_loss'][-1]:.4f} "
                                 f"({time.time() - t0:.0f}s)")

    logging.info("exp7 complete")


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}")
    main()
