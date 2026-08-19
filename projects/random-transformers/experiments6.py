"""exp6 — circuit imitation (H7).

This is the paper's own falsification test for subspace selection. If a random
transformer succeeds because it can be steered into a low-dimensional subspace
where the target function already lives, then it should be able to imitate a
target computation *only when that computation itself fits in a low-dimensional
subspace*. So: build small random target transformers of varying width, and ask
a frozen student to reproduce their output distributions.

    minimise over E, U:   E_x [ KL( p_target(. | x) || p_student(. | x) ) ]

The prediction is a sharp break as the target widens — the paper reports
degradation between 12, 16 and 32 dimensions — and much less degradation for a
fully trained student, which has no such constraint.

Target models use the paper's amplified initialisation (§D.2.2): feed-forward
weights x20 and query/key/value x10 relative to the standard scheme, and an
unembedding of standard deviation 2/sqrt(width). Without it the targets are
near-uniform and every student "succeeds" trivially.

Usage:
    uv run python projects/random-transformers/scripts/run_experiments.py --bg exp6
"""

import logging
import math
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import optax

from lib.model import forward, init_params, n_params, split_params
from lib.results_io import append_result, read_rows
from lib.train import GRAD_CLIP, make_optimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

EXP = "exp6"
VOCAB = 512
SEQ_LEN = 40
TARGET_WIDTHS = (4, 8, 12, 16, 32, 64, 128)
TARGET_LAYERS, TARGET_HEADS = 3, 2
STUDENT_WIDTH, STUDENT_LAYERS, STUDENT_HEADS = 512, 3, 4
MODES = ("random", "full")
SEEDS = (0, 1, 2)
STEPS, BATCH, EVAL_EVERY = 6_000, 256, 500
LR, WD, WARMUP = 1e-3, 1e-3, 500

PROJECT = Path(__file__).parent
JSONL = PROJECT / "results.jsonl"


def init_target(key, d):
    """Paper §D.2.2: the standard initialisation with attention and feed-forward
    amplified, so targets differ from each other and from uniform.

    The paper's main text says query and key are scaled by 10 while the appendix
    writes the resulting standard deviation as 0.4, which is a factor of 20. The
    authors' code settles it — `imitation.py` does:

        u.mlp.c_proj.weight.data *= 20
        u.attn.c_attn.weight.data *= 10
        model.lm_head.weight.data *= 100 / n_embd_target**0.5

    So the QKV factor is 10, and the factor of 20 applies to the MLP *output*
    projection alone, not to both MLP matrices as the phrase "feed forward
    weights" suggests. (It also scales `c_proj.bias`, which is initialised to
    zero and so is a no-op.)
    """
    p = init_params(key, vocab=VOCAB, n_ctx=SEQ_LEN, d=d,
                    n_layer=TARGET_LAYERS, n_head=TARGET_HEADS)
    for i in range(TARGET_LAYERS):
        b = f"block{i}"
        p[f"{b}/attn/W_qkv"] = p[f"{b}/attn/W_qkv"] * 10.0
        p[f"{b}/mlp/W_proj"] = p[f"{b}/mlp/W_proj"] * 20.0
    p["U"] = p["U"] * (100.0 / math.sqrt(d))
    return p


def target_stats(p, d, key):
    """Entropy and cross-input divergence of the target, the paper's sanity check
    that the target is neither uniform nor mode-collapsed."""
    toks = jax.random.randint(key, (256, SEQ_LEN), 0, VOCAB)
    logits = forward(p, toks, n_layer=TARGET_LAYERS, n_head=TARGET_HEADS)
    logp = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), -1)
    ent = float(-(jnp.exp(logp) * logp).sum(-1).mean())
    kl = float((jnp.exp(logp[:-1]) * (logp[:-1] - logp[1:])).sum(-1).mean())
    return ent, kl


def main():
    done = {r["cell"] for r in read_rows(JSONL) if r.get("experiment") == EXP}

    for width in TARGET_WIDTHS:
        for mode in MODES:
            for seed in SEEDS:
                cell = f"circuit/{mode}/target{width}/{seed}"
                if cell in done:
                    continue
                logging.info(f"  {cell}")
                k_t, k_s, k_d, k_stat = jax.random.split(jax.random.key(7_000 + seed), 4)

                target = init_target(k_t, width)
                ent, cross_kl = target_stats(target, width, k_stat)

                student = init_params(k_s, vocab=VOCAB, n_ctx=SEQ_LEN, d=STUDENT_WIDTH,
                                      n_layer=STUDENT_LAYERS, n_head=STUDENT_HEADS)
                trainable, frozen = split_params(student, mode)
                opt = make_optimizer(LR, WD, warmup=WARMUP)
                opt_state = opt.init(trainable)

                def kl_loss(tr, fr, toks):
                    tgt_logp = jax.lax.stop_gradient(jax.nn.log_softmax(
                        forward(target, toks, n_layer=TARGET_LAYERS,
                                n_head=TARGET_HEADS).astype(jnp.float32), -1))
                    stu_logp = jax.nn.log_softmax(forward(
                        {**tr, **fr}, toks, n_layer=STUDENT_LAYERS,
                        n_head=STUDENT_HEADS).astype(jnp.float32), -1)
                    # KL(target || student), averaged over every position
                    return (jnp.exp(tgt_logp) * (tgt_logp - stu_logp)).sum(-1).mean()

                grad_fn = jax.value_and_grad(kl_loss)

                def step(carry, k):
                    tr, os_ = carry
                    toks = jax.random.randint(k, (BATCH, SEQ_LEN), 0, VOCAB)
                    loss, g = grad_fn(tr, frozen, toks)
                    upd, os_ = opt.update(g, os_, tr)
                    return (optax.apply_updates(tr, upd), os_), loss

                @jax.jit
                def run_chunk(tr, os_, k):
                    (tr, os_), losses = jax.lax.scan(
                        step, (tr, os_), jax.random.split(k, EVAL_EVERY))
                    return tr, os_, losses.mean()

                history = {"step": [], "kl": []}
                t0 = time.time()
                k = k_d
                for chunk in range(STEPS // EVAL_EVERY):
                    k, ks = jax.random.split(k)
                    trainable, opt_state, loss = run_chunk(trainable, opt_state, ks)
                    history["step"].append((chunk + 1) * EVAL_EVERY)
                    history["kl"].append(float(loss))
                    logging.info(f"    step {(chunk + 1) * EVAL_EVERY:>5}  KL {float(loss):.4f}")

                # held-out KL on fresh sequences
                eval_toks = jax.random.randint(jax.random.key(999), (2048, SEQ_LEN), 0, VOCAB)
                final_kl = float(jax.jit(kl_loss)(trainable, frozen, eval_toks))

                append_result(JSONL, {
                    "experiment": EXP, "cell": cell,
                    "name": f"circuit imitation — {mode}, target width {width}, seed {seed}",
                    "mode": mode, "target_width": width, "seed": seed,
                    "target_layers": TARGET_LAYERS, "target_heads": TARGET_HEADS,
                    "d": STUDENT_WIDTH, "n_layer": STUDENT_LAYERS, "n_head": STUDENT_HEADS,
                    "vocab": VOCAB, "seq_len": SEQ_LEN,
                    "steps": STEPS, "batch": BATCH, "lr": LR, "wd": WD,
                    "n_trainable_params": n_params(trainable), "n_params": n_params(student),
                    "target_entropy": ent, "target_cross_kl": cross_kl,
                    "final_kl": final_kl, "train_kl": history["kl"][-1],
                    "time_s": time.time() - t0, "history": history,
                })
                logging.info(f"  {cell}  KL={final_kl:.4f} (target entropy {ent:.2f})")

    logging.info("exp6 complete")


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}")
    main()
