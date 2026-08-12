"""Evaluate already-trained models outside the setting they were trained in.

No training happens here. Two transfers, both evaluation-only, because the KDA
has no length-dependent parameters — a model trained at one context size runs at
another unchanged.

  LENGTH   train short, test long (and the reverse). Asks whether the M=256
           result is a property of large contexts at inference time, or an
           artefact of what training selects for. Prediction recorded in
           reports/04: the short-trained model should NOT improve at long
           context, because it never built a prior to fall back on.

  DATASET  the scope test. The retrieval mechanism was itself learned from
           MNIST, so "it generalises" cannot mean "it is free of the training
           distribution" — the question is how coarse its dependence is. Four
           pools of increasing distance from MNIST:

             mnist     held-out MNIST digits            (the reference)
             fashion   Fashion-MNIST                    same medium, new content
             shuffled  MNIST under a FIXED pixel permutation — identical pixel
                       statistics, no spatial structure at all
             noise     blocky random fields             no relation to MNIST

           Identification accuracy is the metric that matters here: it needs no
           normaliser, so it is comparable across pools that have nothing else
           in common. Completion on `noise` is impossible by construction (the
           visible half carries no information about the hidden half), which is
           itself a useful floor to see printed.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/eval_transfer.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

from lib.core import Cfg, PIX, row_mask
from lib import evalsets
from lib.train import Run, build_pools, make_eval, evaluate, append_result, already_done
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20)
MODELS = {
    "recall_M16": "params_exp1.pkl",
    "recall_M256": "params_exp5.pkl",
    "complete_best": "params_exp2_best.pkl",
}
LENGTHS = (4, 16, 64, 256)
N_EVAL = 256


def load(fname):
    with open(PROJECT / fname, "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def synthetic_pools(seed=0):
    """The three non-MNIST pools, all (n, 784) float32 in [0, 1]."""
    rng = np.random.default_rng(seed)
    fm = load_supervised_image("fashion_mnist")
    fashion = np.asarray(fm.X_test).reshape(-1, PIX).astype(np.float32) / 255.0

    # A fixed permutation: same pixels, same marginal statistics, no spatial
    # structure. If retrieval survives this, the keys never used spatial layout.
    perm = rng.permutation(PIX)

    # Blocky low-frequency random fields — 7x7 uniform noise upsampled 4x. No
    # relation to handwriting beyond the array shape.
    low = rng.random((10000, 7, 7)).astype(np.float32)
    noise = np.kron(low, np.ones((1, 4, 4), np.float32)).reshape(-1, PIX)
    noise = (noise - noise.min(1, keepdims=True)) / \
            (noise.max(1, keepdims=True) - noise.min(1, keepdims=True) + 1e-6)
    return fashion, perm, noise


def grid(model_key, pools, mask_np, M, tag):
    """One (model, pool, M) cell. Returns the metric dict for its conditions."""
    mean_img = pools["train"].mean(0)
    ev = evalsets.build(pools, mask_np, M, 4, N_EVAL, mean_img, seed=999)
    p = load(MODELS[model_key])
    eval_fn = make_eval(Run(exp_name="", name="", M=M, cfg=CFG), jnp.array(mask_np))
    m = evaluate(eval_fn, p, ev, mask_np, mean_img, chunk=32 if M >= 256 else 128)
    out = {c: {k: v for k, v in m[c].items() if k != "preds"} for c in m}
    g = out["D_novel_absent"]["nmse"] - out["B_novel_present"]["nmse"]
    logging.info(
        f"  {model_key:<14} {tag:<9} M={M:<4} "
        f"idB={out['B_novel_present']['id_acc']:.3f}  gain={g:+.3f}  "
        f"B={out['B_novel_present']['nmse']:.3f}  D={out['D_novel_absent']['nmse']:.3f}")
    return out, g


def main():
    mask_np = row_mask(14)
    base = Run(exp_name="", name="", M=16, mask_rows=14, cfg=CFG)
    mnist_pools, _ = build_pools(base)
    fashion, perm, noise = synthetic_pools()

    # ── length transfer ──────────────────────────────────────────────────────
    exp = "transfer_length"
    if not already_done(exp):
        logging.info("LENGTH TRANSFER — same models, context sizes they never trained at")
        res = {}
        for mk in MODELS:
            res[mk] = {}
            for M in LENGTHS:
                out, g = grid(mk, mnist_pools, mask_np, M, "mnist")
                res[mk][str(M)] = {"metrics": out, "gain": g}
        append_result(dict(experiment=exp, name="length transfer, evaluation only",
                           trained_at={"recall_M16": 16, "recall_M256": 256,
                                       "complete_best": 16},
                           lengths=list(LENGTHS), n_eval=N_EVAL, time_s=0.0,
                           transfer=res))
        logging.info(f"wrote {exp}")

    # ── dataset transfer ─────────────────────────────────────────────────────
    exp = "transfer_dataset"
    if already_done(exp):
        return
    logging.info("DATASET TRANSFER — the recall mechanism outside MNIST")
    variants = {
        "mnist": mnist_pools,
        "fashion": {"train": mnist_pools["train"], "held": fashion,
                    "held_same": fashion},
        "shuffled": {"train": mnist_pools["train"],
                     "held": mnist_pools["held"][:, perm],
                     "held_same": mnist_pools["held"][:, perm]},
        "noise": {"train": mnist_pools["train"], "held": noise, "held_same": noise},
    }
    res = {}
    for mk in ("recall_M16",):
        res[mk] = {}
        for tag, pools in variants.items():
            out, g = grid(mk, pools, mask_np, 16, tag)
            res[mk][tag] = {"metrics": out, "gain": g}
    append_result(dict(experiment=exp, name="dataset transfer of the recall mechanism",
                       variants=list(variants), M=16, n_eval=N_EVAL, time_s=0.0,
                       note="conditions A/C use the MNIST train pool in every "
                            "variant; only the novel pool (B/D) changes",
                       transfer=res))
    logging.info(f"wrote {exp}")


if __name__ == "__main__":
    main()
