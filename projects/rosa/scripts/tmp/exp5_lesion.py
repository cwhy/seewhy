"""
exp5 lesion: load params_exp1_fmnist.pkl and re-evaluate on Fashion-MNIST test
set, but with ROSA's predicted-token embedding ZEROED (lesioned).

We don't even need to load the saved ROSA SAM — the model architecture only
sees ROSA's contribution via the `rosa_pred_emb` channel into the output
module. Zeroing that channel at eval is the cleanest possible lesion.

Three configurations are reported:
  A)  rosa_pred_emb = 0                 -- ROSA fully lesioned
  B)  rosa_pred_emb = codebook[0]       -- constant indicator-style embedding
  C)  rosa_pred_emb from saved ROSA     -- baseline (matches exp1_fmnist's 89.00%)

If A > C, ROSA was hurting (noise); if A ≈ C, output module was ignoring ROSA.

Usage: uv run python projects/rosa/scripts/tmp/exp5_lesion.py
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))
sys.path.insert(0, str(PROJ_DIR))

from shared_lib.datasets import load_supervised_image                 # noqa: E402
from lib.rosa_core import ROSACore                                    # noqa: E402

# Reuse experiments1.py's forward primitives by importing the module.
import experiments1 as exp1                                            # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    # Read DATASET / EXP_NAME from experiments1.py so the lesion uses the
    # SAME geometry (USE_INDICATORS, T_STREAM, LABEL_POS) as the trained model.
    DATASET     = exp1.DATASET
    EXP_NAME    = exp1.EXP_NAME
    PARAMS_PATH = PROJ_DIR / f"params_{EXP_NAME}.pkl"
    ROSA_PATH   = PROJ_DIR / f"rosa_{EXP_NAME}.json"
    logging.info(f"Lesioning {EXP_NAME} (USE_INDICATORS={exp1.USE_INDICATORS})")

    logging.info("loading data...")
    data = load_supervised_image(DATASET)
    X_test = np.array(data.X_test.reshape(data.n_test_samples, 784)) / 255.0
    y_test = np.array(data.y_test, dtype=np.int32)

    logging.info(f"loading params from {PARAMS_PATH.name}...")
    with open(PARAMS_PATH, "rb") as f:
        p_np = pickle.load(f)
    p = {k: jnp.array(v) for k, v in p_np.items()}
    logging.info(f"loaded {sum(v.size for v in p.values()):,} params")

    @jax.jit
    def forward_with_rosa_emb(p, x_batch, y_batch, rosa_pred_emb, key):
        y_oh = jax.nn.one_hot(y_batch, 10)
        z_sample = exp1.run_input_image(p, x_batch)
        z_label  = exp1.run_input_label(p, y_oh)
        c_q_sample, _ = exp1.diveq_quantize(z_sample, p["codebook"], key)
        c_q_label,  _ = exp1.diveq_quantize(z_label,  p["codebook"], key)
        stream_emb = exp1.build_stream_emb(p, c_q_sample, c_q_label)
        logits = exp1.output_module(p, stream_emb, rosa_pred_emb)
        return logits

    BATCH = 256
    T = exp1.T_STREAM
    D = exp1.TOKEN_DIM
    LABEL_POS = exp1.LABEL_POS

    def eval_with_emb_provider(emb_provider) -> float:
        """emb_provider(B, batch_streams) -> rosa_pred_emb of shape (B, T, D).
        batch_streams is a list[list[int]] OR None (means caller doesn't need it)."""
        preds_all = []
        for i in range(0, len(X_test), BATCH):
            bx = jnp.array(X_test[i:i + BATCH])
            by = jnp.array(y_test[i:i + BATCH])
            B = bx.shape[0]
            rosa_pred_emb = emb_provider(p, B, bx, by)
            logits = forward_with_rosa_emb(p, bx, by, rosa_pred_emb,
                                           jnp.zeros(2, dtype=jnp.uint32))
            digit_logits = logits[:, LABEL_POS, :10]
            preds_all.append(np.array(jnp.argmax(digit_logits, axis=-1)))
        preds = np.concatenate(preds_all)
        return float((preds == y_test).mean())

    # A: rosa_pred_emb = 0
    def emb_zero(p, B, bx, by):
        return jnp.zeros((B, T, D))

    # B: rosa_pred_emb = codebook[0] (broadcast to all positions)
    def emb_const(p, B, bx, by):
        return jnp.broadcast_to(p["codebook"][0][None, None, :], (B, T, D))

    # C: real ROSA (load SAM, run predict)
    def make_emb_real():
        logging.info(f"loading ROSA from {ROSA_PATH.name} ({ROSA_PATH.stat().st_size/1e9:.1f}GB)...")
        t0 = time.perf_counter()
        rosa = ROSACore.load(str(ROSA_PATH))
        logging.info(f"  loaded {rosa.n_states():,} SAM states in {time.perf_counter()-t0:.1f}s")

        def emb_real(p, B, bx, by):
            ids_s, ids_l = exp1.extract_ids(p, bx, by, jnp.zeros(2, dtype=jnp.uint32))
            ids_s_np = np.array(ids_s); ids_l_np = np.array(ids_l)
            streams = exp1.build_int_streams(ids_s_np, ids_l_np)
            predicted_ids_np, _ = exp1.rosa_predict_for_streams(rosa, streams)
            return p["codebook"][jnp.array(predicted_ids_np)]
        return emb_real

    logging.info("\n=== Lesion A: rosa_pred_emb = 0 ===")
    t0 = time.perf_counter()
    acc_A = eval_with_emb_provider(emb_zero)
    logging.info(f"A: test_acc = {acc_A:.4f}  ({time.perf_counter()-t0:.1f}s)")

    logging.info("\n=== Lesion B: rosa_pred_emb = codebook[0] ===")
    t0 = time.perf_counter()
    acc_B = eval_with_emb_provider(emb_const)
    logging.info(f"B: test_acc = {acc_B:.4f}  ({time.perf_counter()-t0:.1f}s)")

    logging.info("\n=== C (no lesion): real ROSA ===")
    t0 = time.perf_counter()
    emb_real = make_emb_real()
    acc_C = eval_with_emb_provider(emb_real)
    logging.info(f"C: test_acc = {acc_C:.4f}  ({time.perf_counter()-t0:.1f}s)")

    logging.info("\n--- summary ---")
    logging.info(f"A  rosa=0           {acc_A:.4f}")
    logging.info(f"B  rosa=codebook[0] {acc_B:.4f}")
    logging.info(f"C  rosa=real        {acc_C:.4f}  (should match exp1_fmnist 0.8900)")
    logging.info(f"A − C = {acc_A - acc_C:+.4f}  (positive ⇒ ROSA hurt; ≈0 ⇒ ROSA ignored)")


if __name__ == "__main__":
    main()
