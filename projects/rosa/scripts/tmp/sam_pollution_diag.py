"""
Root-cause diagnostic for exp20b's late-epoch collapse.

For each of {exp20b (collapsed), exp20c (stable)}:
  A) Load saved params + saved ROSA SAM (the 'as-trained' state)
     and report test accuracy. Should match the run's FINAL number.
  B) Rebuild ROSA from scratch by re-encoding all 60K training samples
     with the *current* params, then commit fresh bursts. Report test
     accuracy with (current params + fresh SAM).

  A vs B tells us:
    A ≈ B  → final params are good; saved SAM isn't polluted (or pollution
            doesn't hurt).
    A < B  → final params are good; saved SAM is polluted → pollution is
            the collapse cause.
    A > B  → polluted SAM was actually helping (stale info from earlier
            epochs carried).

Usage: uv run python projects/rosa/scripts/tmp/sam_pollution_diag.py
"""
from __future__ import annotations

import importlib
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

from shared_lib.datasets import load_supervised_image  # noqa: E402
from lib.rosa_core import ROSACore                      # noqa: E402


def load_data():
    data = load_supervised_image("mnist")
    X_train = np.array(data.X.reshape(data.n_samples,           784)) / 255.0
    X_test  = np.array(data.X_test.reshape(data.n_test_samples, 784)) / 255.0
    y_train = np.array(data.y, dtype=np.int32)
    y_test  = np.array(data.y_test, dtype=np.int32)
    return X_train, y_train, X_test, y_test


def encode_all(p, exp_mod, X, batch_size=1024):
    ids_all = []
    for i in range(0, len(X), batch_size):
        bx = jnp.array(X[i:i + batch_size])
        ids = exp_mod.extract_ids_one(p, bx)
        ids_all.append(np.array(ids))
    return np.concatenate(ids_all, axis=0)


def build_fresh_sam(p, exp_mod, X_train, y_train):
    rosa = ROSACore(vocab_size=exp_mod.VOCAB_SIZE)
    print("  encoding 60K training samples to fresh streams...")
    t0 = time.perf_counter()
    ids_all = encode_all(p, exp_mod, X_train)              # (60000, 16)
    print(f"  encoded in {time.perf_counter()-t0:.1f}s")

    print("  feeding ROSA SAM (60K commit_burst calls)...")
    t0 = time.perf_counter()
    for i in range(ids_all.shape[0]):
        s = [int(t) for t in ids_all[i]]
        s.append(int(y_train[i]) + exp_mod.DIGIT_OFFSET)
        rosa.commit_burst(s)
    print(f"  SAM has {rosa.n_states():,} states ({time.perf_counter()-t0:.1f}s)")
    return rosa


def eval_with_sam(p, exp_mod, rosa, X_test, y_test, batch_size=256):
    preds_all = []
    for i in range(0, len(X_test), batch_size):
        bx = jnp.array(X_test[i:i + batch_size])
        ids = np.array(exp_mod.extract_ids_one(p, bx))
        # Walk SAM with the 16 sample tokens, take prediction at position LABEL_POS=15
        K = exp_mod.K_SAMPLE
        pred_at_label = np.zeros(ids.shape[0], dtype=np.int32)
        for j in range(ids.shape[0]):
            preds = rosa.predict_argmax_along_stream([int(t) for t in ids[j]])
            pred_at_label[j] = preds[exp_mod.LABEL_POS][0]
        rosa_emb_at_label = p["codebook"][jnp.array(pred_at_label)]
        class_embs = p["codebook"][:exp_mod.N_CLASSES_OUT]
        logits = rosa_emb_at_label @ class_embs.T
        digit_logits = logits[:, :10]
        preds_all.append(np.array(jnp.argmax(digit_logits, axis=-1)))
    preds = np.concatenate(preds_all)
    return float((preds == np.array(y_test)).mean())


def run_for(exp_name: str, X_train, y_train, X_test, y_test):
    print(f"\n========== {exp_name} ==========")
    # Each experiments20*.py module uses identical functions; just pick one.
    if exp_name == "exp20b":
        exp_mod = importlib.import_module("experiments20b")
    elif exp_name == "exp20c":
        exp_mod = importlib.import_module("experiments20c")
    else:
        raise ValueError(exp_name)

    params_path = PROJ_DIR / f"params_{exp_name}_mnist.pkl"
    rosa_path   = PROJ_DIR / f"rosa_{exp_name}_mnist.json"

    print(f"  loading params: {params_path.name}")
    with open(params_path, "rb") as f:
        p_np = pickle.load(f)
    p = {k: jnp.array(v) for k, v in p_np.items()}
    print(f"    {sum(v.size for v in p.values()):,} params")

    print(f"  loading 'as-trained' SAM: {rosa_path.name}")
    t0 = time.perf_counter()
    rosa_at_trained = ROSACore.load(str(rosa_path))
    print(f"    {rosa_at_trained.n_states():,} SAM states ({time.perf_counter()-t0:.1f}s)")

    print("\n  A) Eval with as-trained SAM ...")
    t0 = time.perf_counter()
    acc_A = eval_with_sam(p, exp_mod, rosa_at_trained, X_test, y_test)
    print(f"    test_acc = {acc_A:.4f}  ({time.perf_counter()-t0:.1f}s)")

    del rosa_at_trained  # free memory

    print("\n  B) Eval with fresh SAM (rebuilt from current params on 60K train) ...")
    rosa_fresh = build_fresh_sam(p, exp_mod, X_train, y_train)
    t0 = time.perf_counter()
    acc_B = eval_with_sam(p, exp_mod, rosa_fresh, X_test, y_test)
    print(f"    test_acc = {acc_B:.4f}  ({time.perf_counter()-t0:.1f}s)")
    print(f"    fresh SAM had {rosa_fresh.n_states():,} states "
          f"(vs as-trained ~30M — the difference is multi-epoch commits)")

    return {"as_trained": acc_A, "fresh": acc_B, "delta": acc_B - acc_A}


def main():
    X_train, y_train, X_test, y_test = load_data()
    out = {}
    for name in ["exp20c", "exp20b"]:
        out[name] = run_for(name, X_train, y_train, X_test, y_test)

    print("\n========== SUMMARY ==========")
    print(f"  {'run':<10}  {'as-trained SAM':>18}  {'fresh SAM':>12}  {'Δ (fresh − as)':>17}")
    for name, r in out.items():
        print(f"  {name:<10}  {r['as_trained']*100:>16.2f}%  {r['fresh']*100:>10.2f}%  {r['delta']*100:>+15.2f}pp")


if __name__ == "__main__":
    main()
