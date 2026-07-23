"""
Mine concrete test-set examples illustrating the SAM-pollution failure.

For exp20b (final params + as-trained SAM):
  - find test samples where the polluted SAM gives the WRONG digit but the
    fresh SAM (built one-pass with the same final params) gives the RIGHT one;
  - for each, capture:
      * the test image and its 16 sample-token IDs (current encoder)
      * the rightmost-match position in each SAM's sam.text
      * which "burst" that position sits inside, and that burst's stored class
        (y_token = last token of the burst)
      * which epoch the match position belongs to (in as-trained SAM)
      * the predicted_id from each SAM and the resulting 11-class logits

Saves:
  scripts/tmp/walkthrough_data.json   — structured per-example data
  scripts/tmp/walkthrough_*.png/svg   — per-example image strips

Usage:
  uv run python projects/rosa/scripts/tmp/example_walkthrough.py
"""
from __future__ import annotations

import importlib, json, pickle, sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))
sys.path.insert(0, str(PROJ_DIR))

from shared_lib.datasets import load_supervised_image          # noqa: E402
from shared_lib.media import save_matplotlib_figure             # noqa: E402
from lib.rosa_core import ROSACore, _advance_state             # noqa: E402

OUT_DIR = PROJ_DIR / "scripts" / "tmp"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_params(exp_name: str):
    with open(PROJ_DIR / f"params_{exp_name}_mnist.pkl", "rb") as f:
        p_np = pickle.load(f)
    return {k: jnp.array(v) for k, v in p_np.items()}


def encode_all(p, exp_mod, X, batch_size=1024):
    out = []
    for i in range(0, len(X), batch_size):
        bx = jnp.array(X[i:i + batch_size])
        out.append(np.array(exp_mod.extract_ids_one(p, bx)))
    return np.concatenate(out, axis=0)


def build_fresh_sam(p, exp_mod, X_train, y_train):
    rosa = ROSACore(vocab_size=exp_mod.VOCAB_SIZE)
    ids_all = encode_all(p, exp_mod, X_train)
    for i in range(ids_all.shape[0]):
        s = [int(t) for t in ids_all[i]]
        s.append(int(y_train[i]) + exp_mod.DIGIT_OFFSET)
        rosa.commit_burst(s)
    return rosa


def rightmost_match_info(sam, prefix_tokens, T_STREAM):
    """Walk SAM with prefix; at end state, find the rightmost-match position.
    Returns (pos_in_text, match_length, predicted_next_token)."""
    b, c, d, e = sam.b, sam.c, sam.d, sam.e
    v = 0
    for tok in prefix_tokens:
        v = _advance_state(b, c, v, int(tok))
    u = v
    while u != -1:
        i = e[u]
        if d[u] > 0 and 0 <= i < len(sam.text) - 1:
            if not sam.boundary_after[i]:
                return i, d[u], int(sam.text[i + 1])
        u = c[u]
    return -1, 0, -1


def burst_of(sam, pos, T_STREAM):
    """Given a position in sam.text, return the burst (list of T_STREAM ints) it sits inside."""
    burst_start = (pos // T_STREAM) * T_STREAM
    return [int(t) for t in sam.text[burst_start:burst_start + T_STREAM]]


def epoch_of(pos, n_train_per_epoch, T_STREAM):
    """Which training epoch this position was committed in. n_train_per_epoch = 60000 for full MNIST."""
    return pos // (n_train_per_epoch * T_STREAM)


def predict_with_sam(p, exp_mod, rosa, X, batch_size=256):
    pred_ids_all, digit_preds_all, logits_all = [], [], []
    for i in range(0, len(X), batch_size):
        bx = jnp.array(X[i:i + batch_size])
        ids = np.array(exp_mod.extract_ids_one(p, bx))
        pred_at_label = np.zeros(ids.shape[0], dtype=np.int32)
        for j in range(ids.shape[0]):
            preds = rosa.predict_argmax_along_stream([int(t) for t in ids[j]])
            pred_at_label[j] = preds[exp_mod.LABEL_POS][0]
        rosa_emb = p["codebook"][jnp.array(pred_at_label)]
        class_embs = p["codebook"][:exp_mod.N_CLASSES_OUT]
        logits = np.array(rosa_emb @ class_embs.T)        # (B, 11)
        pred_ids_all.append(pred_at_label)
        digit_preds_all.append(np.argmax(logits[:, :10], axis=-1))
        logits_all.append(logits)
    return (np.concatenate(pred_ids_all),
            np.concatenate(digit_preds_all),
            np.concatenate(logits_all, axis=0))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"jax devices: {jax.devices()}")
    data = load_supervised_image("mnist")
    X_train = np.array(data.X.reshape(data.n_samples,           784)) / 255.0
    X_test  = np.array(data.X_test.reshape(data.n_test_samples, 784)) / 255.0
    y_train = np.array(data.y, dtype=np.int32)
    y_test  = np.array(data.y_test, dtype=np.int32)

    exp_mod = importlib.import_module("experiments20b")
    p = load_params("exp20b")
    print("loaded exp20b params")

    print("loading as-trained SAM...")
    t0 = time.perf_counter()
    rosa_pol = ROSACore.load(str(PROJ_DIR / "rosa_exp20b_mnist.json"))
    print(f"  {rosa_pol.n_states():,} states in {time.perf_counter()-t0:.1f}s")

    print("building fresh SAM...")
    t0 = time.perf_counter()
    rosa_fresh = build_fresh_sam(p, exp_mod, X_train, y_train)
    print(f"  {rosa_fresh.n_states():,} states in {time.perf_counter()-t0:.1f}s")

    print("predicting under both SAMs...")
    pid_p, dp_p, lo_p = predict_with_sam(p, exp_mod, rosa_pol,   X_test)
    pid_f, dp_f, lo_f = predict_with_sam(p, exp_mod, rosa_fresh, X_test)

    correct_p = (dp_p == y_test)
    correct_f = (dp_f == y_test)
    fresh_only = np.where(~correct_p & correct_f)[0]
    polluted_only = np.where(correct_p & ~correct_f)[0]
    print(f"polluted acc={correct_p.mean():.4f}  fresh acc={correct_f.mean():.4f}")
    print(f"fresh_only_correct: {len(fresh_only)}    polluted_only_correct: {len(polluted_only)}")

    # Pick examples: 3 from "fresh_only_correct" (the headline failure mode), prefer
    # digits 3, 4, 8 (the catastrophic-loss classes); 1 from "polluted_only_correct"
    # for contrast.
    n_pos_per_class = 16   # 16 first-position tokens + 1 y-token

    rng = np.random.default_rng(0)
    examples = []
    for desired_true in (3, 4, 8):
        cand = [i for i in fresh_only if y_test[i] == desired_true]
        if cand:
            examples.append(int(rng.choice(cand)))
    if polluted_only.size > 0:
        examples.append(int(rng.choice(polluted_only)))
    print(f"selected test indices: {examples}")

    # For each example, capture data
    results = []
    for idx in examples:
        bx = jnp.array(X_test[idx:idx + 1])
        ids = np.array(exp_mod.extract_ids_one(p, bx))[0]   # (16,)
        prefix = [int(t) for t in ids]

        pos_pol, ml_pol, next_pol = rightmost_match_info(rosa_pol.sam, prefix, exp_mod.T_STREAM)
        pos_fre, ml_fre, next_fre = rightmost_match_info(rosa_fresh.sam, prefix, exp_mod.T_STREAM)

        burst_pol = burst_of(rosa_pol.sam,  pos_pol, exp_mod.T_STREAM)
        burst_fre = burst_of(rosa_fresh.sam, pos_fre, exp_mod.T_STREAM)
        epoch_pol = epoch_of(pos_pol, 60_000, exp_mod.T_STREAM)
        # fresh SAM is one-pass: position // T_STREAM = training image index
        train_idx_fre = pos_fre // exp_mod.T_STREAM

        results.append({
            "test_idx":            int(idx),
            "true_digit":          int(y_test[idx]),
            "sample_tokens":       prefix,
            # polluted SAM
            "polluted": {
                "rightmost_pos":   int(pos_pol),
                "match_length":    int(ml_pol),
                "predicted_id":    int(pid_p[idx]),
                "predicted_digit": int(dp_p[idx]),
                "next_token":      int(next_pol),  # should == predicted_id
                "logits":          lo_p[idx].tolist(),
                "matched_burst":   burst_pol,
                "matched_burst_y_token":   burst_pol[-1],
                "approx_commit_epoch":     int(epoch_pol),
            },
            # fresh SAM
            "fresh": {
                "rightmost_pos":   int(pos_fre),
                "match_length":    int(ml_fre),
                "predicted_id":    int(pid_f[idx]),
                "predicted_digit": int(dp_f[idx]),
                "next_token":      int(next_fre),
                "logits":          lo_f[idx].tolist(),
                "matched_burst":   burst_fre,
                "matched_burst_y_token":   burst_fre[-1],
                "training_image_index":    int(train_idx_fre),
                "training_image_true_y":   int(y_train[train_idx_fre]),
            },
        })

    # Save JSON
    out_json = OUT_DIR / "walkthrough_data.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_json}")

    # Plot each example
    urls = []
    for i, r in enumerate(results):
        idx = r["test_idx"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

        # Panel 1: the test image, with prediction summary
        ax = axes[0]
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(
            f"test idx {idx}\n"
            f"true digit: {r['true_digit']}\n"
            f"polluted pred: {r['polluted']['predicted_digit']}    "
            f"fresh pred: {r['fresh']['predicted_digit']}",
            family="monospace", fontsize=10,
        )
        ax.axis("off")

        # Panel 2: fresh-SAM matched training image (we know its index)
        ax = axes[1]
        ti = r["fresh"]["training_image_index"]
        ax.imshow(X_train[ti].reshape(28, 28), cmap="gray")
        ax.set_title(
            f"fresh SAM rightmost match\n"
            f"→ training image #{ti}  true_y={r['fresh']['training_image_true_y']}\n"
            f"committed y_token={r['fresh']['matched_burst_y_token']}\n"
            f"match length {r['fresh']['match_length']}/16",
            family="monospace", fontsize=10,
        )
        ax.axis("off")

        # Panel 3: 11-class logits side-by-side
        ax = axes[2]
        cls = np.arange(11)
        w = 0.35
        ax.bar(cls - w/2, r["polluted"]["logits"], w, label="polluted",
               color="#dc2626", edgecolor="black", linewidth=0.3)
        ax.bar(cls + w/2, r["fresh"]["logits"], w, label="fresh",
               color="#16a34a", edgecolor="black", linewidth=0.3)
        ax.axvline(x=r["true_digit"], linestyle="--", linewidth=0.8, color="#444")
        ax.set_xticks(cls)
        ax.set_xticklabels([str(c) if c < 10 else "∅" for c in cls],
                           family="monospace", fontsize=8)
        ax.set_xlabel("class (0..9 digits, ∅ = null)", family="monospace", fontsize=9)
        ax.set_ylabel("logit = rosa_emb @ class_emb", family="monospace", fontsize=9)
        ax.set_title(
            f"polluted matched burst y_token={r['polluted']['matched_burst_y_token']}  "
            f"(epoch≈{r['polluted']['approx_commit_epoch']})",
            family="monospace", fontsize=9,
        )
        ax.legend(loc="upper right", prop={"family": "monospace", "size": 8})
        ax.grid(alpha=0.2, linewidth=0.4)

        fig.tight_layout()
        url = save_matplotlib_figure(f"rosa_walkthrough_ex{i+1}", fig, format="png", dpi=140)
        urls.append(url)
        plt.close(fig)
        print(f"  example {i+1}: {url}")

    # Also write a compact summary table
    print("\n=== summary table ===")
    print(f"{'idx':>5} {'true':>4} {'pol':>4} {'fre':>4} {'pol_pid':>7} {'fre_pid':>7} "
          f"{'pol_pos':>10} {'fre_pos':>10} {'pol_epoch':>9} {'pol_y_tok':>9} {'fre_y_tok':>9}")
    for r in results:
        pol = r["polluted"]; fre = r["fresh"]
        print(f"{r['test_idx']:>5} {r['true_digit']:>4} "
              f"{pol['predicted_digit']:>4} {fre['predicted_digit']:>4} "
              f"{pol['predicted_id']:>7} {fre['predicted_id']:>7} "
              f"{pol['rightmost_pos']:>10} {fre['rightmost_pos']:>10} "
              f"{pol['approx_commit_epoch']:>9} {pol['matched_burst_y_token']:>9} "
              f"{fre['matched_burst_y_token']:>9}")
    print("\nplot urls:")
    for u in urls:
        print("  " + u)


if __name__ == "__main__":
    main()
