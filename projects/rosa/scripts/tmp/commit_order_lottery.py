"""
Test: how much of the (params, SAM) → test_acc relationship is determined
purely by COMMIT ORDER?

Method: hold params fixed (exp20b final params). Encode all 60K training
samples once (same encoder for everyone). Then for each of several
commit-orderings — sequential, reverse, and many random shuffles — build a
fresh SAM and report test accuracy + per-class accuracies + predicted-ID
for a fixed set of probe test samples.

If the spread across random orderings is wide (e.g., 10pp+), the
"rightmost-match is a commit-order lottery for ambiguous suffixes"
hypothesis is confirmed.

Comparison anchor: the as-trained SAM gives 65.30%.

Usage:
  uv run python projects/rosa/scripts/tmp/commit_order_lottery.py
"""
from __future__ import annotations

import importlib, json, pickle, sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))
sys.path.insert(0, str(PROJ_DIR))

from shared_lib.datasets import load_supervised_image          # noqa: E402
from shared_lib.media import save_matplotlib_figure             # noqa: E402
from lib.rosa_core import ROSACore                              # noqa: E402

N_SHUFFLES = 10                           # number of random orderings to test
N_PROBES   = 40                           # probe test samples to track per-ordering


# ── helpers ───────────────────────────────────────────────────────────────────

def load_params(exp):
    with open(PROJ_DIR / f"params_{exp}_mnist.pkl", "rb") as f:
        p_np = pickle.load(f)
    return {k: jnp.array(v) for k, v in p_np.items()}


def encode_all(p, exp_mod, X, batch_size=1024):
    out = []
    for i in range(0, len(X), batch_size):
        bx = jnp.array(X[i:i + batch_size])
        out.append(np.array(exp_mod.extract_ids_one(p, bx)))
    return np.concatenate(out, axis=0)


def build_sam_with_order(p_vocab_size, ids_train, y_train, order, digit_offset):
    rosa = ROSACore(vocab_size=p_vocab_size)
    for i in order:
        s = [int(t) for t in ids_train[i]]
        s.append(int(y_train[i]) + digit_offset)
        rosa.commit_burst(s)
    return rosa


def predict_with_sam(p, exp_mod, rosa, ids_test, batch_size=256):
    """ids_test: (N, K_SAMPLE) precomputed. Returns (digit_preds, predicted_ids)."""
    N = ids_test.shape[0]
    pred_at_label = np.zeros(N, dtype=np.int32)
    for j in range(N):
        preds = rosa.predict_argmax_along_stream([int(t) for t in ids_test[j]])
        pred_at_label[j] = preds[exp_mod.LABEL_POS][0]
    rosa_emb = p["codebook"][jnp.array(pred_at_label)]
    class_embs = p["codebook"][:exp_mod.N_CLASSES_OUT]
    logits = np.array(rosa_emb @ class_embs.T)
    digit_preds = np.argmax(logits[:, :10], axis=-1)
    return digit_preds, pred_at_label


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
    print("loaded params")

    print("encoding all data once (same encoder for everyone)...")
    t0 = time.perf_counter()
    ids_train = encode_all(p, exp_mod, X_train)
    ids_test  = encode_all(p, exp_mod, X_test)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Pick probe test samples: 4 from each class (so 40 total)
    rng = np.random.default_rng(0)
    probe_idx = []
    for cls in range(10):
        cands = np.where(y_test == cls)[0]
        chosen = rng.choice(cands, size=4, replace=False)
        probe_idx.extend(chosen.tolist())
    probe_idx = np.array(probe_idx)
    probe_true = y_test[probe_idx]
    print(f"selected {len(probe_idx)} probe test samples (4 per class)")
    ids_test_probes = ids_test[probe_idx]

    # Orderings to test
    orderings = [("sequential",  np.arange(60000)),
                 ("reverse",     np.arange(60000)[::-1])]
    for s in range(N_SHUFFLES):
        rng2 = np.random.default_rng(1000 + s)
        orderings.append((f"shuffle_seed_{s}", rng2.permutation(60000)))

    results = []
    for name, order in orderings:
        t0 = time.perf_counter()
        rosa = build_sam_with_order(
            exp_mod.VOCAB_SIZE, ids_train, y_train, order, exp_mod.DIGIT_OFFSET
        )
        t_build = time.perf_counter() - t0

        # Full test eval
        t0 = time.perf_counter()
        digit_preds_full, _ = predict_with_sam(p, exp_mod, rosa, ids_test)
        t_eval = time.perf_counter() - t0
        acc = float((digit_preds_full == y_test).mean())

        # Per-class accuracies
        per_class = {}
        for cls in range(10):
            mask = y_test == cls
            per_class[str(cls)] = float((digit_preds_full[mask] == cls).mean())

        # Predictions on probe set
        probe_preds = digit_preds_full[probe_idx].tolist()

        results.append({
            "ordering":      name,
            "test_acc":      acc,
            "per_class":     per_class,
            "probe_preds":   probe_preds,
            "build_s":       t_build,
            "eval_s":        t_eval,
        })
        print(f"  {name:>20s}: test_acc = {acc:.4f}   (build {t_build:.1f}s, eval {t_eval:.1f}s)")

    # ── Save ──
    out = {
        "as_trained_sam_acc": 0.6530,
        "probe_test_indices": probe_idx.tolist(),
        "probe_true_classes": probe_true.tolist(),
        "orderings":          results,
    }
    out_path = PROJ_DIR / "scripts" / "tmp" / "commit_order_lottery.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")

    # ── Summary ──
    accs = [r["test_acc"] for r in results]
    rand_accs = [r["test_acc"] for r in results if r["ordering"].startswith("shuffle")]
    print("\n=== summary ===")
    print(f"  as-trained (polluted): 0.6530")
    print(f"  sequential:           {results[0]['test_acc']:.4f}")
    print(f"  reverse:              {results[1]['test_acc']:.4f}")
    print(f"  random (n={len(rand_accs)}): mean {np.mean(rand_accs):.4f}, std {np.std(rand_accs):.4f},"
          f" min {min(rand_accs):.4f}, max {max(rand_accs):.4f}")
    print(f"  all orderings: min {min(accs):.4f}, max {max(accs):.4f}, range {max(accs)-min(accs):.4f}")

    # ── Probe-prediction stability: how often does the same probe sample
    #    get different predictions across orderings?
    probe_pred_matrix = np.array([r["probe_preds"] for r in results])    # (n_ord, n_probes)
    probe_modes = []
    for i in range(len(probe_idx)):
        # mode prediction across orderings
        u, c = np.unique(probe_pred_matrix[:, i], return_counts=True)
        probe_modes.append(int(u[c.argmax()]))
    probe_modes = np.array(probe_modes)
    n_diff_from_mode = (probe_pred_matrix != probe_modes[None, :]).sum(axis=1)
    n_diff_from_true = (probe_pred_matrix != probe_true[None, :]).sum(axis=1)
    print(f"\n  per-ordering: mean # probes where pred != mode-across-orderings: "
          f"{n_diff_from_mode.mean():.1f} / {len(probe_idx)}")
    print(f"  per-ordering: mean # probes where pred != true: "
          f"{n_diff_from_true.mean():.1f} / {len(probe_idx)}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    names = [r["ordering"] for r in results]
    cols  = ["#3b82f6" if "shuffle" in n else ("#16a34a" if n == "sequential" else "#dc2626")
             for n in names]
    ax.bar(range(len(results)), accs, color=cols, edgecolor="black", linewidth=0.4)
    ax.axhline(0.6530, linestyle="--", color="#666", linewidth=0.8, label="as-trained (polluted) = 65.30%")
    ax.axhline(np.mean(rand_accs), linestyle=":", color="#3b82f6", linewidth=0.8,
               label=f"random mean = {np.mean(rand_accs)*100:.2f}%")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, fontsize=7.5, rotation=70, ha="right", family="monospace")
    ax.set_ylabel("test accuracy", family="monospace")
    ax.set_title("Test accuracy across commit orderings\n"
                 "(same final params, same encoded data — only ordering varies)",
                 family="monospace", fontsize=10.5)
    ax.set_ylim(0.6, 0.9)
    ax.legend(loc="lower right", prop={"family": "monospace", "size": 8.5})
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)

    # Per-class spread across random orderings
    ax = axes[1]
    classes = np.arange(10)
    per_class_random = np.array([
        [r["per_class"][str(c)] for c in classes] for r in results if r["ordering"].startswith("shuffle")
    ])     # (n_random, 10)
    ax.boxplot([per_class_random[:, c] for c in classes],
               positions=classes, widths=0.6,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#dbeafe", edgecolor="black"))
    seq_pc = [results[0]["per_class"][str(c)] for c in classes]
    ax.scatter(classes, seq_pc, color="#16a34a", marker="*", s=80, zorder=10,
               label="sequential")
    rev_pc = [results[1]["per_class"][str(c)] for c in classes]
    ax.scatter(classes, rev_pc, color="#dc2626", marker="v", s=50, zorder=10,
               label="reverse")
    ax.set_xticks(classes)
    ax.set_xlabel("digit class", family="monospace")
    ax.set_ylabel("per-class accuracy", family="monospace")
    ax.set_title("Per-class accuracy spread across random orderings\n"
                 "(boxplot = random shuffles; * = sequential; ▼ = reverse)",
                 family="monospace", fontsize=10.5)
    ax.legend(loc="lower left", prop={"family": "monospace", "size": 8.5})
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)

    fig.tight_layout()
    url = save_matplotlib_figure("rosa_commit_order_lottery", fig, format="svg")
    plt.close(fig)
    print(f"\nplot: {url}")


if __name__ == "__main__":
    main()
