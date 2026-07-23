"""
Full root-cause diagnostic for the SAM-pollution effect that caused exp20b's
late-epoch 'collapse'.

Generates a structured JSON of measurements + a set of plots.

Five analyses:
  D1. Cross-run codebook + token-assignment drift (exp20b final vs exp20c final).
  D2. Within-SAM token-distribution drift across `sam.text` chunks
      (early vs late commits in the same SAM).
  D3. As-trained-SAM vs fresh-SAM head-to-head on the test set:
      - predicted-ID agreement
      - which predictions are correct
      - confusion matrix breakdown
  D4. Where the rightmost-match lives in `sam.text` for failed vs correct
      test samples — does failure correlate with late-position matches?
  D5. Per-class accuracy under each (params, SAM) config.

Usage:
  uv run python projects/rosa/scripts/tmp/sam_diagnosis_full.py
"""
from __future__ import annotations

import importlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))
sys.path.insert(0, str(PROJ_DIR))

from shared_lib.datasets import load_supervised_image          # noqa: E402
from shared_lib.media import save_matplotlib_figure             # noqa: E402
from lib.rosa_core import ROSACore, _advance_state, _predict_deterministic   # noqa: E402

OUT_DIR = PROJ_DIR / "scripts" / "tmp"
OUT_DIR.mkdir(exist_ok=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_params(exp_name: str):
    with open(PROJ_DIR / f"params_{exp_name}_mnist.pkl", "rb") as f:
        p_np = pickle.load(f)
    return {k: jnp.array(v) for k, v in p_np.items()}


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
    ids_all = encode_all(p, exp_mod, X_train)
    for i in range(ids_all.shape[0]):
        s = [int(t) for t in ids_all[i]]
        s.append(int(y_train[i]) + exp_mod.DIGIT_OFFSET)
        rosa.commit_burst(s)
    return rosa


# ── D1. Cross-run codebook & token-assignment drift ───────────────────────────

def D1_cross_run_drift(p_b, p_c, exp_mod, X_train, y_train, n_imgs: int = 5000):
    """How much do exp20b and exp20c disagree about how to tokenize the same image?"""
    print("\n=== D1. Cross-run codebook + token-assignment drift ===")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_train), size=n_imgs, replace=False)
    X = X_train[idx]
    ids_b = encode_all(p_b, exp_mod, X)             # (n_imgs, K_SAMPLE)
    ids_c = encode_all(p_c, exp_mod, X)
    n_pos = exp_mod.K_SAMPLE

    # 1. Token disagreement at each position
    disagree = (ids_b != ids_c).mean(axis=0)        # (K_SAMPLE,) frac samples differ at each pos
    overall_disagree = (ids_b != ids_c).mean()

    # 2. Set-overlap of IDs used at each position
    overlap = []
    for t in range(n_pos):
        a, b = set(ids_b[:, t].tolist()), set(ids_c[:, t].tolist())
        ov = len(a & b) / max(1, len(a | b))
        overlap.append(ov)

    # 3. L2 distance between corresponding codebook entries
    cb_b = np.array(p_b["codebook"])
    cb_c = np.array(p_c["codebook"])
    cb_diff = np.linalg.norm(cb_b - cb_c, axis=-1)  # per entry
    cb_b_norm = np.linalg.norm(cb_b, axis=-1)
    cb_c_norm = np.linalg.norm(cb_c, axis=-1)

    out = {
        "n_imgs": n_imgs,
        "overall_token_disagreement_rate": float(overall_disagree),
        "per_position_disagreement_rate": disagree.tolist(),
        "per_position_id_set_overlap": overlap,
        "codebook_entry_l2_distance_summary": {
            "mean":   float(cb_diff.mean()),
            "median": float(np.median(cb_diff)),
            "p90":    float(np.quantile(cb_diff, 0.90)),
            "max":    float(cb_diff.max()),
            "min":    float(cb_diff.min()),
        },
        "codebook_norm_b_mean": float(cb_b_norm.mean()),
        "codebook_norm_c_mean": float(cb_c_norm.mean()),
    }
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv:.4f}")
        elif isinstance(v, list):
            print(f"  {k}: {[f'{x:.3f}' for x in v]}")
        else:
            print(f"  {k}: {v if isinstance(v, int) else f'{v:.4f}'}")
    return out, cb_diff


# ── D2. Within-SAM token drift across sam.text positions ──────────────────────

def D2_within_sam_drift(rosa, T_STREAM: int = 17, n_chunks: int = 10):
    """Split sam.text into n_chunks equal-time slices; compare token distributions
    at burst position 0 (the first sample-token after a boundary)."""
    print(f"\n=== D2. Within-SAM drift across {n_chunks} time-chunks ===")
    text = np.asarray(rosa.sam.text, dtype=np.int32)
    n = len(text)
    print(f"  sam.text length: {n:,}")

    # Approximate: bursts of length T_STREAM start at indices [0, T_STREAM, 2*T_STREAM, ...]
    n_bursts = n // T_STREAM
    bursts = text[:n_bursts * T_STREAM].reshape(n_bursts, T_STREAM)
    print(f"  n_bursts: {n_bursts:,}")

    # Compute first-position token histogram across chunks
    chunk_size = n_bursts // n_chunks
    pos = 0           # position 0 = first sample-token in each burst
    per_chunk_hist = []
    per_chunk_unique = []
    for c in range(n_chunks):
        chunk = bursts[c * chunk_size:(c + 1) * chunk_size, pos]
        cnt = Counter(chunk.tolist())
        per_chunk_hist.append(dict(cnt))
        per_chunk_unique.append(len(cnt))

    # KL divergence between chunk 0 and chunk k, on the union of seen tokens.
    def normalize_hist(hist, vocab):
        total = sum(hist.get(v, 0) for v in vocab) + 1e-6
        return [hist.get(v, 0) / total for v in vocab]

    kls_to_first = []
    kls_to_last  = []
    union_vocab = sorted(set().union(*per_chunk_hist))
    p_first = np.array(normalize_hist(per_chunk_hist[0], union_vocab)) + 1e-10
    p_last  = np.array(normalize_hist(per_chunk_hist[-1], union_vocab)) + 1e-10
    p_first /= p_first.sum()
    p_last  /= p_last.sum()
    for c in range(n_chunks):
        q = np.array(normalize_hist(per_chunk_hist[c], union_vocab)) + 1e-10
        q /= q.sum()
        kls_to_first.append(float((q * np.log(q / p_first)).sum()))
        kls_to_last.append(float((q * np.log(q / p_last)).sum()))

    out = {
        "n_chunks":        n_chunks,
        "chunk_n_bursts":  int(chunk_size),
        "per_chunk_unique_first_pos_tokens": per_chunk_unique,
        "kl_to_first_chunk":                 kls_to_first,
        "kl_to_last_chunk":                  kls_to_last,
        "union_vocab_size":                  len(union_vocab),
    }
    for k, v in out.items():
        if isinstance(v, list):
            print(f"  {k}: {[f'{x:.3f}' if isinstance(x,float) else str(x) for x in v]}")
        else:
            print(f"  {k}: {v}")
    return out


# ── D3. As-trained vs fresh SAM head-to-head on test ──────────────────────────

def predict_with_sam(p, exp_mod, rosa, X_test, batch_size: int = 256):
    """Return predicted_id_at_label (B,), predicted_digit (B,) for each test sample."""
    pred_ids_all, digit_preds_all = [], []
    for i in range(0, len(X_test), batch_size):
        bx = jnp.array(X_test[i:i + batch_size])
        ids = np.array(exp_mod.extract_ids_one(p, bx))
        pred_at_label = np.zeros(ids.shape[0], dtype=np.int32)
        for j in range(ids.shape[0]):
            preds = rosa.predict_argmax_along_stream([int(t) for t in ids[j]])
            pred_at_label[j] = preds[exp_mod.LABEL_POS][0]
        rosa_emb = p["codebook"][jnp.array(pred_at_label)]
        class_embs = p["codebook"][:exp_mod.N_CLASSES_OUT]
        logits = rosa_emb @ class_embs.T
        digit_logits = logits[:, :10]
        pred_ids_all.append(pred_at_label)
        digit_preds_all.append(np.array(jnp.argmax(digit_logits, axis=-1)))
    return np.concatenate(pred_ids_all), np.concatenate(digit_preds_all)


def D3_head_to_head(p, exp_mod, rosa_polluted, rosa_fresh, X_test, y_test):
    """Compare predictions from the two SAMs side-by-side."""
    print("\n=== D3. As-trained vs fresh SAM (test set head-to-head) ===")
    pid_pol, dpred_pol = predict_with_sam(p, exp_mod, rosa_polluted, X_test)
    pid_fre, dpred_fre = predict_with_sam(p, exp_mod, rosa_fresh,    X_test)

    correct_pol = (dpred_pol == y_test)
    correct_fre = (dpred_fre == y_test)

    out = {
        "acc_polluted":  float(correct_pol.mean()),
        "acc_fresh":     float(correct_fre.mean()),
        # Joint outcomes
        "both_correct":           int(((correct_pol) & (correct_fre)).sum()),
        "both_wrong":             int(((~correct_pol) & (~correct_fre)).sum()),
        "polluted_only_correct":  int((correct_pol & ~correct_fre).sum()),
        "fresh_only_correct":     int((~correct_pol & correct_fre).sum()),
        # ID-level agreement
        "predicted_id_agreement_rate":  float((pid_pol == pid_fre).mean()),
        "digit_pred_agreement_rate":    float((dpred_pol == dpred_fre).mean()),
        # Whether the polluted ID is even in the digit-class range (IDs 0..9)
        "polluted_pid_in_digit_range":  float(((pid_pol >= 0) & (pid_pol < 10)).mean()),
        "fresh_pid_in_digit_range":     float(((pid_fre >= 0) & (pid_fre < 10)).mean()),
        # Whether the polluted ID is null (10)
        "polluted_pid_is_null":         float((pid_pol == 10).mean()),
        "fresh_pid_is_null":            float((pid_fre == 10).mean()),
    }
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return out, (pid_pol, dpred_pol, pid_fre, dpred_fre)


# ── D4. Rightmost-match position analysis ─────────────────────────────────────

def rightmost_match_position(sam, prefix_tokens, T_STREAM):
    """Walk SAM with prefix; at end state, find the rightmost-match position
    in sam.text via the deterministic rule. Returns (position_in_text, length_of_match)."""
    b, c, d, e = sam.b, sam.c, sam.d, sam.e
    v = 0
    for tok in prefix_tokens:
        v = _advance_state(b, c, v, int(tok))
    # Walk suffix-links collecting rightmost end positions
    u = v
    while u != -1:
        i = e[u]
        # The deterministic rule respects boundaries
        if d[u] > 0 and 0 <= i < len(sam.text) - 1:
            if not sam.boundary_after[i]:
                return i, d[u]
        u = c[u]
    return -1, 0


def D4_rightmost_match_positions(rosa_polluted, p, exp_mod, X_test, y_test,
                                  correct_mask, n_samples: int = 500):
    """For sampled correct and failed test samples, find where the rightmost
    match is in sam.text. Compare distributions."""
    print(f"\n=== D4. Rightmost-match positions for {n_samples} correct + {n_samples} failed test samples ===")
    text_len = len(rosa_polluted.sam.text)
    print(f"  sam.text length: {text_len:,}")

    failed_idx  = np.where(~correct_mask)[0]
    correct_idx = np.where(correct_mask)[0]
    rng = np.random.default_rng(0)
    failed_idx  = rng.choice(failed_idx,  size=min(n_samples, len(failed_idx)),  replace=False)
    correct_idx = rng.choice(correct_idx, size=min(n_samples, len(correct_idx)), replace=False)

    def positions_for(idxs):
        positions, match_lens = [], []
        for i in idxs:
            bx = jnp.array(X_test[i:i+1])
            ids = np.array(exp_mod.extract_ids_one(p, bx))[0]
            pos, ml = rightmost_match_position(rosa_polluted.sam, [int(t) for t in ids],
                                               exp_mod.T_STREAM)
            positions.append(pos)
            match_lens.append(ml)
        return np.array(positions), np.array(match_lens)

    pos_fail, ml_fail = positions_for(failed_idx)
    pos_corr, ml_corr = positions_for(correct_idx)

    out = {
        "text_len":                                text_len,
        "failed_mean_position_fraction":           float(pos_fail.mean() / text_len),
        "correct_mean_position_fraction":          float(pos_corr.mean() / text_len),
        "failed_median_position_fraction":         float(np.median(pos_fail) / text_len),
        "correct_median_position_fraction":        float(np.median(pos_corr) / text_len),
        "failed_match_length_mean":                float(ml_fail.mean()),
        "correct_match_length_mean":               float(ml_corr.mean()),
        "failed_match_length_median":              float(np.median(ml_fail)),
        "correct_match_length_median":             float(np.median(ml_corr)),
        # Fraction of matches in last 10% of text
        "failed_in_last_decile":                   float((pos_fail > 0.9 * text_len).mean()),
        "correct_in_last_decile":                  float((pos_corr > 0.9 * text_len).mean()),
    }
    for k, v in out.items():
        print(f"  {k}: {v if isinstance(v, int) else f'{v:.4f}'}")
    return out, (pos_fail, ml_fail, pos_corr, ml_corr)


# ── D5. Per-class accuracy comparison ─────────────────────────────────────────

def D5_per_class(dpred_pol, dpred_fre, y_test):
    print("\n=== D5. Per-class accuracy (polluted vs fresh) ===")
    out = {"polluted": {}, "fresh": {}, "delta_polluted_minus_fresh": {}}
    for cls in range(10):
        mask = y_test == cls
        if mask.sum() == 0:
            continue
        a_pol = float((dpred_pol[mask] == cls).mean())
        a_fre = float((dpred_fre[mask] == cls).mean())
        out["polluted"][str(cls)] = a_pol
        out["fresh"][str(cls)]    = a_fre
        out["delta_polluted_minus_fresh"][str(cls)] = a_pol - a_fre
        print(f"  digit {cls}: pol={a_pol*100:5.1f}%  fre={a_fre*100:5.1f}%  Δ={(a_pol-a_fre)*100:+.1f}pp")
    return out


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_codebook_drift(cb_diff_b_vs_c, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cb_diff_b_vs_c, bins=50, color="#3b82f6", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("‖codebook_b[i] − codebook_c[i]‖₂", family="monospace")
    ax.set_ylabel("count of codebook entries", family="monospace")
    ax.set_title("Codebook-entry distance: exp20b vs exp20c (same arch, different LR schedule)",
                 family="monospace", fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_kl_curve(d2, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    n = d2["n_chunks"]
    xs = np.arange(n)
    ax.plot(xs, d2["kl_to_first_chunk"], "o-", label="KL(chunk_k || chunk_0)", color="#dc2626")
    ax.plot(xs, d2["kl_to_last_chunk"],  "o-", label="KL(chunk_k || chunk_last)", color="#16a34a")
    ax.set_xlabel("time-chunk index (0=earliest commits, last=latest)", family="monospace")
    ax.set_ylabel("KL divergence (nats)", family="monospace")
    ax.set_title("Within-SAM token-distribution drift across training time",
                 family="monospace", fontsize=10.5)
    ax.legend(loc="upper center", prop={"family": "monospace", "size": 9})
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_rightmost_positions(d4_data, out_path: Path):
    pos_fail, ml_fail, pos_corr, ml_corr = d4_data
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist([pos_fail, pos_corr], bins=30,
                  label=["failed", "correct"],
                  color=["#dc2626", "#16a34a"], alpha=0.7)
    axes[0].set_xlabel("rightmost-match position in sam.text", family="monospace")
    axes[0].set_ylabel("count", family="monospace")
    axes[0].set_title("Where in sam.text the rightmost match lives",
                      family="monospace", fontsize=10)
    axes[0].legend(prop={"family": "monospace", "size": 9})
    axes[0].grid(alpha=0.2, linewidth=0.5)

    axes[1].hist([ml_fail, ml_corr], bins=30,
                  label=["failed", "correct"],
                  color=["#dc2626", "#16a34a"], alpha=0.7)
    axes[1].set_xlabel("matched suffix length d[u]", family="monospace")
    axes[1].set_ylabel("count", family="monospace")
    axes[1].set_title("Length of the matched suffix",
                      family="monospace", fontsize=10)
    axes[1].legend(prop={"family": "monospace", "size": 9})
    axes[1].grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"jax devices: {jax.devices()}")
    X_train, y_train, X_test, y_test = load_data()

    exp_mod = importlib.import_module("experiments20b")

    print("\nLoading exp20b params + as-trained SAM...")
    p_b = load_params("exp20b")
    t0 = time.perf_counter()
    rosa_b_pol = ROSACore.load(str(PROJ_DIR / "rosa_exp20b_mnist.json"))
    print(f"  {rosa_b_pol.n_states():,} states in {time.perf_counter()-t0:.1f}s")

    print("\nLoading exp20c params...")
    p_c = load_params("exp20c")

    # D1 — cross-run drift (no SAM needed; just compare params)
    d1, cb_diff = D1_cross_run_drift(p_b, p_c, exp_mod, X_train, y_train)

    # D2 — within-SAM drift in exp20b (the polluted SAM)
    d2 = D2_within_sam_drift(rosa_b_pol, T_STREAM=exp_mod.T_STREAM)

    # D3 — head-to-head: exp20b params + polluted SAM vs + fresh SAM
    print("\nBuilding fresh SAM for exp20b params (1 pass)...")
    t0 = time.perf_counter()
    rosa_b_fresh = build_fresh_sam(p_b, exp_mod, X_train, y_train)
    print(f"  fresh SAM has {rosa_b_fresh.n_states():,} states ({time.perf_counter()-t0:.1f}s)")
    d3, (pid_pol, dpred_pol, pid_fre, dpred_fre) = D3_head_to_head(
        p_b, exp_mod, rosa_b_pol, rosa_b_fresh, X_test, y_test
    )

    # D4 — rightmost-match positions for failed vs correct (using polluted SAM)
    correct_mask_pol = (dpred_pol == y_test)
    d4, d4_data = D4_rightmost_match_positions(
        rosa_b_pol, p_b, exp_mod, X_test, y_test, correct_mask_pol
    )

    # D5 — per-class breakdown
    d5 = D5_per_class(dpred_pol, dpred_fre, y_test)

    # ── Plots ──
    fig1 = plot_codebook_drift(cb_diff, OUT_DIR / "diag_codebook_drift.svg")
    url1 = save_matplotlib_figure("rosa_diag_codebook_drift", fig1, format="svg")
    plt.close(fig1)

    fig2 = plot_kl_curve(d2, OUT_DIR / "diag_kl_curve.svg")
    url2 = save_matplotlib_figure("rosa_diag_kl_curve", fig2, format="svg")
    plt.close(fig2)

    fig3 = plot_rightmost_positions(d4_data, OUT_DIR / "diag_rightmost.svg")
    url3 = save_matplotlib_figure("rosa_diag_rightmost", fig3, format="svg")
    plt.close(fig3)

    out_full = {
        "plots": {"cb_drift_svg": url1, "kl_curve_svg": url2, "rightmost_svg": url3},
        "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5,
    }
    out_path = OUT_DIR / "sam_diagnosis_full.json"
    with open(out_path, "w") as f:
        json.dump(out_full, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Plot URLs:\n  cb_drift:  {url1}\n  kl_curve:  {url2}\n  rightmost: {url3}")


if __name__ == "__main__":
    main()
