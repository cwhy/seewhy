"""
Pareto tuning round 12: deeper bilinear readout (M^depth) and multi-query.

Building on exp16's finding: h_yhat = M @ h_answer compresses d²→d before the MLP.

New ideas:
  A) Depth: apply M multiple times (with LN between to prevent scale explosion)
       depth=1: h  = LN(M @ h_answer)
       depth=2: h  = LN(M @ LN(M @ h_answer))   — 2-hop pixel graph walk
       depth=3: h  = LN(M @ LN(M @ LN(M @ h_answer)))

     M @ h = Σ_t (h·emb_t) emb_t   is a weighted sum of pixel embeddings.
     Applying again: M @ (M@h) asks "which pixels align with the already-aligned set?"
     This captures second-order pixel co-occurrence structure.

  B) Multi-query: K independent answer vectors → K parallel bilinear probes
       H_answer ∈ R^{K×d}
       features = concat([M @ h_k for k in 1..K])  ∈ R^{K·d}
       logits   = MLP(LN(features))

     Each h_k looks for a different "direction" in the image.
     Richer than a single probe; far cheaper than reading all d² Gram entries.

  C) Multi-query + depth: each probe applied twice.

─── Param counts (factored3(4,4,4), d=64, emb_dim=64) ────────────────────────
  depth=1, mlp=M:   d*(3+M) + 11M + 10  =  64*(3+M) + 11M + 10
    mlp=128: 64*131 + 1418 = 9,802   (exp16 baseline: 97.82%)
    mlp=512: 64*515 + 5642 = 38,602  (exp16 baseline: 98.07%)

  depth=2, mlp=M:   adds one extra LN (2d params), rest same
    mlp=128: 9,802 + 128 = 9,930     (~same budget, can M² help?)
    mlp=512: 38,730

  depth=3, mlp=M:   adds two extra LNs (4d params)
    mlp=128: 9,802 + 256 = 10,058

  K-query, mlp=M: K*d (H_answer) + 2*K*d (LN on K*d-dim) + mlp*K*d + 11*M + 10
    K=2, mlp=128: 2*64*(3+128) + 1418 = 16,768 + 1418 = 18,186
    K=4, mlp=64:  4*64*(3+64)  +  714 = 17,152 + 714  = 17,866
    K=8, mlp=32:  8*64*(3+32)  +  362 = 17,920 + 362  = 18,282

─── Baselines (from exp16) ────────────────────────────────────────────────────
  depth=1 mlp=128:  9,802 non-emb → 97.82%
  depth=1 mlp=512: 38,602 non-emb → 98.07%
  f3(4,4,4) GELU mlp=128 (standard): 533,898 non-emb → 98.46%
"""

import json
import time
from pathlib import Path
import sys
import subprocess
from functools import partial

import jax
import jax.numpy as jnp
import optax
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_VAL_BINS_DEFAULT = 32
BATCH_SIZE = 128
LR         = 1e-3
LR_MUON    = 0.02
SEED       = 42

JSONL = Path(__file__).parent / "results_mnist.jsonl"
PLOT  = Path(__file__).parent / "plot_pareto_mnist.py"

T_IDX = jnp.arange(784)
ROWS  = T_IDX // 28
COLS  = T_IDX % 28


# ── utils ──────────────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

def n_nonemb(p, emb_keys):
    return sum(v.size for k, v in p.items() if k not in emb_keys)

def n_total(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")

def replot():
    subprocess.run([sys.executable, str(PLOT)], check=False)


# ── augmentation ───────────────────────────────────────────────────────────────

def random_noise_batch(imgs, rng, scale):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)

AUG_NOISE30 = "noise30"
_NOISE_SCALE = {AUG_NOISE30: 30.0}


# ── Muon ───────────────────────────────────────────────────────────────────────

def _newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = jnp.linalg.norm(G)
    X = G / (norm + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a*X + b*(A@X) + c*(A@(A@X))
    return X * norm

def ns_nesterov_direction(momentum=0.95, ns_steps=5):
    def init(params):
        return {"mu": jax.tree_util.tree_map(jnp.zeros_like, params)}
    def update(updates, state, params=None):
        mu       = state["mu"]
        new_mu   = jax.tree_util.tree_map(lambda m, g: momentum*m + g, mu, updates)
        nesterov = jax.tree_util.tree_map(lambda m, g: momentum*m + g, new_mu, updates)
        out = jax.tree_util.tree_map(
            lambda g: _newton_schulz(g, ns_steps) if g.ndim == 2 else g, nesterov)
        return out, {"mu": new_mu}
    return optax.GradientTransformation(init, update)

def make_muon_tx(params, emb_keys, adam_schedule, muon_schedule):
    mat_keys = frozenset(k for k, v in params.items()
                         if v.ndim == 2 and k not in emb_keys)
    labels = {k: "muon" if k in mat_keys else "adam" for k in params}
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform({
            "muon": optax.chain(
                ns_nesterov_direction(momentum=0.95),
                optax.scale_by_learning_rate(muon_schedule),
            ),
            "adam": optax.adam(adam_schedule),
        }, labels),
    )


# ── factored3 embedding ────────────────────────────────────────────────────────

def _f3_emb(params, x, d_r, d_c, d_v):
    B    = x.shape[0]
    K    = params["val_emb"].shape[0]
    bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
    re   = params["row_emb"][ROWS]
    ce   = params["col_emb"][COLS]
    ve   = params["val_emb"][bins]
    pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
    emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, d_r * d_c * d_v)
    return emb


# ── model builders ─────────────────────────────────────────────────────────────

def make_deep_bilinear(d_r, d_c, d_v, mlp_dim, depth=1, n_queries=1,
                       n_val_bins=N_VAL_BINS_DEFAULT):
    """
    Deep / multi-query bilinear readout of the Gram matrix M.

    depth:     number of times M is applied (with LN between applications)
    n_queries: number of independent answer vectors; their outputs are
               concatenated before the MLP.

    Architecture:
      For each query k:
        h = h_answer_k                          ∈ R^d
        for _ in range(depth):
            h = M @ h                           ∈ R^d
            h = LN(h, ln_gamma_hop_i, ...)      (per-hop LN, shared across queries)
      features = concat([h_k for k=1..K])       ∈ R^{K·d}
      features = LN(features, ln_gamma_out, ...) (final LN before MLP)
      logits   = GELU(W1 @ features + b1) → W2 → 10

    Non-emb params:
      n_queries * d      (H_answer)
      (depth) * 2 * d    (per-hop LN params, shared across queries)
      2 * n_queries * d  (output LN)
      mlp_dim * n_queries * d + mlp_dim  (W1, b1)
      10 * mlp_dim + 10                  (W2, b2)
    """
    emb_dim   = d_r * d_c * d_v
    feat_dim  = n_queries * emb_dim
    emb_keys  = frozenset({"row_emb", "col_emb", "val_emb"})

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))                 * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))                 * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (n_val_bins, d_v))         * 0.02,
        # Answer vectors: one per query (1D → Adam)
        "H_answer": jax.random.normal(next(kg).get(), (n_queries, emb_dim))     * 0.02,
        # Output LN (applied to the K*d-dim concatenated features)
        "ln_out_gamma": jnp.ones(feat_dim),
        "ln_out_beta":  jnp.zeros(feat_dim),
        # MLP
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }
    # Per-hop LN params (shared across all queries — same normalisation at each hop)
    for i in range(depth):
        p[f"ln_hop{i}_gamma"] = jnp.ones(emb_dim)
        p[f"ln_hop{i}_beta"]  = jnp.zeros(emb_dim)

    def fwd(params, x):
        emb = _f3_emb(params, x, d_r, d_c, d_v)            # (B, 784, d)
        M   = jnp.einsum("bti,btj->bij", emb, emb)          # (B, d, d) — keep as matrix!

        outs = []
        for q in range(n_queries):
            h = params["H_answer"][q]                        # (d,)
            h = jnp.broadcast_to(h, (M.shape[0], emb_dim))  # (B, d)
            for i in range(depth):
                h = jnp.einsum("bij,bj->bi", M, h)           # (B, d) = M @ h
                h = layer_norm(h, params[f"ln_hop{i}_gamma"],
                                  params[f"ln_hop{i}_beta"])
            outs.append(h)

        features = jnp.concatenate(outs, axis=-1)            # (B, K*d)
        features = layer_norm(features, params["ln_out_gamma"], params["ln_out_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, emb_keys


# ── training ───────────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          max_epochs=30, target_acc=0.995, aug=AUG_NOISE30):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    adam_sched  = optax.cosine_decay_schedule(LR,      total_steps)
    muon_sched  = optax.cosine_decay_schedule(LR_MUON, total_steps)
    tx = make_muon_tx(params, emb_keys, adam_sched, muon_sched)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    noise_scale = _NOISE_SCALE[aug]

    loss_fn = lambda p, bx, by: jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))

    def acc_fn(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i+BATCH_SIZE]), 1)
            for i in range(0, len(x), BATCH_SIZE)])
        return float(jnp.mean(preds == y))

    @partial(jax.jit, static_argnums=(0,))
    def step(opt_, p, s, bx, by):
        loss, g = jax.value_and_grad(loss_fn)(p, bx, by)
        u, s2   = opt_.update(g, s, p)
        return optax.apply_updates(p, u), s2, loss

    t0 = time.perf_counter()
    train_acc = test_acc = 0.0
    for epoch in range(max_epochs):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b+1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub, scale=noise_scale)
            params, opt_state, _ = step(tx, params, opt_state, bx, by)

        train_acc = acc_fn(params, X_tr, y_tr)
        test_acc  = acc_fn(params, X_te, y_te)
        logging.info(
            f"[{name}] epoch {epoch:2d}  "
            f"train={train_acc:.4f}  test={test_acc:.4f}  "
            f"{time.perf_counter()-t0:.1f}s"
        )
        if test_acc >= target_acc:
            break

    return train_acc, test_acc, time.perf_counter() - t0, epoch + 1


if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    F3 = (4, 4, 4)   # emb_dim = 64

    experiments = [
        # ── Group A: Depth (apply M multiple times) ───────────────────────────
        # Nearly same param count as depth=1 (only adds LN params per extra hop).
        # Does the 2nd/3rd application of M add useful structure?
        ("f3(4,4,4) bilinear depth=2 mlp=128 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=128, depth=2, n_queries=1),
         AUG_NOISE30, 30,
         "M²: 2-hop pixel walk; ~9.9K non-emb vs depth=1 97.82%"),

        ("f3(4,4,4) bilinear depth=3 mlp=128 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=128, depth=3, n_queries=1),
         AUG_NOISE30, 30,
         "M³: 3-hop pixel walk; ~10.1K non-emb"),

        ("f3(4,4,4) bilinear depth=2 mlp=512 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=512, depth=2, n_queries=1),
         AUG_NOISE30, 30,
         "M² depth=2 larger budget; ~38.7K vs depth=1+mlp512 98.07%"),

        # ── Group B: Multi-query (K independent answer vectors) ───────────────
        # Each h_k extracts a different direction from M.
        # Budget roughly matched to depth=1 mlp=128 (~10K) and mlp=512 (~38K).
        ("f3(4,4,4) bilinear K=2 mlp=128 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=128, depth=1, n_queries=2),
         AUG_NOISE30, 30,
         "K=2 queries; 2*64-dim features; ~18K non-emb"),

        ("f3(4,4,4) bilinear K=4 mlp=64 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=64, depth=1, n_queries=4),
         AUG_NOISE30, 30,
         "K=4 queries, small MLP; ~18K non-emb; more directions, less width"),

        ("f3(4,4,4) bilinear K=8 mlp=32 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=32, depth=1, n_queries=8),
         AUG_NOISE30, 30,
         "K=8 queries, tiny MLP; ~18K non-emb; many directions"),

        ("f3(4,4,4) bilinear K=4 mlp=128 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=128, depth=1, n_queries=4),
         AUG_NOISE30, 30,
         "K=4 queries, bigger MLP; ~34K non-emb"),

        # ── Group C: Multi-query + depth ─────────────────────────────────────
        # Can combining both yield the best of both worlds?
        ("f3(4,4,4) bilinear K=2 depth=2 mlp=128 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=128, depth=2, n_queries=2),
         AUG_NOISE30, 30,
         "K=2 queries × depth=2; ~18K non-emb"),

        ("f3(4,4,4) bilinear K=4 depth=2 mlp=64 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=64, depth=2, n_queries=4),
         AUG_NOISE30, 30,
         "K=4 queries × depth=2; ~34K non-emb"),
    ]

    prev_best = {
        9_802:   ("f3 bilinear depth=1 mlp=128", 0.9782),
        38_602:  ("f3 bilinear depth=1 mlp=512", 0.9807),
        533_898: ("f3 GELU mlp=128",             0.9846),
    }

    results = []
    for label, fwd, params, emb_keys, aug, max_ep, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        total  = n_total(params)
        emb    = total - nonemb
        closest = min(prev_best.keys(), key=lambda x: abs(x - nonemb))
        base_name, base_acc = prev_best[closest]
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  emb={emb:,}  total={total:,}  aug={aug}\n"
            f"  baseline: {base_name} = {base_acc:.4f}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug,
        )
        delta = te_acc - base_acc
        row = {
            "name": label, "experiment": "exp17",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, emb, tr_acc, te_acc, epochs, elapsed,
                        base_acc, base_name))
        logging.info(f"  → {te_acc:.4f}  vs {base_name} {base_acc:.4f}: {delta:+.4f}")

    print(f"\n{'='*124}")
    print(f"{'Variant':<55} {'NonEmb':>8} {'Train':>8} {'Test':>8} {'Delta':>7} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*124}")
    for label, nonemb, emb, tr, te, ep, t, base, bname in results:
        flag  = " ★" if te > base else "  "
        delta = f"{te-base:+.4f}"
        print(f"{label:<55} {nonemb:>8,} {tr:>7.4f} {te:>7.4f}{flag} {delta:>7} {ep:>4} {t:>7.1f}s")
    print(f"{'='*124}")
    print("\nBaselines:")
    for ne, (nm, acc) in prev_best.items():
        print(f"  {ne:>8,}  {acc:.4f}  {nm}")

    logging.info("Updating Pareto plot...")
    replot()
