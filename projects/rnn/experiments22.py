"""
exp22 — Improved ablations: deeper pure MLP, mean-pool readout, spatial binning.

Three families, all with noise30 augmentation:

  A. pure_mlp (raw pixels, no embeddings):
     - Tiny 1-layer sizes: mlp=2/4/8   (below exp21's mlp=16 floor)
     - 2-layer MLP at same total budget as 1-layer (deeper, not wider)

  B. no_gram_mean  — same f3(4,4,4) embedding, but pool emb across positions
     first (first-order statistics), then apply K dot-product queries:
         mean_emb = mean_t(emb_t)   ∈ R^d         (1st-order; no M needed)
         features = mean_emb @ H_answer^T  ∈ R^K
         logits   = MLP(LN(features))
     Isolates: does using emb_t as *value* (the M@h trick) matter?
     non_emb = K*d + 2*K + mlp*K + mlp*11 + 10

  C. no_gram_bin4  — 4×4 spatial binning (16 bins) before readout:
         scores_k  = einsum("btd,d->bt", H, h_answer_k)   ∈ R^{B×784}
         bin_scores = pool scores into 4×4 grid → R^{B×16}
         features  = concat(bin_scores_k)  ∈ R^{B×16*K}
         logits    = MLP(LN(features))
     Intermediate between full spatial (no_gram_fullH, 784*K features)
     and mean pooling (K features). 16×K features.
     non_emb = K*d + 2*(16*K) + mlp*(16*K) + mlp*11 + 10
"""

import json, time, sys, subprocess
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import optax
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_VAL_BINS  = 32
BATCH_SIZE  = 128
LR          = 1e-3
LR_MUON     = 0.02
SEED        = 42

JSONL = Path(__file__).parent / "results_mnist.jsonl"
ABLATION_PLOT = Path(__file__).parent / "plot_ablation_mnist.py"

ROWS = jnp.arange(784) // 28   # pixel → row index (0-27)
COLS = jnp.arange(784) % 28    # pixel → col index (0-27)

# 4×4 spatial bins: which bin does each pixel belong to? (0-15)
BIN4_IDX = (ROWS // 7) * 4 + (COLS // 7)   # 7 = 28//4


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

def random_noise_batch(imgs, rng, scale=30.0):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)


# ── Muon ───────────────────────────────────────────────────────────────────────

def _newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = jnp.linalg.norm(G)
    X = G / (norm + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a*X + b*(A@X) + c*(A@(A@X))
    return X * norm

def ns_nesterov_direction(momentum=0.95):
    def init(p):   return {"mu": jax.tree_util.tree_map(jnp.zeros_like, p)}
    def update(g, s, p=None):
        mu  = s["mu"]
        mu2 = jax.tree_util.tree_map(lambda m, gi: momentum*m + gi, mu, g)
        nest = jax.tree_util.tree_map(lambda m, gi: momentum*m + gi, mu2, g)
        out  = jax.tree_util.tree_map(
            lambda gi: _newton_schulz(gi) if gi.ndim == 2 else gi, nest)
        return out, {"mu": mu2}
    return optax.GradientTransformation(init, update)

def make_muon_tx(params, emb_keys, adam_sched, muon_sched):
    mat_keys = frozenset(k for k, v in params.items()
                         if v.ndim == 2 and k not in emb_keys)
    labels = {k: "muon" if k in mat_keys else "adam" for k in params}
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform({
            "muon": optax.chain(ns_nesterov_direction(), optax.scale_by_learning_rate(muon_sched)),
            "adam": optax.adam(adam_sched),
        }, labels),
    )


# ── f3 embedding helper ────────────────────────────────────────────────────────

def _f3_emb(params, x, d_r, d_c, d_v):
    B    = x.shape[0]
    bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    re   = params["row_emb"][ROWS]
    ce   = params["col_emb"][COLS]
    ve   = params["val_emb"][bins]
    pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
    return (pos[None, :, :, None] * ve[:, :, None, :]).reshape(
               x.shape[0], 784, d_r * d_c * d_v)              # (B, 784, d)

def _f3_params(d_r, d_c, d_v, kg):
    return {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))        * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))        * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v))* 0.02,
    }

F3_EMB_KEYS = frozenset({"row_emb", "col_emb", "val_emb"})


# ── A: pure MLP (1-layer) ──────────────────────────────────────────────────────

def make_pure_mlp(mlp_dim):
    """1-hidden-layer GELU MLP on normalised raw pixels [0,1].
    All params are non-emb. total = mlp*784 + mlp*11 + 10.
    """
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, 784)) / 784**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    def fwd(params, x):
        h = jax.nn.gelu(x / 255.0 @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]
    return fwd, p, frozenset()


# ── A: pure MLP (2-layer) ──────────────────────────────────────────────────────

def make_pure_mlp_2layer(h1, h2):
    """2-hidden-layer GELU MLP on normalised raw pixels.
    total = 784*h1 + h1 + h1*h2 + h2 + h2*10 + 10.
    """
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "W1": jax.random.normal(next(kg).get(), (h1, 784)) / 784**0.5,
        "b1": jnp.zeros(h1),
        "W2": jax.random.normal(next(kg).get(), (h2, h1)) / h1**0.5,
        "b2": jnp.zeros(h2),
        "W3": jax.random.normal(next(kg).get(), (10, h2)) * 0.01,
        "b3": jnp.zeros(10),
    }
    def fwd(params, x):
        h = jax.nn.gelu(x / 255.0 @ params["W1"].T + params["b1"])
        h = jax.nn.gelu(h @ params["W2"].T + params["b2"])
        return h @ params["W3"].T + params["b3"]
    return fwd, p, frozenset()


# ── B: no_gram_mean (mean-pool embeddings → K dot products) ───────────────────

def make_no_gram_mean(d_r, d_c, d_v, mlp_dim, n_queries):
    """f3 embedding + mean-pool over positions (first-order statistics only).

    mean_emb = (1/784) Σ_t emb_t   ∈ R^d
    features = mean_emb @ H_answer^T  ∈ R^K   (K dot products)
    logits   = MLP(LN(features))

    non_emb = K*d + 2*K + mlp*K + mlp*11 + 10
    """
    d        = d_r * d_c * d_v
    feat_dim = n_queries
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        **_f3_params(d_r, d_c, d_v, kg),
        "H_answer":  jax.random.normal(next(kg).get(), (n_queries, d))       * 0.02,
        "ln_gamma":  jnp.ones(feat_dim),
        "ln_beta":   jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }

    def fwd(params, x):
        emb      = _f3_emb(params, x, d_r, d_c, d_v)           # (B, 784, d)
        mean_emb = jnp.mean(emb, axis=1)                        # (B, d)
        features = mean_emb @ params["H_answer"].T              # (B, K)
        features = layer_norm(features, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, F3_EMB_KEYS


# ── C: no_gram_bin4 (4×4 spatial binning → 16 scores per query) ───────────────

def make_no_gram_bin4(d_r, d_c, d_v, mlp_dim, n_queries):
    """f3 embedding + 4×4 spatial binning (16 bins) before MLP.

    For each query k: scores_t = emb_t · h_answer_k ∈ R^{784}
    Pool each score into its 4×4 bin (7×7 pixels/bin) by averaging.
    features = concat(bin_scores_k)  ∈ R^{16*K}
    logits   = MLP(LN(features))

    non_emb = K*d + 2*(16*K) + mlp*(16*K) + mlp*11 + 10
    """
    d        = d_r * d_c * d_v
    n_bins   = 16                   # 4×4 spatial grid
    feat_dim = n_queries * n_bins
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        **_f3_params(d_r, d_c, d_v, kg),
        "H_answer":  jax.random.normal(next(kg).get(), (n_queries, d))       * 0.02,
        "ln_gamma":  jnp.ones(feat_dim),
        "ln_beta":   jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }

    def fwd(params, x):
        B   = x.shape[0]
        emb = _f3_emb(params, x, d_r, d_c, d_v)               # (B, 784, d)
        # dot product with each answer vector: (B, 784, K)
        scores = jnp.einsum("btd,kd->btk", emb, params["H_answer"])
        # spatial binning: average within each of the 16 bins
        # shape (B, 784, K) → accumulate into (B, 16, K) via segment mean
        bin_sums = jax.ops.segment_sum(
            scores.transpose(1, 0, 2),       # (784, B, K)
            BIN4_IDX, num_segments=n_bins    # (16, B, K)
        ).transpose(1, 0, 2)                 # (B, 16, K)
        bin_scores = bin_sums / (784 / n_bins)  # average (each bin has 784/16=49 pixels)
        features = bin_scores.reshape(B, feat_dim)              # (B, 16*K)
        features = layer_norm(features, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, F3_EMB_KEYS


# ── training loop ──────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te, max_epochs=30):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    adam_sched  = optax.cosine_decay_schedule(LR,      total_steps)
    muon_sched  = optax.cosine_decay_schedule(LR_MUON, total_steps)
    tx        = make_muon_tx(params, emb_keys, adam_sched, muon_sched)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    loss_fn   = lambda p, bx, by: jnp.mean(
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
            bx = random_noise_batch(bx, sub)
            params, opt_state, _ = step(tx, params, opt_state, bx, by)
        train_acc = acc_fn(params, X_tr, y_tr)
        test_acc  = acc_fn(params, X_te, y_te)
        logging.info(f"[{name}] epoch {epoch:2d}  train={train_acc:.4f}  "
                     f"test={test_acc:.4f}  {time.perf_counter()-t0:.1f}s")

    return train_acc, test_acc, time.perf_counter() - t0


if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    experiments = [
        # ── A: pure_mlp 1-layer tiny ──────────────────────────────────────────
        ("pure_mlp mlp=2 +noise30",  *make_pure_mlp(2)),
        ("pure_mlp mlp=4 +noise30",  *make_pure_mlp(4)),
        ("pure_mlp mlp=8 +noise30",  *make_pure_mlp(8)),

        # ── A: pure_mlp 2-layer (deeper at same budget as 1-layer) ───────────
        # 1-layer mlp=8  ≈ 6,370 total → 2-layer h1=8, h2=8 ≈ 6,442
        ("pure_mlp 2layer h=8 +noise30",   *make_pure_mlp_2layer(8,  8)),
        # 1-layer mlp=16 ≈ 12,730 total → 2-layer h1=16, h2=16 ≈ 13,002
        ("pure_mlp 2layer h=16 +noise30",  *make_pure_mlp_2layer(16, 16)),
        # 1-layer mlp=32 ≈ 25,450 total → 2-layer h1=32, h2=32 ≈ 25,962
        ("pure_mlp 2layer h=32 +noise30",  *make_pure_mlp_2layer(32, 32)),
        # 1-layer mlp=64 ≈ 50,890 total → 2-layer h1=64, h2=64 ≈ 52,490
        ("pure_mlp 2layer h=64 +noise30",  *make_pure_mlp_2layer(64, 64)),
        # 2-layer with narrow bottleneck (cheaper W1, wider W2)
        ("pure_mlp 2layer h1=8 h2=32 +noise30",  *make_pure_mlp_2layer(8,  32)),
        ("pure_mlp 2layer h1=16 h2=64 +noise30", *make_pure_mlp_2layer(16, 64)),

        # ── B: no_gram_mean (first-order: mean emb → K dot products) ─────────
        # K=16: 16 features → small MLP
        ("no_gram_mean f3(4,4,4) K=16 mlp=16 +noise30", *make_no_gram_mean(4,4,4, 16, 16)),
        ("no_gram_mean f3(4,4,4) K=16 mlp=32 +noise30", *make_no_gram_mean(4,4,4, 32, 16)),
        # K=32: 32 features — richer first-order
        ("no_gram_mean f3(4,4,4) K=32 mlp=8 +noise30",  *make_no_gram_mean(4,4,4,  8, 32)),
        ("no_gram_mean f3(4,4,4) K=32 mlp=16 +noise30", *make_no_gram_mean(4,4,4, 16, 32)),
        ("no_gram_mean f3(4,4,4) K=32 mlp=32 +noise30", *make_no_gram_mean(4,4,4, 32, 32)),
        # K=64: same output dim as bilinear K=1 (d=64 features)
        ("no_gram_mean f3(4,4,4) K=64 mlp=8 +noise30",  *make_no_gram_mean(4,4,4,  8, 64)),
        ("no_gram_mean f3(4,4,4) K=64 mlp=16 +noise30", *make_no_gram_mean(4,4,4, 16, 64)),
        ("no_gram_mean f3(4,4,4) K=64 mlp=32 +noise30", *make_no_gram_mean(4,4,4, 32, 64)),
        ("no_gram_mean f3(4,4,4) K=64 mlp=64 +noise30", *make_no_gram_mean(4,4,4, 64, 64)),

        # ── C: no_gram_bin4 (4×4 spatial grid → 16 bins per query) ───────────
        # K=1: 16 spatial features — very cheap
        ("no_gram_bin4 f3(4,4,4) K=1 mlp=8 +noise30",  *make_no_gram_bin4(4,4,4,  8, 1)),
        ("no_gram_bin4 f3(4,4,4) K=1 mlp=16 +noise30", *make_no_gram_bin4(4,4,4, 16, 1)),
        ("no_gram_bin4 f3(4,4,4) K=1 mlp=32 +noise30", *make_no_gram_bin4(4,4,4, 32, 1)),
        ("no_gram_bin4 f3(4,4,4) K=1 mlp=64 +noise30", *make_no_gram_bin4(4,4,4, 64, 1)),
        # K=4: 64 spatial features — same dim as bilinear K=1 output (d=64)
        ("no_gram_bin4 f3(4,4,4) K=4 mlp=8 +noise30",  *make_no_gram_bin4(4,4,4,  8, 4)),
        ("no_gram_bin4 f3(4,4,4) K=4 mlp=16 +noise30", *make_no_gram_bin4(4,4,4, 16, 4)),
        ("no_gram_bin4 f3(4,4,4) K=4 mlp=32 +noise30", *make_no_gram_bin4(4,4,4, 32, 4)),
    ]

    results = []
    for label, fwd, params, emb_keys in experiments:
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(f"\n{'='*60}\n{label}\n  non-emb={nonemb:,}  total={ntotal:,}\n{'='*60}")
        tr_acc, te_acc, elapsed = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test)
        row = {
            "name": label, "experiment": "exp22",
            "n_nonemb_params": nonemb,
            "n_total_params":  ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": 30, "time_s": round(elapsed, 1), "notes": "",
        }
        append_result(row)
        results.append((label, nonemb, ntotal, tr_acc, te_acc, elapsed))
        logging.info(f"  → train={tr_acc:.4f}  test={te_acc:.4f}")

    results.sort(key=lambda r: r[1])
    print(f"\n{'='*115}")
    print(f"{'Variant':<52} {'NonEmb':>8} {'Total':>8} {'Train':>8} {'Test':>8} {'Time':>8}")
    print(f"{'-'*115}")
    for label, nonemb, ntotal, tr, te, t in results:
        print(f"{label:<52} {nonemb:>8,} {ntotal:>8,} {tr:>8.4f} {te:>8.4f} {t:>7.1f}s")
    print(f"{'='*115}")

    subprocess.run([sys.executable, str(ABLATION_PLOT)], check=False)
