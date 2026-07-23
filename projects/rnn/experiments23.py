"""
exp23 — no_gram_reshape: learned 2D reduction with h_helper.

Motivation: no_gram_fullH feeds all 784 dot-product scores to W1
(expensive). no_gram_bin4 pools into a 4×4 grid (cheap but uses
hardcoded spatial labels). This experiment uses a *learned* reduction:

    scores_k = H @ h_answer_k          ∈ R^{B×784}
    scores_2d = reshape(scores_k, (B, 28, 28))   # implicit 2D layout
    feat_k = scores_2d @ h_helper_k   ∈ R^{B×28}  (learned column projection)

h_helper_k is a learned 28-dim vector that acts as a soft column selector —
equivalent to taking a weighted average of the 28 column positions for each
of the 28 rows.  No hardcoded bin boundaries; the network decides how to
pool the spatial dimension.

Two variants:
  A. one_helper — single h_helper per query (28 features per query):
         feat_k = scores_2d @ h_helper_k   ∈ R^{B×28}
     non_emb = K*(d + 28 + 56) + mlp*(28*K + 11) + 10

  B. two_helpers — row + col helper per query (56 features per query):
         row_feat_k = scores_2d @ h_col_k               ∈ R^{B×28}
         col_feat_k = einsum("brc,r->bc", scores_2d, h_row_k)  ∈ R^{B×28}
         feat_k = concat([row_feat_k, col_feat_k])      ∈ R^{B×56}
     non_emb = K*(d + 56 + 112) + mlp*(56*K + 11) + 10
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

N_VAL_BINS = 32
BATCH_SIZE = 128
LR         = 1e-3
LR_MUON    = 0.02
SEED       = 42

JSONL        = Path(__file__).parent / "results_mnist.jsonl"
ABLATION_PLOT = Path(__file__).parent / "plot_ablation_mnist.py"

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28


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
    def init(p):  return {"mu": jax.tree_util.tree_map(jnp.zeros_like, p)}
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
            "muon": optax.chain(ns_nesterov_direction(),
                                optax.scale_by_learning_rate(muon_sched)),
            "adam": optax.adam(adam_sched),
        }, labels),
    )


# ── f3 embedding ───────────────────────────────────────────────────────────────

def _f3_emb(params, x, d_r, d_c, d_v):
    B    = x.shape[0]
    bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    re   = params["row_emb"][ROWS]
    ce   = params["col_emb"][COLS]
    ve   = params["val_emb"][bins]
    pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
    return (pos[None, :, :, None] * ve[:, :, None, :]).reshape(
               B, 784, d_r * d_c * d_v)

def _f3_params(d_r, d_c, d_v, kg):
    return {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))         * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))         * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
    }

F3_EMB_KEYS = frozenset({"row_emb", "col_emb", "val_emb"})


# ── A: no_gram_reshape_1h (one h_helper per query) ────────────────────────────

def make_no_gram_reshape_1h(d_r, d_c, d_v, mlp_dim, n_queries=1):
    """
    scores_k = H @ h_answer_k              ∈ R^{B×784}
    scores_2d = reshape → (B, 28, 28)
    feat_k = scores_2d @ h_helper_k        ∈ R^{B×28}   (pool col axis)
    features = concat(feat_k)              ∈ R^{B×28*K}
    logits = MLP(LN(features))

    non_emb = K*d + K*28 + 2*(28*K) + mlp*(28*K) + mlp + 10*mlp + 10
            = K*(d + 28 + 56) + mlp*(28*K + 11) + 10
    """
    d        = d_r * d_c * d_v
    feat_dim = n_queries * 28
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        **_f3_params(d_r, d_c, d_v, kg),
        "H_answer":  jax.random.normal(next(kg).get(), (n_queries, d))  * 0.02,
        "H_helper":  jax.random.normal(next(kg).get(), (n_queries, 28)) * 0.02,
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
        feats = []
        for k in range(n_queries):
            scores   = jnp.einsum("btd,d->bt", emb,
                                  params["H_answer"][k])       # (B, 784)
            scores2d = scores.reshape(B, 28, 28)               # (B, 28, 28)
            feat_k   = scores2d @ params["H_helper"][k]        # (B, 28)
            feats.append(feat_k)
        features = jnp.concatenate(feats, axis=-1)             # (B, 28*K)
        features = layer_norm(features, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, F3_EMB_KEYS


# ── B: no_gram_reshape_2h (row + col helpers per query → 56 features) ─────────

def make_no_gram_reshape_2h(d_r, d_c, d_v, mlp_dim, n_queries=1):
    """
    scores_2d = reshape(H @ h_answer_k, (B, 28, 28))
    row_feat_k = scores_2d @ h_col_k          ∈ R^{B×28}  (project col axis)
    col_feat_k = einsum("brc,r->bc", scores_2d, h_row_k)  ∈ R^{B×28}  (project row axis)
    feat_k = concat([row_feat_k, col_feat_k]) ∈ R^{B×56}
    features = concat(feat_k)                 ∈ R^{B×56*K}
    logits = MLP(LN(features))

    non_emb = K*(d + 56 + 112) + mlp*(56*K + 11) + 10
    """
    d        = d_r * d_c * d_v
    feat_dim = n_queries * 56
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        **_f3_params(d_r, d_c, d_v, kg),
        "H_answer":  jax.random.normal(next(kg).get(), (n_queries, d))  * 0.02,
        "H_col":     jax.random.normal(next(kg).get(), (n_queries, 28)) * 0.02,
        "H_row":     jax.random.normal(next(kg).get(), (n_queries, 28)) * 0.02,
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
        feats = []
        for k in range(n_queries):
            scores   = jnp.einsum("btd,d->bt", emb,
                                  params["H_answer"][k])       # (B, 784)
            s2d      = scores.reshape(B, 28, 28)               # (B, row, col)
            row_feat = s2d @ params["H_col"][k]                # (B, 28) — project cols
            col_feat = jnp.einsum("brc,r->bc", s2d,
                                  params["H_row"][k])          # (B, 28) — project rows
            feats.append(jnp.concatenate([row_feat, col_feat], axis=-1))  # (B, 56)
        features = jnp.concatenate(feats, axis=-1)             # (B, 56*K)
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
        # ── 1-helper: 28 features per query ───────────────────────────────────
        # non_emb = K*(d+84) + mlp*(28K+11) + 10   [d=64]
        ("no_gram_reshape_1h f3(4,4,4) K=1 mlp=8 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 8,  n_queries=1)),
        ("no_gram_reshape_1h f3(4,4,4) K=1 mlp=16 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 16, n_queries=1)),
        ("no_gram_reshape_1h f3(4,4,4) K=1 mlp=32 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 32, n_queries=1)),
        ("no_gram_reshape_1h f3(4,4,4) K=1 mlp=64 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 64, n_queries=1)),
        ("no_gram_reshape_1h f3(4,4,4) K=4 mlp=8 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 8,  n_queries=4)),
        ("no_gram_reshape_1h f3(4,4,4) K=4 mlp=16 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 16, n_queries=4)),
        ("no_gram_reshape_1h f3(4,4,4) K=4 mlp=32 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 32, n_queries=4)),

        # ── 2-helpers: 56 features per query (row profile + col profile) ──────
        # non_emb = K*(d+168) + mlp*(56K+11) + 10
        ("no_gram_reshape_2h f3(4,4,4) K=1 mlp=8 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 8,  n_queries=1)),
        ("no_gram_reshape_2h f3(4,4,4) K=1 mlp=16 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 16, n_queries=1)),
        ("no_gram_reshape_2h f3(4,4,4) K=1 mlp=32 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 32, n_queries=1)),
        ("no_gram_reshape_2h f3(4,4,4) K=4 mlp=8 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 8,  n_queries=4)),
        ("no_gram_reshape_2h f3(4,4,4) K=4 mlp=16 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 16, n_queries=4)),
        ("no_gram_reshape_2h f3(4,4,4) K=4 mlp=32 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 32, n_queries=4)),
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
            "name": label, "experiment": "exp23",
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
    print(f"{'Variant':<55} {'NonEmb':>8} {'Total':>8} {'Train':>8} {'Test':>8} {'Time':>7}")
    print(f"{'-'*115}")
    for label, nonemb, ntotal, tr, te, t in results:
        print(f"{label:<55} {nonemb:>8,} {ntotal:>8,} {tr:>8.4f} {te:>8.4f} {t:>6.1f}s")
    print(f"{'='*115}")

    subprocess.run([sys.executable, str(ABLATION_PLOT)], check=False)
