"""
Fashion-MNIST study: architecture families from the MNIST Pareto front.

Tests the top families on Fashion-MNIST with light tuning per family.
40 epochs (vs 30 on MNIST); compares noise=15 vs noise=30 for key models.

Families:
  1. gram_f3    — f3(4,4,4) bilinear (Gram M accumulator + factored3 emb)
  2. gram_joint — joint(d=16) bilinear (Gram M, full lookup emb; small non-emb end)
  3. reshape    — no_gram_reshape_1h (best no-Gram ablation)
  4. pure_mlp   — raw-pixel 1-layer MLP baseline
  5. pure_mlp2  — raw-pixel 2-layer MLP baseline

Usage:
    uv run python projects/rnn/experiments_fashion.py
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

JSONL = Path(__file__).parent / "results_fashion_mnist.jsonl"
PLOT  = Path(__file__).parent / "plot_pareto_fashion.py"

ROWS  = jnp.arange(784) // 28
COLS  = jnp.arange(784) % 28
T_IDX = jnp.arange(784)


# ── utilities ────────────────────────────────────────────────────────────────

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

def random_noise_batch(imgs, rng, scale):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)


# ── Muon optimizer ────────────────────────────────────────────────────────────

def _newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = jnp.linalg.norm(G)
    X = G / (norm + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a*X + b*(A@X) + c*(A@(A@X))
    return X * norm

def ns_nesterov_direction(momentum=0.95):
    def init(p): return {"mu": jax.tree_util.tree_map(jnp.zeros_like, p)}
    def update(g, s, p=None):
        mu   = s["mu"]
        mu2  = jax.tree_util.tree_map(lambda m, gi: momentum*m + gi, mu, g)
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


# ── f3 embedding helpers ──────────────────────────────────────────────────────

def _f3_emb(params, x, d_r, d_c, d_v):
    bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    re   = params["row_emb"][ROWS]
    ce   = params["col_emb"][COLS]
    ve   = params["val_emb"][bins]
    pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
    return (pos[None, :, :, None] * ve[:, :, None, :]).reshape(
               x.shape[0], 784, d_r * d_c * d_v)

def _f3_params(d_r, d_c, d_v, kg):
    return {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))         * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))         * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
    }

F3_EMB_KEYS    = frozenset({"row_emb", "col_emb", "val_emb"})
JOINT_EMB_KEYS = frozenset({"joint_emb"})


# ── Family 1: f3(4,4,4) bilinear (Gram model) ────────────────────────────────

def make_f3_bilinear(d_r, d_c, d_v, mlp_dim, n_queries=1, depth=1):
    """Gram state M = Σ emb⊗emb; readout = M @ h_answer (bilinear).
    non_emb = K*d*(3+mlp) + depth*2*d + 11*mlp + 10
    emb     = 28*d_r + 28*d_c + 32*d_v
    """
    d        = d_r * d_c * d_v
    feat_dim = n_queries * d
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        **_f3_params(d_r, d_c, d_v, kg),
        "H_answer":     jax.random.normal(next(kg).get(), (n_queries, d)) * 0.02,
        "ln_out_gamma": jnp.ones(feat_dim),
        "ln_out_beta":  jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }
    for i in range(depth):
        p[f"ln_hop{i}_gamma"] = jnp.ones(d)
        p[f"ln_hop{i}_beta"]  = jnp.zeros(d)

    def fwd(params, x):
        B   = x.shape[0]
        emb = _f3_emb(params, x, d_r, d_c, d_v)
        M   = jnp.einsum("bti,btj->bij", emb, emb)
        outs = []
        for q in range(n_queries):
            h = jnp.broadcast_to(params["H_answer"][q], (B, d))
            for i in range(depth):
                h = jnp.einsum("bij,bj->bi", M, h)
                h = layer_norm(h, params[f"ln_hop{i}_gamma"],
                               params[f"ln_hop{i}_beta"])
            outs.append(h)
        features = jnp.concatenate(outs, axis=-1)
        features = layer_norm(features, params["ln_out_gamma"], params["ln_out_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, F3_EMB_KEYS


# ── Family 2: joint(d=16) bilinear (large lookup emb) ────────────────────────

def make_joint_bilinear(d, mlp_dim, n_queries=1, depth=1):
    """Full joint embedding E[position, bin] + Gram bilinear readout.
    emb     = 784 * 32 * d  (large table)
    non_emb = K*d*(3+mlp) + depth*2*d + 11*mlp + 10
    """
    feat_dim = n_queries * d
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "joint_emb":    jax.random.normal(next(kg).get(), (784, N_VAL_BINS, d)) * 0.02,
        "H_answer":     jax.random.normal(next(kg).get(), (n_queries, d))       * 0.02,
        "ln_out_gamma": jnp.ones(feat_dim),
        "ln_out_beta":  jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }
    for i in range(depth):
        p[f"ln_hop{i}_gamma"] = jnp.ones(d)
        p[f"ln_hop{i}_beta"]  = jnp.zeros(d)

    def fwd(params, x):
        B    = x.shape[0]
        bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
        emb  = params["joint_emb"][T_IDX, bins]
        M    = jnp.einsum("bti,btj->bij", emb, emb)
        outs = []
        for q in range(n_queries):
            h = jnp.broadcast_to(params["H_answer"][q], (B, d))
            for i in range(depth):
                h = jnp.einsum("bij,bj->bi", M, h)
                h = layer_norm(h, params[f"ln_hop{i}_gamma"],
                               params[f"ln_hop{i}_beta"])
            outs.append(h)
        features = jnp.concatenate(outs, axis=-1)
        features = layer_norm(features, params["ln_out_gamma"], params["ln_out_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, JOINT_EMB_KEYS


# ── Family 3: no_gram_reshape_1h ─────────────────────────────────────────────

def make_no_gram_reshape_1h(d_r, d_c, d_v, mlp_dim, n_queries=1):
    """No Gram state: reshape H@h_answer to (28,28), apply h_helper → 28 features.
    non_emb = K*(d+84) + mlp*(28K+11) + 10
    """
    d        = d_r * d_c * d_v
    feat_dim = n_queries * 28
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        **_f3_params(d_r, d_c, d_v, kg),
        "H_answer": jax.random.normal(next(kg).get(), (n_queries, d))  * 0.02,
        "H_helper": jax.random.normal(next(kg).get(), (n_queries, 28)) * 0.02,
        "ln_gamma": jnp.ones(feat_dim),
        "ln_beta":  jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }

    def fwd(params, x):
        B   = x.shape[0]
        emb = _f3_emb(params, x, d_r, d_c, d_v)
        feats = []
        for k in range(n_queries):
            scores   = jnp.einsum("btd,d->bt", emb, params["H_answer"][k])
            scores2d = scores.reshape(B, 28, 28)
            feats.append(scores2d @ params["H_helper"][k])
        features = jnp.concatenate(feats, axis=-1)
        features = layer_norm(features, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, F3_EMB_KEYS


# ── Family 4/5: pure MLP (raw pixels, no embedding) ──────────────────────────

def make_pure_mlp(mlp_dim):
    """1-hidden-layer GELU MLP on normalised raw pixels.
    total = mlp*784 + mlp + mlp*10 + 10
    """
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, 784)) / 784**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))  * 0.01,
        "b2": jnp.zeros(10),
    }
    def fwd(params, x):
        h = jax.nn.gelu(x / 255.0 @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]
    return fwd, p, frozenset()


def make_pure_mlp_2layer(h1, h2):
    """2-hidden-layer GELU MLP on normalised raw pixels.
    total = 784*h1 + h1 + h1*h2 + h2 + h2*10 + 10
    """
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "W1": jax.random.normal(next(kg).get(), (h1,  784)) / 784**0.5,
        "b1": jnp.zeros(h1),
        "W2": jax.random.normal(next(kg).get(), (h2,  h1))  / h1**0.5,
        "b2": jnp.zeros(h2),
        "W3": jax.random.normal(next(kg).get(), (10,  h2))  * 0.01,
        "b3": jnp.zeros(10),
    }
    def fwd(params, x):
        h = jax.nn.gelu(x / 255.0 @ params["W1"].T + params["b1"])
        h = jax.nn.gelu(h         @ params["W2"].T + params["b2"])
        return h @ params["W3"].T + params["b3"]
    return fwd, p, frozenset()


# ── training loop ─────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          max_epochs=40, noise_scale=30.0):
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
            bx = random_noise_batch(bx, sub, noise_scale)
            params, opt_state, _ = step(tx, params, opt_state, bx, by)
        train_acc = acc_fn(params, X_tr, y_tr)
        test_acc  = acc_fn(params, X_te, y_te)
        logging.info(f"[{name}] epoch {epoch:2d}  train={train_acc:.4f}  "
                     f"test={test_acc:.4f}  {time.perf_counter()-t0:.1f}s")
    return train_acc, test_acc, time.perf_counter() - t0


if __name__ == "__main__":
    logging.info("Loading Fashion-MNIST...")
    data    = load_supervised_image("fashion_mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # (label, fwd, params, emb_keys, noise_scale, family)
    experiments = [
        # ── gram_f3: f3(4,4,4) bilinear, noise=30 ─────────────────────────────
        ("gram_f3 K=1 mlp=32 +noise30",
         *make_f3_bilinear(4,4,4,  32, n_queries=1), 30.0, "gram_f3"),
        ("gram_f3 K=2 mlp=32 +noise30",
         *make_f3_bilinear(4,4,4,  32, n_queries=2), 30.0, "gram_f3"),
        ("gram_f3 K=1 mlp=64 +noise30",
         *make_f3_bilinear(4,4,4,  64, n_queries=1), 30.0, "gram_f3"),
        ("gram_f3 K=1 mlp=96 +noise30",
         *make_f3_bilinear(4,4,4,  96, n_queries=1), 30.0, "gram_f3"),
        ("gram_f3 K=2 mlp=128 +noise30",
         *make_f3_bilinear(4,4,4, 128, n_queries=2), 30.0, "gram_f3"),
        ("gram_f3 K=4 mlp=64 +noise30",
         *make_f3_bilinear(4,4,4,  64, n_queries=4), 30.0, "gram_f3"),
        ("gram_f3 K=4 mlp=128 +noise30",
         *make_f3_bilinear(4,4,4, 128, n_queries=4), 30.0, "gram_f3"),
        ("gram_f3 K=4 depth=2 mlp=128 +noise30",
         *make_f3_bilinear(4,4,4, 128, n_queries=4, depth=2), 30.0, "gram_f3"),
        ("gram_f3 K=4 mlp=256 +noise30",
         *make_f3_bilinear(4,4,4, 256, n_queries=4), 30.0, "gram_f3"),
        ("gram_f3 K=4 depth=2 mlp=256 +noise30",
         *make_f3_bilinear(4,4,4, 256, n_queries=4, depth=2), 30.0, "gram_f3"),

        # ── gram_f3: noise=15 tuning (Fashion textures may prefer less noise) ─
        ("gram_f3 K=4 mlp=128 +noise15",
         *make_f3_bilinear(4,4,4, 128, n_queries=4), 15.0, "gram_f3"),
        ("gram_f3 K=4 depth=2 mlp=128 +noise15",
         *make_f3_bilinear(4,4,4, 128, n_queries=4, depth=2), 15.0, "gram_f3"),
        ("gram_f3 K=4 mlp=256 +noise15",
         *make_f3_bilinear(4,4,4, 256, n_queries=4), 15.0, "gram_f3"),
        ("gram_f3 K=4 depth=2 mlp=256 +noise15",
         *make_f3_bilinear(4,4,4, 256, n_queries=4, depth=2), 15.0, "gram_f3"),

        # ── gram_joint: joint(d=16), noise=30 (small non-emb end) ─────────────
        ("gram_joint d=16 K=1 mlp=16 +noise30",
         *make_joint_bilinear(16, 16, n_queries=1), 30.0, "gram_joint"),
        ("gram_joint d=16 K=2 mlp=32 +noise30",
         *make_joint_bilinear(16, 32, n_queries=2), 30.0, "gram_joint"),
        ("gram_joint d=16 K=2 mlp=64 +noise30",
         *make_joint_bilinear(16, 64, n_queries=2), 30.0, "gram_joint"),

        # ── reshape: no_gram_reshape_1h, noise=30 and noise=15 ────────────────
        ("reshape_1h K=4 mlp=64 +noise30",
         *make_no_gram_reshape_1h(4,4,4,  64, n_queries=4), 30.0, "reshape"),
        ("reshape_1h K=4 mlp=128 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 128, n_queries=4), 30.0, "reshape"),
        ("reshape_1h K=4 mlp=256 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 256, n_queries=4), 30.0, "reshape"),
        ("reshape_1h K=4 mlp=256 +noise15",
         *make_no_gram_reshape_1h(4,4,4, 256, n_queries=4), 15.0, "reshape"),

        # ── pure_mlp 1-layer, noise=30 ────────────────────────────────────────
        ("pure_mlp 1L mlp=64 +noise30",  *make_pure_mlp( 64), 30.0, "pure_mlp"),
        ("pure_mlp 1L mlp=128 +noise30", *make_pure_mlp(128), 30.0, "pure_mlp"),
        ("pure_mlp 1L mlp=256 +noise30", *make_pure_mlp(256), 30.0, "pure_mlp"),
        ("pure_mlp 1L mlp=512 +noise30", *make_pure_mlp(512), 30.0, "pure_mlp"),

        # ── pure_mlp 2-layer, noise=30 ────────────────────────────────────────
        ("pure_mlp 2L h=128x64 +noise30",
         *make_pure_mlp_2layer(128,  64), 30.0, "pure_mlp2"),
        ("pure_mlp 2L h=256x128 +noise30",
         *make_pure_mlp_2layer(256, 128), 30.0, "pure_mlp2"),
        ("pure_mlp 2L h=512x256 +noise30",
         *make_pure_mlp_2layer(512, 256), 30.0, "pure_mlp2"),
    ]

    results = []
    for label, fwd, params, emb_keys, noise, family in experiments:
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(f"\n{'='*60}\n{label}\n"
                     f"  non-emb={nonemb:,}  total={ntotal:,}\n{'='*60}")
        tr_acc, te_acc, elapsed = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            noise_scale=noise)
        row = {
            "name": label, "experiment": "fashion1",
            "family": family,
            "n_nonemb_params": nonemb,
            "n_total_params":  ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": 40, "time_s": round(elapsed, 1),
        }
        append_result(row)
        results.append((label, nonemb, ntotal, tr_acc, te_acc, family))
        logging.info(f"  → train={tr_acc:.4f}  test={te_acc:.4f}")

    results.sort(key=lambda r: r[1])
    print(f"\n{'='*110}")
    print(f"{'Variant':<52} {'Family':<12} {'NonEmb':>8} {'Total':>10} {'Train':>8} {'Test':>8}")
    print(f"{'-'*110}")
    for label, nonemb, ntotal, tr, te, fam in results:
        print(f"{label:<52} {fam:<12} {nonemb:>8,} {ntotal:>10,} {tr:>8.4f} {te:>8.4f}")
    print(f"{'='*110}")

    subprocess.run([sys.executable, str(PLOT)], check=False)
