"""
exp24 — no_gram_reshape higher-end sweep.

Push reshape_1h and reshape_2h into the 8K–35K total-param range to see
how they compare to the Gram bilinear Pareto front at the same budgets:

  Gram bilinear reference points:
    7,882 total  → 97.94%   (f3 K=1 mlp=96)
   18,666 total  → 98.28%   (f3 K=2 mlp=128)
   35,434 total  → 98.54%   (f3 K=4 mlp=128)
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

JSONL         = Path(__file__).parent / "results_mnist.jsonl"
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


def _f3_emb(params, x, d_r, d_c, d_v):
    B    = x.shape[0]
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

F3_EMB_KEYS = frozenset({"row_emb", "col_emb", "val_emb"})


def make_no_gram_reshape_1h(d_r, d_c, d_v, mlp_dim, n_queries=1):
    """scores_2d @ h_helper → 28 features per query.
    non_emb = K*(d+84) + mlp*(28K+11) + 10
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


def make_no_gram_reshape_2h(d_r, d_c, d_v, mlp_dim, n_queries=1):
    """row_feat + col_feat → 56 features per query.
    non_emb = K*(d+168) + mlp*(56K+11) + 10
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
        emb = _f3_emb(params, x, d_r, d_c, d_v)
        feats = []
        for k in range(n_queries):
            scores = jnp.einsum("btd,d->bt", emb, params["H_answer"][k])
            s2d    = scores.reshape(B, 28, 28)
            feats.append(jnp.concatenate([
                s2d @ params["H_col"][k],
                jnp.einsum("brc,r->bc", s2d, params["H_row"][k]),
            ], axis=-1))
        features = jnp.concatenate(feats, axis=-1)
        features = layer_norm(features, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]
    return fwd, p, F3_EMB_KEYS


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

    # non_emb = K*(64+84) + mlp*(28K+11) + 10   for 1h
    # non_emb = K*(64+168) + mlp*(56K+11) + 10  for 2h
    experiments = [
        # ── 1h, K=4 higher MLP ────────────────────────────────────────────────
        # K=4 mlp=64:  592 + 64*123 + 10 = 8,474 non-emb  (8,826 total)
        ("no_gram_reshape_1h f3(4,4,4) K=4 mlp=64 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 64,  n_queries=4)),
        # K=4 mlp=128: 592 + 128*123 + 10 = 16,346 non-emb (16,698 total)
        ("no_gram_reshape_1h f3(4,4,4) K=4 mlp=128 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 128, n_queries=4)),
        # K=4 mlp=256: 592 + 256*123 + 10 = 32,090 non-emb (32,442 total)
        ("no_gram_reshape_1h f3(4,4,4) K=4 mlp=256 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 256, n_queries=4)),

        # ── 1h, K=8 ───────────────────────────────────────────────────────────
        # K=8 mlp=64:  1184 + 64*235 + 10 = 16,234 non-emb (16,586 total)
        ("no_gram_reshape_1h f3(4,4,4) K=8 mlp=64 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 64,  n_queries=8)),
        # K=8 mlp=128: 1184 + 128*235 + 10 = 31,274 non-emb (31,626 total)
        ("no_gram_reshape_1h f3(4,4,4) K=8 mlp=128 +noise30",
         *make_no_gram_reshape_1h(4,4,4, 128, n_queries=8)),

        # ── 2h, K=4 higher MLP ────────────────────────────────────────────────
        # K=4 mlp=64:  928 + 64*235 + 10 = 15,978 non-emb (16,330 total)
        ("no_gram_reshape_2h f3(4,4,4) K=4 mlp=64 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 64,  n_queries=4)),
        # K=4 mlp=128: 928 + 128*235 + 10 = 31,018 non-emb (31,370 total)
        ("no_gram_reshape_2h f3(4,4,4) K=4 mlp=128 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 128, n_queries=4)),

        # ── 2h, K=8 ───────────────────────────────────────────────────────────
        # K=8 mlp=64:  1856 + 64*459 + 10 = 31,242 non-emb (31,594 total)
        ("no_gram_reshape_2h f3(4,4,4) K=8 mlp=64 +noise30",
         *make_no_gram_reshape_2h(4,4,4, 64,  n_queries=8)),
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
            "name": label, "experiment": "exp24",
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
