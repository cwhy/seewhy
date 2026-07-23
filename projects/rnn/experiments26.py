"""
exp26 — f3 small-d: push d=8 and d=16 across the MNIST Pareto front range.

The existing Pareto front uses f3(4,4,4) d=64 for the medium/high range.
The outer Gram scales as K*d*(3+mlp) — using d=16 instead of d=64 makes
the outer model 4x cheaper, allowing much larger K/mlp at the same non-emb budget.

At 5 epochs (quick test):
  f3(4,4,1) d=16 K=4 mlp=64  → 5,034 nonemb, 97.2%  (vs f3(4,4,4) K=4 mlp=64: 17,994 nonemb, 97.7%)
  f3(4,4,1) d=16 K=4 mlp=128 → 9,834 nonemb, 97.6%  (vs f3(4,4,4) K=4 mlp=128: 35,082 nonemb, 98.5%)

Same accuracy, 3-4x fewer total params on x-axis.

X-axis strategy: plot on TOTAL params (non-emb + emb) so the advantage of smaller
emb (d=16: 256 emb vs d=64: 352 emb) is captured.

Usage:
    uv run python projects/rnn/experiments26.py
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

JSONL = Path(__file__).parent / "results_mnist.jsonl"
PLOT  = Path(__file__).parent / "plot_pareto_gram_embed.py"

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


F3_EMB_KEYS = frozenset({"row_emb", "col_emb", "val_emb"})


def make_f3_bilinear(d_r, d_c, d_v, mlp_dim, n_queries=1, depth=1):
    d        = d_r * d_c * d_v
    feat_dim = n_queries * d
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))         * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))         * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
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
        B    = x.shape[0]
        bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS]
        ce   = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, d)
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

    # non_emb = K*d*(3+mlp) + depth*2*d + 11*mlp + 10
    # emb     = 28*d_r + 28*d_c + 32*d_v
    #   d=8:  176 emb    d=16: 256 emb    d=64: 352 emb
    experiments = [
        # ── d=8 (f3 2,2,2) — ultra small end ─────────────────────────────────
        ("f3(2,2,2) d=8 K=2 mlp=16 +noise30",
         *make_f3_bilinear(2,2,2, 16,  n_queries=2)),
        ("f3(2,2,2) d=8 K=4 mlp=16 +noise30",
         *make_f3_bilinear(2,2,2, 16,  n_queries=4)),
        ("f3(2,2,2) d=8 K=4 mlp=32 +noise30",
         *make_f3_bilinear(2,2,2, 32,  n_queries=4)),
        ("f3(2,2,2) d=8 K=8 mlp=32 +noise30",
         *make_f3_bilinear(2,2,2, 32,  n_queries=8)),
        ("f3(2,2,2) d=8 K=8 mlp=64 +noise30",
         *make_f3_bilinear(2,2,2, 64,  n_queries=8)),

        # ── d=16 (f3 4,4,1) — main sweep ─────────────────────────────────────
        ("f3(4,4,1) d=16 K=1 mlp=8 +noise30",
         *make_f3_bilinear(4,4,1, 8,   n_queries=1)),
        ("f3(4,4,1) d=16 K=1 mlp=16 +noise30",
         *make_f3_bilinear(4,4,1, 16,  n_queries=1)),
        ("f3(4,4,1) d=16 K=2 mlp=16 +noise30",
         *make_f3_bilinear(4,4,1, 16,  n_queries=2)),
        ("f3(4,4,1) d=16 K=1 mlp=32 +noise30",
         *make_f3_bilinear(4,4,1, 32,  n_queries=1)),
        ("f3(4,4,1) d=16 K=2 mlp=32 +noise30",
         *make_f3_bilinear(4,4,1, 32,  n_queries=2)),
        ("f3(4,4,1) d=16 K=4 mlp=32 +noise30",
         *make_f3_bilinear(4,4,1, 32,  n_queries=4)),
        ("f3(4,4,1) d=16 K=4 mlp=64 +noise30",
         *make_f3_bilinear(4,4,1, 64,  n_queries=4)),
        ("f3(4,4,1) d=16 K=4 mlp=128 +noise30",
         *make_f3_bilinear(4,4,1, 128, n_queries=4)),
        ("f3(4,4,1) d=16 K=8 mlp=128 +noise30",
         *make_f3_bilinear(4,4,1, 128, n_queries=8)),
        ("f3(4,4,1) d=16 K=8 mlp=256 +noise30",
         *make_f3_bilinear(4,4,1, 256, n_queries=8)),
    ]

    results = []
    for label, fwd, params, emb_keys in experiments:
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(f"\n{'='*60}\n{label}\n"
                     f"  non-emb={nonemb:,}  total={ntotal:,}\n{'='*60}")
        tr_acc, te_acc, elapsed = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test)
        row = {
            "name": label, "experiment": "exp26",
            "family": "f3_small_d",
            "n_nonemb_params": nonemb,
            "n_total_params":  ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": 30, "time_s": round(elapsed, 1), "notes": "",
        }
        append_result(row)
        results.append((label, nonemb, ntotal, tr_acc, te_acc))
        logging.info(f"  → train={tr_acc:.4f}  test={te_acc:.4f}")

    results.sort(key=lambda r: r[2])
    print(f"\n{'='*105}")
    print(f"{'Variant':<52} {'NonEmb':>8} {'Total':>8} {'Train':>8} {'Test':>8}")
    print(f"{'-'*105}")
    for label, nonemb, ntotal, tr, te in results:
        print(f"{label:<52} {nonemb:>8,} {ntotal:>8,} {tr:>8.4f} {te:>8.4f}")
    print(f"{'='*105}")

    subprocess.run([sys.executable, str(PLOT)], check=False)
