"""
Fashion-MNIST fashion6: joint larger MLP + bins sweep; flat_gram d=16 bins sweep.

Three directions:

  A. joint(d=16) larger MLP (bins=32, noise=30)
     Best joint so far: K=4 mlp=256 → 88.84% at ~420K total
     Push K=4 mlp=512/1024 and K=8 depth=2 mlp=256/512 to see if MLP is the bottleneck.

  B. joint(d=16) bins sweep (K=4, mlp=256, noise=30)
     joint emb = 784 * n_val_bins * d  — bins directly multiply emb size:
       bins=64  → emb ≈ 803K  total ≈ 822K
       bins=128 → emb ≈ 1.6M  total ≈ 1.625M
     Does richer per-position grayscale lookup help despite the emb cost?

  C. flat_gram d=16 mlp=512 bins sweep (noise=15)
     Best flat_gram: d=16 mlp=512 bins=32 → 90.17% at 137K
     Bins barely change total params (val_emb = n_val_bins * d_v = n_val_bins * 1):
       +32/+96/+224 params for bins=64/128/256 vs bins=32

Reference (bins=32):
  joint(d=16) K=4  mlp=256            → 88.84%  total ~420K
  flat_gram d=16   mlp=512            → 90.17%  total ~137K
  gram_f3 K=8 d2   mlp=256 bins=256  → 90.17%  total ~137K
  gram_f3 K=8      mlp=512 bins=256  → 90.23%  total ~271K  ← best gram_f3

Usage:
    uv run python projects/rnn/experiments_fashion6.py
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

BATCH_SIZE = 128
LR         = 1e-3
LR_MUON    = 0.02
SEED       = 42

JSONL = Path(__file__).parent / "results_fashion_mnist.jsonl"
PLOT  = Path(__file__).parent / "plot_pareto_fashion.py"

ROWS  = jnp.arange(784) // 28
COLS  = jnp.arange(784) % 28
T_IDX = jnp.arange(784)


# ── utilities ─────────────────────────────────────────────────────────────────

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


# ── f3 embedding (configurable bins) ──────────────────────────────────────────

def _f3_emb(params, x, d_r, d_c, d_v, n_val_bins):
    bins = jnp.floor(x / 255.0 * (n_val_bins - 1)).astype(jnp.int32)
    re   = params["row_emb"][ROWS]
    ce   = params["col_emb"][COLS]
    ve   = params["val_emb"][bins]
    pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
    return (pos[None, :, :, None] * ve[:, :, None, :]).reshape(
               x.shape[0], 784, d_r * d_c * d_v)

def _f3_params(d_r, d_c, d_v, n_val_bins, kg):
    return {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))            * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))            * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (n_val_bins, d_v))    * 0.02,
    }

F3_EMB_KEYS    = frozenset({"row_emb", "col_emb", "val_emb"})
JOINT_EMB_KEYS = frozenset({"joint_emb"})


# ── A/B: joint(d) bilinear with configurable bins ─────────────────────────────

def make_joint_bilinear(d, mlp_dim, n_queries=1, depth=1, n_val_bins=32):
    """Full joint embedding E[position, bin] + Gram bilinear readout.
    emb     = 784 * n_val_bins * d   (grows linearly with bins!)
    non_emb = K*d*(3+mlp) + depth*2*d + 11*mlp + 10
    """
    feat_dim = n_queries * d
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "joint_emb":    jax.random.normal(next(kg).get(), (784, n_val_bins, d)) * 0.02,
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
        bins = jnp.floor(x / 255.0 * (n_val_bins - 1)).astype(jnp.int32)
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


# ── C: flat_gram with configurable bins ───────────────────────────────────────

def make_flat_gram(d_r, d_c, d_v, mlp_dim, n_val_bins=32):
    """Gram matrix M flattened to d² features → LN → MLP.
    emb     = 28*d_r + 28*d_c + n_val_bins*d_v   (n_val_bins barely changes total)
    non_emb = d²*(2+mlp) + 11*mlp + 10
    """
    d        = d_r * d_c * d_v
    feat_dim = d * d
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        **_f3_params(d_r, d_c, d_v, n_val_bins, kg),
        "ln_gamma": jnp.ones(feat_dim),
        "ln_beta":  jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }

    def fwd(params, x):
        B   = x.shape[0]
        emb = _f3_emb(params, x, d_r, d_c, d_v, n_val_bins)
        M   = jnp.einsum("bti,btj->bij", emb, emb)
        features = M.reshape(B, feat_dim)
        features = layer_norm(features, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, F3_EMB_KEYS


# ── training loop ─────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          max_epochs=40, noise_scale=15.0):
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
        # ── A: joint(d=16) larger MLP — K=4, bins=32 ─────────────────────────
        # baseline: K=4 mlp=256 → 88.84%  total ~420K
        # non_emb ≈ 234 + 75*mlp  (W1 is mlp×64, manageable for Muon)
        ("joint d=16 K=4 mlp=512 +noise30",
         *make_joint_bilinear(16, 512,  n_queries=4, n_val_bins=32), 30.0, "gram_joint"),
        ("joint d=16 K=4 mlp=1024 +noise30",
         *make_joint_bilinear(16, 1024, n_queries=4, n_val_bins=32), 30.0, "gram_joint"),

        # depth=2 variants: feat_dim=K*d=128 for K=8 → W1 is mlp×128 (better ratio)
        ("joint d=16 K=8 d2 mlp=256 +noise30",
         *make_joint_bilinear(16, 256,  n_queries=8, depth=2, n_val_bins=32), 30.0, "gram_joint"),
        ("joint d=16 K=8 d2 mlp=512 +noise30",
         *make_joint_bilinear(16, 512,  n_queries=8, depth=2, n_val_bins=32), 30.0, "gram_joint"),
        ("joint d=16 K=8 d2 mlp=1024 +noise30",
         *make_joint_bilinear(16, 1024, n_queries=8, depth=2, n_val_bins=32), 30.0, "gram_joint"),

        # ── B: joint(d=16) bins sweep — K=4, mlp=256 ─────────────────────────
        # emb = 784 * bins * 16; bins=64 → ~803K emb, bins=128 → ~1.6M emb
        ("joint d=16 K=4 mlp=256 bins=64 +noise30",
         *make_joint_bilinear(16, 256, n_queries=4, n_val_bins=64),  30.0, "gram_joint"),
        ("joint d=16 K=4 mlp=256 bins=128 +noise30",
         *make_joint_bilinear(16, 256, n_queries=4, n_val_bins=128), 30.0, "gram_joint"),

        # ── C: flat_gram d=16 mlp=512 bins sweep ─────────────────────────────
        # Best flat_gram: d=16 mlp=512 bins=32 → 90.17% at 137K
        # val_emb = bins * d_v = bins * 1 → +32/+96/+224 total params vs bins=32
        ("flat_gram f3(4,4,1) d=16 mlp=512 bins=64 +noise15",
         *make_flat_gram(4,4,1, 512, n_val_bins=64),  15.0, "flat_gram"),
        ("flat_gram f3(4,4,1) d=16 mlp=512 bins=128 +noise15",
         *make_flat_gram(4,4,1, 512, n_val_bins=128), 15.0, "flat_gram"),
        ("flat_gram f3(4,4,1) d=16 mlp=512 bins=256 +noise15",
         *make_flat_gram(4,4,1, 512, n_val_bins=256), 15.0, "flat_gram"),
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
            "name": label, "experiment": "fashion6",
            "family": family,
            "n_nonemb_params": nonemb,
            "n_total_params":  ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": 40, "time_s": round(elapsed, 1),
        }
        append_result(row)
        results.append((label, nonemb, ntotal, tr_acc, te_acc, family))
        logging.info(f"  → train={tr_acc:.4f}  test={te_acc:.4f}")

    results.sort(key=lambda r: r[2])
    print(f"\n{'='*120}")
    print(f"{'Variant':<60} {'Family':<11} {'NonEmb':>9} {'Total':>9} {'Train':>8} {'Test':>8}")
    print(f"{'-'*120}")
    for label, nonemb, ntotal, tr, te, fam in results:
        print(f"{label:<60} {fam:<11} {nonemb:>9,} {ntotal:>9,} {tr:>8.4f} {te:>8.4f}")
    print(f"{'='*120}")

    subprocess.run([sys.executable, str(PLOT)], check=False)
