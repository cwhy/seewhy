"""
exp21 — Ablation: what does the Gram matrix M contribute?

Two baselines that strip away M:

  A. no_gram_bilinear  — same f3(4,4,4) embedding, but instead of forming
     M = Σ_t emb_t ⊗ emb_t ∈ R^{d×d}, directly read H @ h_answer_k where
     H ∈ R^{B×784×d}:

         for each query k:
             scores_k = einsum("btd,d->bt", H, h_answer_k)  # (B, 784)
         features = concat(scores_k)   ∈ R^{B×784*K}
         logits   = MLP(LN(features))

     non_emb = K*d + 2*(784*K) + mlp*(784*K) + mlp*11 + 10

  B. conv_mlp  — no embeddings at all; standard MLP on normalised pixel values:

         x ∈ [0,1]^{B×784}    (noise30 aug applied before normalising)
         h = GELU(W1 @ x + b1)
         logits = W2 @ h + b2

     All params are "non-emb" (no embedding lookup).

Both use noise30 augmentation. Results tagged experiment="exp21".
n_total_params is stored alongside n_nonemb_params.
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

def replot():
    subprocess.run([sys.executable, str(PLOT)], check=False)

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


# ── factored3 embedding helper ─────────────────────────────────────────────────

def _f3_emb(params, x, d_r, d_c, d_v, n_val_bins):
    B    = x.shape[0]
    bins = jnp.floor(x / 255.0 * (n_val_bins - 1)).astype(jnp.int32)
    re   = params["row_emb"][ROWS]                                         # (784, d_r)
    ce   = params["col_emb"][COLS]                                         # (784, d_c)
    ve   = params["val_emb"][bins]                                         # (B, 784, d_v)
    pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)      # (784, d_r*d_c)
    emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(
               B, 784, d_r * d_c * d_v)                                   # (B, 784, d)
    return emb


# ── model A: no-Gram bilinear ──────────────────────────────────────────────────

def make_no_gram_bilinear(d_r, d_c, d_v, mlp_dim, n_queries=1,
                          n_val_bins=N_VAL_BINS_DEFAULT):
    """f3 embedding + direct H @ h_answer readout (no Gram matrix formed).

    H ∈ R^{B×784×d};  for each query k: scores_k = H @ h_answer_k ∈ R^{B×784}
    features = concat(scores_k) ∈ R^{B×784*K},  LN,  GELU-MLP → 10

    non_emb = K*d + 2*(784*K) + mlp*(784*K) + mlp + 10*mlp + 10
    emb     = 28*d_r + 28*d_c + n_val_bins*d_v    (tiny for f3)
    """
    d        = d_r * d_c * d_v
    feat_dim = n_queries * 784
    emb_keys = frozenset({"row_emb", "col_emb", "val_emb"})

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "row_emb":    jax.random.normal(next(kg).get(), (28, d_r))            * 0.02,
        "col_emb":    jax.random.normal(next(kg).get(), (28, d_c))            * 0.02,
        "val_emb":    jax.random.normal(next(kg).get(), (n_val_bins, d_v))    * 0.02,
        "H_answer":   jax.random.normal(next(kg).get(), (n_queries, d))       * 0.02,
        "ln_gamma":   jnp.ones(feat_dim),
        "ln_beta":    jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }

    def fwd(params, x):
        emb = _f3_emb(params, x, d_r, d_c, d_v, n_val_bins)        # (B, 784, d)
        scores = []
        for k in range(n_queries):
            ha = params["H_answer"][k]                               # (d,)
            scores.append(jnp.einsum("btd,d->bt", emb, ha))         # (B, 784)
        features = jnp.concatenate(scores, axis=-1)                  # (B, 784*K)
        features = layer_norm(features, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, emb_keys


# ── model B: conventional MLP on raw pixels ────────────────────────────────────

def make_conv_mlp(mlp_dim):
    """Standard 1-hidden-layer GELU MLP on normalised pixel values [0,1].

    Input:  x ∈ [0,1]^{B×784}   (raw pixels / 255, after noise aug)
    Head:   GELU(W1 @ x + b1) → W2 → 10

    All params are non-emb (no embedding lookup).
    non_emb = total = 784*mlp + mlp + 10*mlp + 10
    """
    emb_keys = frozenset()
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, 784)) / 784**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))  * 0.01,
        "b2": jnp.zeros(10),
    }

    def fwd(params, x):
        x_norm = x / 255.0                                           # (B, 784)
        h = jax.nn.gelu(x_norm @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, emb_keys


# ── training loop ──────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          max_epochs=30, aug=AUG_NOISE30):
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

    return train_acc, test_acc, time.perf_counter() - t0, epoch + 1


if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    experiments = [
        # ── A: no-Gram bilinear, f3(4,4,4), K=1 ─────────────────────────────
        # non_emb = 1*64 + 2*784 + mlp*784 + mlp*11 + 10
        ("no_gram f3(4,4,4) K=1 mlp=8 +noise30",
         *make_no_gram_bilinear(4, 4, 4, 8),
         "ablation-A K=1"),

        ("no_gram f3(4,4,4) K=1 mlp=16 +noise30",
         *make_no_gram_bilinear(4, 4, 4, 16),
         "ablation-A K=1"),

        ("no_gram f3(4,4,4) K=1 mlp=32 +noise30",
         *make_no_gram_bilinear(4, 4, 4, 32),
         "ablation-A K=1"),

        ("no_gram f3(4,4,4) K=1 mlp=64 +noise30",
         *make_no_gram_bilinear(4, 4, 4, 64),
         "ablation-A K=1"),

        ("no_gram f3(4,4,4) K=1 mlp=128 +noise30",
         *make_no_gram_bilinear(4, 4, 4, 128),
         "ablation-A K=1"),

        ("no_gram f3(4,4,4) K=1 mlp=256 +noise30",
         *make_no_gram_bilinear(4, 4, 4, 256),
         "ablation-A K=1"),

        # ── A: no-Gram bilinear, K=4 ─────────────────────────────────────────
        ("no_gram f3(4,4,4) K=4 mlp=32 +noise30",
         *make_no_gram_bilinear(4, 4, 4, 32, n_queries=4),
         "ablation-A K=4"),

        ("no_gram f3(4,4,4) K=4 mlp=64 +noise30",
         *make_no_gram_bilinear(4, 4, 4, 64, n_queries=4),
         "ablation-A K=4"),

        # ── B: conventional MLP on raw pixels ────────────────────────────────
        # total = non_emb = 784*mlp + mlp*11 + 10
        ("conv_mlp mlp=16 +noise30",
         *make_conv_mlp(16),
         "ablation-B"),

        ("conv_mlp mlp=32 +noise30",
         *make_conv_mlp(32),
         "ablation-B"),

        ("conv_mlp mlp=64 +noise30",
         *make_conv_mlp(64),
         "ablation-B"),

        ("conv_mlp mlp=128 +noise30",
         *make_conv_mlp(128),
         "ablation-B"),

        ("conv_mlp mlp=256 +noise30",
         *make_conv_mlp(256),
         "ablation-B"),

        ("conv_mlp mlp=512 +noise30",
         *make_conv_mlp(512),
         "ablation-B"),
    ]

    results = []
    for label, fwd, params, emb_keys, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  total={ntotal:,}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
        )
        row = {
            "name": label, "experiment": "exp21",
            "n_nonemb_params": nonemb,
            "n_total_params":  ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, ntotal, tr_acc, te_acc, elapsed))
        logging.info(f"  → train={tr_acc:.4f}  test={te_acc:.4f}")

    results.sort(key=lambda r: r[1])
    print(f"\n{'='*110}")
    print(f"{'Variant':<48} {'NonEmb':>8} {'Total':>8} {'Train':>8} {'Test':>8} {'Time':>8}")
    print(f"{'-'*110}")
    for label, nonemb, ntotal, tr, te, t in results:
        print(f"{label:<48} {nonemb:>8,} {ntotal:>8,} {tr:>8.4f} {te:>8.4f} {t:>7.1f}s")
    print(f"{'='*110}")

    replot()
