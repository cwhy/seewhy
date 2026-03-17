"""
Pareto tuning round 15: smaller joint embedding dimensions (d=4, 8, 16).

Prior work only tested joint(d=32). Here we sweep d=4/8/16 with the bilinear
multi-query readout. Smaller d → smaller M (d×d Gram), but cheaper non-emb
params → can afford wider MLP at the same budget.

Embedding table sizes:
  joint(d=32): 784 * 32 * 32 = 802,816 emb params
  joint(d=16): 784 * 32 * 16 = 401,408 emb params
  joint(d=8):  784 * 32 *  8 = 200,704 emb params
  joint(d=4):  784 * 32 *  4 = 100,352 emb params

Key isolating comparisons (exact same non-emb budget):
  joint(d=16) K=1 mlp=16 (522)  vs  f3(2,2,4) K=1 mlp=16 (522 → 85.75%)
  joint(d=16) K=2 mlp=16 (826)  vs  f3(2,2,4) K=2 mlp=16 (826 → 90.63%)
  joint(d=8)  K=1 mlp=64 (1266) vs  f3(2,2,2) K=1 mlp=64 (1266 → 82.85%)
  joint(d=8)  K=1 mlp=128(2482) vs  f3(2,2,2) K=1 mlp=128(2482 → 87.41%)

These pairs share identical non-emb budget and identical emb_dim — the only
difference is joint (full per-position lookup) vs factored (outer product).
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


def make_joint_bilinear(d, mlp_dim, n_queries=1, depth=1,
                        n_val_bins=N_VAL_BINS_DEFAULT):
    """joint(d) embedding + deep multi-query bilinear readout.

    Embedding: E[position, bin] ∈ R^d, table shape (784, K, d).
    Gram:      M = Σ_t emb_t ⊗ emb_t  ∈ R^{d×d}
    Readout:   features = concat([LN(M @ h_k) for k in queries])  ∈ R^{K*d}
    Head:      GELU MLP → 10 logits

    non_emb = K*d*(3 + mlp) + depth*2*d + 11*mlp + 10
    emb     = 784 * n_val_bins * d
    """
    feat_dim = n_queries * d
    emb_keys = frozenset({"joint_emb"})

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
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
        K    = params["joint_emb"].shape[1]
        bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
        emb  = params["joint_emb"][T_IDX, bins]                      # (B, 784, d)
        M    = jnp.einsum("bti,btj->bij", emb, emb)                  # (B, d, d)
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

    return fwd, p, emb_keys


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
        # ── joint(d=4): extreme low end ───────────────────────────────────────
        ("joint(d=4) K=1 mlp=16 +noise30",
         *make_joint_bilinear(4, 16),
         "270 non-emb; 100K emb table; sub-300 territory"),

        ("joint(d=4) K=4 mlp=16 +noise30",
         *make_joint_bilinear(4, 16, n_queries=4),
         "498 non-emb; K=4 at tiny budget; vs f3(2,2,4) K=1 mlp=16 (522→85.75%)"),

        ("joint(d=4) K=1 mlp=32 +noise30",
         *make_joint_bilinear(4, 32),
         "510 non-emb; vs f3(2,2,4) K=1 mlp=16 (522→85.75%)"),

        ("joint(d=4) K=4 mlp=32 +noise30",
         *make_joint_bilinear(4, 32, n_queries=4),
         "930 non-emb; K=4 queries d=4"),

        # ── joint(d=8) ────────────────────────────────────────────────────────
        ("joint(d=8) K=1 mlp=32 +noise30",
         *make_joint_bilinear(8, 32),
         "658 non-emb; new territory"),

        ("joint(d=8) K=1 mlp=64 +noise30",     # EXACT budget match
         *make_joint_bilinear(8, 64),
         "1266 non-emb; EXACT budget as f3(2,2,2) K=1 mlp=64 (1266→82.85%): joint vs factored at d=8"),

        ("joint(d=8) K=4 mlp=32 +noise30",
         *make_joint_bilinear(8, 32, n_queries=4),
         "1498 non-emb; K=4 d=8; vs joint(d=32) K=1 mlp=32 (1546→96.62%)"),

        ("joint(d=8) K=1 mlp=128 +noise30",    # EXACT budget match
         *make_joint_bilinear(8, 128),
         "2482 non-emb; EXACT budget as f3(2,2,2) K=1 mlp=128 (2482→87.41%): joint vs factored at d=8"),

        # ── joint(d=16) ───────────────────────────────────────────────────────
        ("joint(d=16) K=1 mlp=16 +noise30",    # EXACT budget match
         *make_joint_bilinear(16, 16),
         "522 non-emb; EXACT budget as f3(2,2,4) K=1 mlp=16 (522→85.75%): joint vs factored at d=16"),

        ("joint(d=16) K=2 mlp=16 +noise30",    # EXACT budget match
         *make_joint_bilinear(16, 16, n_queries=2),
         "826 non-emb; EXACT budget as f3(2,2,4) K=2 mlp=16 (826→90.63%): joint vs factored at d=16"),

        ("joint(d=16) K=1 mlp=32 +noise30",
         *make_joint_bilinear(16, 32),
         "954 non-emb; new territory"),

        ("joint(d=16) K=2 mlp=32 +noise30",
         *make_joint_bilinear(16, 32, n_queries=2),
         "1514 non-emb; can d=16 K=2 match joint(d=32) K=1 mlp=32 (1546→96.62%)?"),

        ("joint(d=16) K=1 mlp=64 +noise30",
         *make_joint_bilinear(16, 64),
         "1818 non-emb; fill gap between 1546 and 2922"),

        ("joint(d=16) K=4 mlp=32 +noise30",
         *make_joint_bilinear(16, 32, n_queries=4),
         "2634 non-emb; K=4 with d=16"),

        ("joint(d=16) K=2 mlp=64 +noise30",
         *make_joint_bilinear(16, 64, n_queries=2),
         "2890 non-emb; vs joint(d=32) K=1 mlp=64 (2922→97.39%)"),

        ("joint(d=16) K=1 mlp=128 +noise30",
         *make_joint_bilinear(16, 128),
         "3546 non-emb; fill gap 2922→5130"),
    ]

    baselines = {
        270:   ("f3(2,2,4) K=1 mlp=8",          0.8492),
        522:   ("f3(2,2,4) K=1 mlp=16",         0.8575),
        826:   ("f3(2,2,4) K=2 mlp=16",         0.9063),
        858:   ("joint(d=32) K=1 mlp=16",       0.9531),
        1_266: ("f3(2,2,2) K=1 mlp=64",         0.8285),
        1_546: ("joint(d=32) K=1 mlp=32",       0.9662),
        2_130: ("f3(4,4,4) K=1 mlp=24",         0.9680),
        2_482: ("f3(2,2,2) K=1 mlp=128",        0.8741),
        2_922: ("joint(d=32) K=1 mlp=64",       0.9739),
        5_130: ("f3(4,4,4) K=1 mlp=64",         0.9772),
    }

    results = []
    for label, fwd, params, emb_keys, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        total  = n_total(params)
        closest = min(baselines.keys(), key=lambda x: abs(x - nonemb))
        base_name, base_acc = baselines[closest]
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  emb={total-nonemb:,}  "
            f"baseline: {base_name}={base_acc:.4f}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
        )
        row = {
            "name": label, "experiment": "exp20",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, total-nonemb, tr_acc, te_acc, elapsed,
                        base_acc, base_name))
        logging.info(f"  → {te_acc:.4f}  vs {base_name} {base_acc:.4f}: {te_acc-base_acc:+.4f}")

    results.sort(key=lambda r: r[1])
    print(f"\n{'='*120}")
    print(f"{'Variant':<48} {'NonEmb':>7} {'Emb':>8} {'Train':>8} {'Test':>8} {'Delta':>7} {'Time':>8}")
    print(f"{'-'*120}")
    for label, nonemb, emb, tr, te, t, base, bname in results:
        flag  = " ★" if te > base else "  "
        print(f"{label:<48} {nonemb:>7,} {emb:>8,} {tr:>7.4f} {te:>7.4f}{flag} {te-base:>+7.4f} {t:>7.1f}s")
    print(f"{'='*120}")

    # Pareto front
    prior = [(306,0.8492),(522,0.8575),(826,0.9063),(858,0.9531),(930,0.9243),
             (1530,0.9590),(1546,0.9662),(2130,0.9680),(2922,0.9739),
             (5130,0.9772),(7530,0.9794),(18314,0.9828),(18442,0.9840),(35082,0.9854)]
    combined = sorted(prior + [(r[1],r[4]) for r in results], key=lambda x: x[0])
    print("\nPareto front:")
    best = -1
    for ne, acc in combined:
        if acc > best:
            best = acc
            print(f"  {ne:>8,}  {acc:.4f}")

    replot()
