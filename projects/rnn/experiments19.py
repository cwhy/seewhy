"""
Pareto tuning round 14: smaller embedding dimensions for the bilinear architecture.

exp18 established the low-end Pareto using f3(4,4,4) (emb_dim=64).
Here we try f3(2,4,4) (emb_dim=32), f3(2,2,4) (emb_dim=16), f3(2,2,2) (emb_dim=8).

Key insight: with bilinear readout h = M @ h_answer ∈ R^{emb_dim},
the MLP input is emb_dim (NOT emb_dim²). So smaller emb_dim → cheaper W1.
At the same non-emb budget, a smaller emb_dim allows a WIDER MLP.

Trade-off: smaller emb_dim → less expressive M (e.g. 8×8 vs 64×64 Gram) but
more MLP capacity for the same param budget.

─── Key isolating comparisons ───────────────────────────────────────────────
f3(2,4,4) emb_dim=32 vs joint(d=32):
  Both have emb_dim=32, same MLP cost — isolates embedding quality vs factorization.
  f3(2,4,4) K=1 mlp=16  = 858 non-emb  [joint K=1 mlp=16: 95.31%]
  f3(2,4,4) K=1 mlp=32  = 1,546 non-emb  [joint K=1 mlp=32: 96.62%]
  f3(2,4,4) K=1 mlp=64  = 2,922 non-emb  [joint K=1 mlp=64: 97.39%]

f3(2,2,4) emb_dim=16 multi-query vs joint single-query at same budget:
  f3(2,2,4) K=2 mlp=16 = 826 non-emb  [joint K=1 mlp=16: 95.31%]
  f3(2,2,4) K=2 mlp=32 = 1,514 non-emb  [joint K=1 mlp=32: 96.62%]
  f3(2,2,4) K=2 mlp=64 = 2,890 non-emb  [joint K=1 mlp=64: 97.39%]

f3(2,2,2) emb_dim=8 — extreme small state (8×8 Gram = 64-dim):
  K=1/4 at various MLP widths; wide MLP compensates tiny emb_dim?

─── Non-emb formula ─────────────────────────────────────────────────────────
non_emb = K*d           (H_answer, 1D→Adam if K=1, 2D→Muon if K>1)
        + 2*K*d         (ln_out on K*d-dim features)
        + depth*2*d     (per-hop LN, shared across queries)
        + K*d*mlp       (W1)
        + mlp           (b1)
        + 10*mlp        (W2)
        + 10            (b2)
= K*d*(1 + 2 + mlp) + depth*2*d + mlp*11 + 10
= K*d*(3 + mlp) + depth*2*d + 11*mlp + 10
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


def make_f3_bilinear(d_r, d_c, d_v, mlp_dim, n_queries=1, depth=1,
                     n_val_bins=N_VAL_BINS_DEFAULT):
    """factored3 embedding + deep multi-query bilinear readout."""
    emb_dim  = d_r * d_c * d_v
    feat_dim = n_queries * emb_dim
    emb_keys = frozenset({"row_emb", "col_emb", "val_emb"})

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))                 * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))                 * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (n_val_bins, d_v))         * 0.02,
        "H_answer":     jax.random.normal(next(kg).get(), (n_queries, emb_dim)) * 0.02,
        "ln_out_gamma": jnp.ones(feat_dim),
        "ln_out_beta":  jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }
    for i in range(depth):
        p[f"ln_hop{i}_gamma"] = jnp.ones(emb_dim)
        p[f"ln_hop{i}_beta"]  = jnp.zeros(emb_dim)

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS]
        ce   = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, emb_dim)
        M    = jnp.einsum("bti,btj->bij", emb, emb)
        outs = []
        for q in range(n_queries):
            h = jnp.broadcast_to(params["H_answer"][q], (B, emb_dim))
            for i in range(depth):
                h = jnp.einsum("bij,bj->bi", M, h)
                h = layer_norm(h, params[f"ln_hop{i}_gamma"], params[f"ln_hop{i}_beta"])
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
        # ── f3(2,4,4) emb_dim=32: direct comparison with joint(d=32) ─────────
        # Same MLP input dim, same non-emb — isolates embedding quality.
        ("f3(2,4,4) bilinear K=1 mlp=16 +noise30",
         *make_f3_bilinear(2, 4, 4, 16),
         "858 non-emb; vs joint K=1 mlp=16 95.31%: does factorization hurt?"),

        ("f3(2,4,4) bilinear K=1 mlp=32 +noise30",
         *make_f3_bilinear(2, 4, 4, 32),
         "1546 non-emb; vs joint K=1 mlp=32 96.62%"),

        ("f3(2,4,4) bilinear K=1 mlp=64 +noise30",
         *make_f3_bilinear(2, 4, 4, 64),
         "2922 non-emb; vs joint K=1 mlp=64 97.39%"),

        ("f3(2,4,4) bilinear K=2 mlp=32 +noise30",
         *make_f3_bilinear(2, 4, 4, 32, n_queries=2),
         "2666 non-emb; K=2 queries with emb_dim=32"),

        ("f3(2,4,4) bilinear K=4 mlp=32 +noise30",
         *make_f3_bilinear(2, 4, 4, 32, n_queries=4),
         "4906 non-emb; K=4 queries with emb_dim=32"),

        # ── f3(2,2,4) emb_dim=16: smaller state, cheaper MLP per neuron ──────
        # With emb_dim=16, W1 is 4× cheaper than emb_dim=64 at same mlp width.
        ("f3(2,2,4) bilinear K=1 mlp=8 +noise30",
         *make_f3_bilinear(2, 2, 4, 8),
         "306 non-emb; extreme tiny; does emb_dim=16 avoid mlp=4 collapse?"),

        ("f3(2,2,4) bilinear K=1 mlp=16 +noise30",
         *make_f3_bilinear(2, 2, 4, 16),
         "522 non-emb"),

        ("f3(2,2,4) bilinear K=2 mlp=16 +noise30",
         *make_f3_bilinear(2, 2, 4, 16, n_queries=2),
         "826 non-emb; vs joint K=1 mlp=16 95.31%: K=2 with d=16 vs K=1 with d=32?"),

        ("f3(2,2,4) bilinear K=2 mlp=32 +noise30",
         *make_f3_bilinear(2, 2, 4, 32, n_queries=2),
         "1514 non-emb; vs joint K=1 mlp=32 96.62%"),

        ("f3(2,2,4) bilinear K=2 mlp=64 +noise30",
         *make_f3_bilinear(2, 2, 4, 64, n_queries=2),
         "2890 non-emb; vs joint K=1 mlp=64 97.39%"),

        ("f3(2,2,4) bilinear K=4 mlp=32 +noise30",
         *make_f3_bilinear(2, 2, 4, 32, n_queries=4),
         "2634 non-emb; K=4 queries, emb_dim=16, feat=64"),

        ("f3(2,2,4) bilinear K=4 mlp=64 +noise30",
         *make_f3_bilinear(2, 2, 4, 64, n_queries=4),
         "5034 non-emb; vs f3(4,4,4) K=1 mlp=64 97.72%"),

        # ── f3(2,2,2) emb_dim=8: tiny state (8×8 Gram), very wide MLP ────────
        ("f3(2,2,2) bilinear K=1 mlp=24 +noise30",
         *make_f3_bilinear(2, 2, 2, 24),
         "506 non-emb; d=8; avoids mlp=4 collapse with wider MLP?"),

        ("f3(2,2,2) bilinear K=1 mlp=64 +noise30",
         *make_f3_bilinear(2, 2, 2, 64),
         "1266 non-emb; d=8 mlp=64 vs f3(4,4,4) K=1 mlp=16 at 1530"),

        ("f3(2,2,2) bilinear K=1 mlp=128 +noise30",
         *make_f3_bilinear(2, 2, 2, 128),
         "2482 non-emb; d=8 wide MLP vs f3(4,4,4) K=1 mlp=24 2130→96.80%"),

        ("f3(2,2,2) bilinear K=4 mlp=64 +noise30",
         *make_f3_bilinear(2, 2, 2, 64, n_queries=4),
         "2874 non-emb; K=4 queries d=8, feat=32"),

        ("f3(2,2,2) bilinear K=4 mlp=128 +noise30",
         *make_f3_bilinear(2, 2, 2, 128, n_queries=4),
         "5626 non-emb; K=4 d=8 large MLP"),
    ]

    # baselines from exp18 low-end sweep
    baselines = {
        858:   ("joint K=1 mlp=16",         0.9531),
        1_546: ("joint K=1 mlp=32",         0.9662),
        2_130: ("f3(4,4,4) K=1 mlp=24",     0.9680),
        2_922: ("joint K=1 mlp=64",         0.9739),
        5_130: ("f3(4,4,4) K=1 mlp=64",     0.9772),
        7_530: ("f3(4,4,4) K=1 mlp=96",     0.9794),
    }

    results = []
    for label, fwd, params, emb_keys, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        total  = n_total(params)
        closest = min(baselines.keys(), key=lambda x: abs(x - nonemb))
        base_name, base_acc = baselines[closest]
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  emb={total-nonemb:,}  aug=noise30\n"
            f"  baseline: {base_name} = {base_acc:.4f}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
        )
        row = {
            "name": label, "experiment": "exp19",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, elapsed, base_acc, base_name))
        logging.info(f"  → {te_acc:.4f}  vs {base_name} {base_acc:.4f}: {te_acc-base_acc:+.4f}")

    results.sort(key=lambda r: r[1])
    print(f"\n{'='*116}")
    print(f"{'Variant':<50} {'NonEmb':>7} {'Train':>8} {'Test':>8} {'Delta':>7} {'Time':>8}")
    print(f"{'-'*116}")
    for label, nonemb, tr, te, t, base, bname in results:
        flag  = " ★" if te > base else "  "
        print(f"{label:<50} {nonemb:>7,} {tr:>7.4f} {te:>7.4f}{flag} {te-base:>+7.4f} {t:>7.1f}s")
    print(f"{'='*116}")

    # Print Pareto front
    all_res = sorted(results, key=lambda r: r[1])
    print("\nPareto front (new + prior):")
    prior = [(630,0.706),(858,0.9531),(930,0.9243),(1530,0.9590),(1546,0.9662),
             (2130,0.9680),(2922,0.9739),(4970,0.9769),(5130,0.9772),(7530,0.9794),
             (18314,0.9828),(18442,0.9840),(35082,0.9854)]
    combined = prior + [(r[1], r[3]) for r in results]
    combined.sort(key=lambda x: x[0])
    best_acc = -1
    for ne, acc in combined:
        if acc > best_acc:
            best_acc = acc
            print(f"  {ne:>8,}  {acc:.4f}")

    logging.info("Updating Pareto plot...")
    replot()
