"""
Pareto tuning round 8: joint (position × value) embedding.

New architecture: instead of the factored outer product
    emb_t = row_emb[r] ⊗ col_emb[c] ⊗ val_emb[bin]
use a direct joint lookup
    emb_t = joint_emb[t, bin]
where joint_emb ∈ R^{784 × n_bins × d} — one embedding vector per (position, bin) pair.

Key differences vs factored3:
  + No factorization assumption: position and value interact freely
  + More expressive: joint_emb has 784*32*d = 25,088*d emb params vs factored3's ~(28+28+32)*d
  - Much larger embedding table (embedding params don't count toward non-emb metric)
  - May overfit more → noise aug more important

Non-emb formula (same as factored3 at same d):
  state_dim = d^2
  non-emb = state_dim*(2 + mlp) + 11*mlp + 10

Planned comparisons (same non-emb as factored3 baselines):
  d=32 mlp=32   → ~35K  (factored3(2,4,4) mlp=32:  95.85%)
  d=32 mlp=64   → ~68K  (factored3(2,4,4) mlp=64:  96.43% +noise30)
  d=32 mlp=128  → ~135K (factored3(2,4,4) mlp=128: 96.90%)
  d=32 mlp=256  → ~267K (factored3(2,4,4) mlp=256: 97.11%)
  d=64 mlp=128  → ~534K (factored3(4,4,4) mlp=128: 98.46%)
  d=64 mlp=256  → ~1.06M (factored3(4,4,4) mlp=256: 98.65%)
  d=64 mlp=512  → ~2.11M (factored3(4,4,4) mlp=512: 98.81%)
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


# ── utils ─────────────────────────────────────────────────────────────────────

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


# ── augmentation ──────────────────────────────────────────────────────────────

def random_noise_batch(imgs, rng, scale):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)

AUG_NOISE15 = "noise15"
AUG_NOISE30 = "noise30"
_NOISE_SCALE = {AUG_NOISE15: 15.0, AUG_NOISE30: 30.0}


# ── Muon ──────────────────────────────────────────────────────────────────────

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

def make_adam_tx(schedule):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))


# ── model builders ─────────────────────────────────────────────────────────────

T_IDX = jnp.arange(784)


def make_joint_gelu(d, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """Joint (position × value) embedding.

    joint_emb[pos, bin] ∈ R^d — one vector per (pixel position, value bin) pair.
    emb_t = joint_emb[t, bin_t]
    M = Σ_t emb_t ⊗ emb_t  (same Gram matrix as factored3)
    → LayerNorm → GELU MLP → 10

    Embedding params: 784 * n_bins * d  (not counted in non-emb)
    Non-emb: d^2*(2 + mlp) + 11*mlp + 10  (identical to factored3 at same d)
    """
    state_dim = d * d
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        # 3D embedding table — indexed as [position, bin, feature]
        "joint_emb": jax.random.normal(
            next(kg).get(), (784, n_val_bins, d)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    # joint_emb is 3D → ndim != 2 → already excluded from Muon automatically
    # but list it explicitly for clarity
    emb_keys = frozenset({"joint_emb"})

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["joint_emb"].shape[1]
        bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)  # (B, 784)
        # emb[b, t, :] = joint_emb[t, bins[b, t], :]
        emb  = params["joint_emb"][T_IDX, bins]                   # (B, 784, d)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h    = jax.nn.gelu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, emb_keys


def make_factored3_gelu(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """factored3 baseline for comparison."""
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2
    ROWS = T_IDX // 28
    COLS = T_IDX % 28
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb":  jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb":  jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    emb_keys = frozenset({"row_emb", "col_emb", "val_emb"})

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K-1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS];  ce = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:,:,None] * ce[:,None,:]).reshape(784, d_r*d_c)
        emb  = (pos[None,:,:,None] * ve[:,:,None,:]).reshape(B, 784, emb_dim)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h    = jax.nn.gelu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, emb_keys


# ── training ──────────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          max_epochs=30, target_acc=0.995, aug=AUG_NOISE30, optimizer="muon"):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    adam_sched  = optax.cosine_decay_schedule(LR,      total_steps)
    muon_sched  = optax.cosine_decay_schedule(LR_MUON, total_steps)

    if optimizer == "muon":
        tx = make_muon_tx(params, emb_keys, adam_sched, muon_sched)
    else:
        tx = make_adam_tx(adam_sched)

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


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # Factored3 baselines (best known at each budget, from exp12)
    factored3_best = {
        35_178:    0.9585,   # (2,4,4) mlp=32  +noise15
        68_298:    0.9643,   # (2,4,4) mlp=64  +noise30
        134_538:   0.9690,   # (2,4,4) mlp=128 +noise30
        267_018:   0.9711,   # (2,4,4) mlp=256 +noise15
        533_898:   0.9846,   # (4,4,4) mlp=128 +noise30
        1_059_594: 0.9865,   # (4,4,4) mlp=256 +noise30
        2_110_986: 0.9881,   # (4,4,4) mlp=512 +noise30
    }

    # (label, fwd, params, emb_keys, aug, max_ep, opt, notes)
    experiments = [
        # ── Small budgets: joint vs factored3(2,4,4) ──────────────────────────
        # d=32 → state_dim=1024, same as factored3(2,4,4)

        ("joint(d=32) mlp=32 GELU +noise30 Muon",
         *make_joint_gelu(32, 32),
         AUG_NOISE30, 30, "muon",
         "joint emb ~35K, vs factored3(2,4,4) mlp=32 95.85%"),

        ("joint(d=32) mlp=64 GELU +noise30 Muon",
         *make_joint_gelu(32, 64),
         AUG_NOISE30, 30, "muon",
         "joint emb ~68K, vs factored3(2,4,4) mlp=64 96.43%"),

        ("joint(d=32) mlp=128 GELU +noise30 Muon",
         *make_joint_gelu(32, 128),
         AUG_NOISE30, 30, "muon",
         "joint emb ~135K, vs factored3(2,4,4) mlp=128 96.90%"),

        ("joint(d=32) mlp=256 GELU +noise30 Muon",
         *make_joint_gelu(32, 256),
         AUG_NOISE30, 30, "muon",
         "joint emb ~267K, vs factored3(2,4,4) mlp=256 97.11%"),

        # ── Large budgets: joint vs factored3(4,4,4) ──────────────────────────
        # d=64 → state_dim=4096, same as factored3(4,4,4)

        ("joint(d=64) mlp=128 GELU +noise30 Muon",
         *make_joint_gelu(64, 128),
         AUG_NOISE30, 30, "muon",
         "joint emb ~534K, vs factored3(4,4,4) mlp=128 98.46%"),

        ("joint(d=64) mlp=256 GELU +noise30 Muon",
         *make_joint_gelu(64, 256),
         AUG_NOISE30, 30, "muon",
         "joint emb ~1.06M, vs factored3(4,4,4) mlp=256 98.65%"),

        ("joint(d=64) mlp=512 GELU +noise30 Muon",
         *make_joint_gelu(64, 512),
         AUG_NOISE30, 30, "muon",
         "joint emb ~2.11M, vs factored3(4,4,4) mlp=512 98.81%"),
    ]

    results = []
    for label, fwd, params, emb_keys, aug, max_ep, opt_name, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        total  = n_total(params)
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  total={total:,}  "
            f"emb={total-nonemb:,}  opt={opt_name}  aug={aug}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug, optimizer=opt_name,
        )
        baseline = factored3_best.get(nonemb, 0.0)
        delta = f"{te_acc - baseline:+.4f}" if baseline else "new"
        row = {
            "name": label, "experiment": "exp13",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, epochs, elapsed, baseline))
        logging.info(f"  → saved  (vs factored3 baseline {baseline:.4f}: {delta})")

    print(f"\n{'='*108}")
    print(f"{'Variant':<48} {'NonEmb':>10} {'Train':>8} {'Test':>8} {'vs f3':>7} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*108}")
    for label, nonemb, tr, te, ep, t, base in results:
        delta = f"{te-base:+.4f}" if base else "  new"
        flag  = " ★" if te > base else (" ✗" if base and te < base else "  ")
        print(f"{label:<48} {nonemb:>10,} {tr:>7.4f} {te:>7.4f}{flag} {delta:>7} {ep:>4} {t:>7.1f}s")
    print(f"{'='*108}")

    logging.info("Updating Pareto plot...")
    replot()
