"""
Pareto tuning round 4: GELU is the key discovery from exp8.

Findings from exp8:
  ✓ GELU  beats ReLU significantly: +0.27% at 534K, +0.43% at 1.06M
  ✗ DyT   hurts (LayerNorm's mean centering of the Gram matrix is important)
  ✗ RMSNorm hurts (mean centering important for PSD Gram matrix)
  ✗ Muon (NS-Adam) underperforms — NS preconditioning conflicts with Adam's second moment
  ✗ Spatial gate (sgate) hurts — gating some pixels away loses information
  ✗ bins=64 + noise: no benefit over bins=32 + noise

Strategy for exp9:
  1. GELU + mlp=512 / mlp=1024  — does GELU unlock larger MLP capacity?
  2. GELU + more epochs (50ep) — exploit the better loss landscape
  3. GELU + Muon (true Muon: NS + SGD momentum, not NS + Adam)
     True Muon paper: separate optimizers for matrices vs scalars/embeddings.
  4. GELU + spatial gate — combo: smooth activations + position attention
  5. GELU + bins=64                — finer quantisation with smooth activation
  6. factored3(4,4,6) GELU + noise — bigger value dim with GELU
  7. factored3(2,4,4) GELU + noise — small Pareto point with GELU
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
SEED       = 42

JSONL  = Path(__file__).parent / "results_mnist.jsonl"
PLOT   = Path(__file__).parent / "plot_pareto_mnist.py"
EMB_KEYS = {"row_emb", "col_emb", "val_emb", "pos_emb"}


# ── utils ─────────────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

def n_nonemb(p): return sum(v.size for k,v in p.items() if k not in EMB_KEYS)
def n_total(p):  return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")

def replot():
    subprocess.run([sys.executable, str(PLOT)], check=False)


# ── augmentation ──────────────────────────────────────────────────────────────

def random_noise_batch(imgs, rng, scale=15.0):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)

AUG_NONE  = "none"
AUG_NOISE = "noise"


# ── True Muon (Newton-Schulz + SGD momentum, Adam for non-matrices) ───────────

def _newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = jnp.linalg.norm(G)
    X = G / (norm + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a*X + b*(A@X) + c*(A@(A@X))
    return X * norm


def ns_nesterov_direction(momentum=0.95, ns_steps=5):
    """Pure direction transform: Nesterov momentum + Newton-Schulz on 2D tensors.
    Does not apply a learning rate — combine with optax.scale_by_learning_rate().
    """
    def init(params):
        return {"mu": jax.tree_util.tree_map(jnp.zeros_like, params)}

    def update(updates, state, params=None):
        mu = state["mu"]
        new_mu = jax.tree_util.tree_map(
            lambda m, g: momentum * m + g, mu, updates)
        # Nesterov lookahead
        nesterov = jax.tree_util.tree_map(
            lambda m, g: momentum * m + g, new_mu, updates)
        # NS-orthogonalise 2D matrix gradients, pass through scalars/1D unchanged
        out = jax.tree_util.tree_map(
            lambda g: _newton_schulz(g, ns_steps) if g.ndim == 2 else g,
            nesterov)
        return out, {"mu": new_mu}

    return optax.GradientTransformation(init, update)


def make_muon_adam_tx(params, schedule):
    """True Muon: NS+Nesterov for 2-D weights (lr=0.02 cosine), Adam for rest.

    Key difference from exp8: no Adam second-moment for weight matrices.
    The NS step provides the direction; a separate lr schedule controls step size.
    """
    total_steps  = schedule.init_value if hasattr(schedule, 'init_value') else None
    # Muon uses a higher base LR than Adam since NS normalises the update scale
    muon_schedule = optax.cosine_decay_schedule(0.02, 30 * (60_000 // BATCH_SIZE))

    mat_keys = frozenset(k for k, v in params.items()
                         if v.ndim == 2 and k not in EMB_KEYS)
    labels = {k: "muon" if k in mat_keys else "adam" for k in params}

    muon_branch = optax.chain(
        ns_nesterov_direction(momentum=0.95),
        optax.scale_by_learning_rate(muon_schedule),
    )
    adam_branch = optax.adam(schedule)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform({"muon": muon_branch, "adam": adam_branch}, labels),
    )


def make_adam_tx(schedule):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))


# ── models (factored3 with GELU or ReLU) ─────────────────────────────────────

T_IDX = jnp.arange(784)
ROWS  = T_IDX // 28
COLS  = T_IDX % 28


def _make_factored3(d_r, d_c, d_v, mlp_dim, n_val_bins, activation, extra_params=None):
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2
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
    if extra_params:
        p.update(extra_params(state_dim, kg))

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K-1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS]
        ce   = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r*d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, emb_dim)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h    = activation(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


def make_factored3_gelu(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    return _make_factored3(d_r, d_c, d_v, mlp_dim, n_val_bins, jax.nn.gelu)

def make_factored3_relu(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    return _make_factored3(d_r, d_c, d_v, mlp_dim, n_val_bins, jax.nn.relu)


def make_factored3_gelu_sgate(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """GELU + spatial gate: σ(row_score + col_score) weights each pixel's contribution."""
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb":  jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb":  jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "gate_row": jnp.zeros(28),
        "gate_col": jnp.zeros(28),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K-1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS]
        ce   = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r*d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, emb_dim)
        gate = jax.nn.sigmoid(params["gate_row"][ROWS] + params["gate_col"][COLS])
        M    = jnp.einsum("t,bti,btj->bij", gate, emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h    = jax.nn.gelu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


# ── training ──────────────────────────────────────────────────────────────────

def train(name, fwd, params, X_tr, y_tr, X_te, y_te,
          max_epochs=30, target_acc=0.986, aug=AUG_NONE, optimizer="adam"):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    schedule    = optax.cosine_decay_schedule(LR, total_steps)

    if optimizer == "muon":
        tx = make_muon_adam_tx(params, schedule)
    else:
        tx = make_adam_tx(schedule)

    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

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
            if aug == AUG_NOISE:
                aug_rng, sub = jax.random.split(aug_rng)
                bx = random_noise_batch(bx, sub)
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

    experiments = [
        # ── 1. GELU + larger MLP (does GELU unlock capacity above mlp=256?) ───
        ("factored3(4,4,4) mlp=512 GELU +noise",
         *make_factored3_gelu(4, 4, 4, 512), AUG_NOISE, 30, "adam",
         "GELU, 2.11M non-emb, does capacity help beyond 1.06M?"),
        ("factored3(4,4,4) mlp=256 GELU +noise 50ep",
         *make_factored3_gelu(4, 4, 4, 256), AUG_NOISE, 50, "adam",
         "GELU, 1.06M, more epochs — previous model not saturated"),

        # ── 2. GELU + Muon (true Muon: NS + SGD momentum, not NS + Adam) ──────
        # Proper Muon: lr_muon=0.02 + Nesterov, Adam only for embeddings/scalars
        ("factored3(4,4,4) mlp=128 GELU +noise Muon",
         *make_factored3_gelu(4, 4, 4, 128), AUG_NOISE, 30, "muon",
         "GELU + true Muon (NS+SGD) for W1/W2"),
        ("factored3(4,4,4) mlp=256 GELU +noise Muon",
         *make_factored3_gelu(4, 4, 4, 256), AUG_NOISE, 30, "muon",
         "GELU + true Muon, 1.06M"),

        # ── 3. GELU + spatial gate ────────────────────────────────────────────
        # sgate alone hurt; combined with GELU the smoother gradient may help gates
        ("factored3(4,4,4) mlp=128 GELU +noise +sgate",
         *make_factored3_gelu_sgate(4, 4, 4, 128), AUG_NOISE, 30, "adam",
         "GELU + spatial gate, 534K+56 params"),
        ("factored3(4,4,4) mlp=256 GELU +noise +sgate",
         *make_factored3_gelu_sgate(4, 4, 4, 256), AUG_NOISE, 30, "adam",
         "GELU + spatial gate, 1.06M"),

        # ── 4. GELU + finer bins ──────────────────────────────────────────────
        ("factored3(4,4,4) mlp=128 GELU +noise bins=64",
         *make_factored3_gelu(4, 4, 4, 128, n_val_bins=64), AUG_NOISE, 30, "adam",
         "GELU + bins=64 + noise"),
        ("factored3(4,4,4) mlp=256 GELU +noise bins=64",
         *make_factored3_gelu(4, 4, 4, 256, n_val_bins=64), AUG_NOISE, 30, "adam",
         "GELU + bins=64 + noise, 1.06M"),

        # ── 5. GELU at other Pareto points ────────────────────────────────────
        ("factored3(2,4,4) mlp=128 GELU +noise",
         *make_factored3_gelu(2, 4, 4, 128), AUG_NOISE, 30, "adam",
         "GELU at 135K Pareto point"),
        ("factored3(4,4,6) mlp=64 GELU +noise",
         *make_factored3_gelu(4, 4, 6, 64), AUG_NOISE, 30, "adam",
         "GELU, state=9216, 609K non-emb"),
    ]

    results = []
    for label, fwd, params, aug, max_ep, opt_name, notes in experiments:
        nonemb = n_nonemb(params)
        total  = n_total(params)
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  total={total:,}  opt={opt_name}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug, optimizer=opt_name,
        )
        row = {
            "name": label, "experiment": "exp9",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, epochs, elapsed))
        logging.info(f"  → saved to JSONL")

    print(f"\n{'='*92}")
    print(f"{'Variant':<46} {'NonEmb':>10} {'Train':>8} {'Test':>8} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*92}")
    prev_best = {134_538: 0.9636, 533_898: 0.9831, 608_970: 0.9743, 1_059_594: 0.9844, 2_110_986: 0.9816}
    for label, nonemb, tr, te, ep, t in results:
        flag = " ★" if te > prev_best.get(nonemb, 0.98) else ("✓" if te >= 0.985 else " ")
        print(f"{label:<46} {nonemb:>10,} {tr:>7.4f} {te:>7.4f}{flag:2} {ep:>4} {t:>7.1f}s")
    print(f"{'='*92}")

    logging.info("Updating Pareto plot...")
    replot()
