"""
Pareto tuning round 5: push beyond 98.60% with GELU + Muon.

Findings from exp9:
  ✓ GELU + Muon (true) is the best combo: 98.60% at 1.06M in 11 epochs
  ✓ GELU + Muon at 534K: 98.36% (beats Adam+GELU 98.31%)
  ✓ GELU mlp=512 Adam: 98.56% at 2.11M
  ✗ sgate: consistently hurts accuracy
  ✗ bins=64: no benefit over bins=32

Questions:
  1. Can GELU + Muon at 2.11M push past 98.60%?
  2. Does stronger noise aug (±30) help generalize further?
  3. Does a deeper MLP (3-layer) help with GELU?
  4. Can larger factored dims (4,4,4) → (4,8,4) etc. help?
  5. Can GELU + Muon dominate at the 135K Pareto point?
  6. Does a 2-head approach (two independent Gram matrices concatenated) help?
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

AUG_NONE       = "none"
AUG_NOISE      = "noise"
AUG_NOISE_HARD = "noise30"   # stronger noise ±30px


# ── Muon (NS + Nesterov SGD for 2-D weights, Adam for rest) ──────────────────

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
        mu = state["mu"]
        new_mu    = jax.tree_util.tree_map(lambda m, g: momentum*m + g, mu, updates)
        nesterov  = jax.tree_util.tree_map(lambda m, g: momentum*m + g, new_mu, updates)
        out = jax.tree_util.tree_map(
            lambda g: _newton_schulz(g, ns_steps) if g.ndim == 2 else g, nesterov)
        return out, {"mu": new_mu}
    return optax.GradientTransformation(init, update)

def make_muon_tx(params, adam_schedule, muon_schedule):
    mat_keys = frozenset(k for k, v in params.items()
                         if v.ndim == 2 and k not in EMB_KEYS)
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


# ── model builders ────────────────────────────────────────────────────────────

T_IDX = jnp.arange(784)
ROWS  = T_IDX // 28
COLS  = T_IDX % 28


def make_factored3_gelu(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """factored3 with LayerNorm + GELU (best found so far)."""
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb":  jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb":  jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),  "ln_beta": jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
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
    return fwd, p


def make_factored3_gelu_deep(d_r, d_c, d_v, h1, h2, n_val_bins=N_VAL_BINS_DEFAULT):
    """3-layer MLP: state → h1 → h2 → 10, all GELU."""
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb":  jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb":  jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),  "ln_beta": jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (h1, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(h1),
        "W2": jax.random.normal(next(kg).get(), (h2, h1)) / h1**0.5,
        "b2": jnp.zeros(h2),
        "W3": jax.random.normal(next(kg).get(), (10, h2)) * 0.01,
        "b3": jnp.zeros(10),
    }
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
        h_1  = jax.nn.gelu(M           @ params["W1"].T + params["b1"])
        h_2  = jax.nn.gelu(h_1         @ params["W2"].T + params["b2"])
        return h_2 @ params["W3"].T + params["b3"]
    return fwd, p


def make_factored3_gelu_2head(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """Two independent factored3 Gram matrices concatenated → MLP.

    Each head has its own row/col/val embeddings.  The two heads provide
    complementary views of the pixel statistics at the same non-emb cost
    (~same W1 size, half the state per head but same total MLP input).
    Non-emb: 2*(2*state_half + ...) but state_half = (d_r*d_c*d_v/√2)^2...

    Concretely: head dim = d_r*d_c*d_v,  each Gram is (emb_dim)^2.
    We concatenate two Gram matrices → input to W1 is 2*state_dim.
    """
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        # head 1
        "row_emb_a": jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb_a": jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb_a": jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        # head 2
        "row_emb_b": jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb_b": jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb_b": jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(2*state_dim),
        "ln_beta":  jnp.zeros(2*state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, 2*state_dim)) / (2*state_dim)**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }

    def gram_head(params, bins, row_key, col_key, val_key):
        B = bins.shape[0]
        re  = params[row_key][ROWS]
        ce  = params[col_key][COLS]
        ve  = params[val_key][bins]
        pos = (re[:,:,None] * ce[:,None,:]).reshape(784, d_r*d_c)
        emb = (pos[None,:,:,None] * ve[:,:,None,:]).reshape(B, 784, emb_dim)
        return jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb_a"].shape[0]
        bins = jnp.floor(x / 255.0 * (K-1)).astype(jnp.int32)
        Ma   = gram_head(params, bins, "row_emb_a", "col_emb_a", "val_emb_a")
        Mb   = gram_head(params, bins, "row_emb_b", "col_emb_b", "val_emb_b")
        M    = jnp.concatenate([Ma, Mb], axis=-1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h    = jax.nn.gelu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


# ── training ──────────────────────────────────────────────────────────────────

def train(name, fwd, params, X_tr, y_tr, X_te, y_te,
          max_epochs=30, target_acc=0.988, aug=AUG_NOISE, optimizer="muon"):
    num_batches  = len(X_tr) // BATCH_SIZE
    total_steps  = max_epochs * num_batches
    adam_sched   = optax.cosine_decay_schedule(LR,      total_steps)
    muon_sched   = optax.cosine_decay_schedule(LR_MUON, total_steps)

    if optimizer == "muon":
        tx = make_muon_tx(params, adam_sched, muon_sched)
    else:
        tx = make_adam_tx(adam_sched)

    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    noise_scale = 30.0 if aug == AUG_NOISE_HARD else 15.0

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
            if aug in (AUG_NOISE, AUG_NOISE_HARD):
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

    experiments = [
        # ── 1. GELU + Muon at 2.11M — Muon gave +0.20% at 1.06M, apply at 2.11M ─
        ("factored3(4,4,4) mlp=512 GELU +noise Muon",
         *make_factored3_gelu(4, 4, 4, 512), AUG_NOISE, 30, "muon",
         "GELU + Muon at 2.11M, prev best 98.56%"),

        # ── 2. Stronger noise augmentation ─────────────────────────────────────
        # ±30px = 12% of 255; more robust val_emb learning
        ("factored3(4,4,4) mlp=256 GELU +noise30 Muon",
         *make_factored3_gelu(4, 4, 4, 256), AUG_NOISE_HARD, 30, "muon",
         "stronger noise aug ±30px, GELU+Muon, 1.06M"),
        ("factored3(4,4,4) mlp=512 GELU +noise30 Muon",
         *make_factored3_gelu(4, 4, 4, 512), AUG_NOISE_HARD, 30, "muon",
         "stronger noise aug ±30px, GELU+Muon, 2.11M"),

        # ── 3. Deep MLP (3-layer GELU) ─────────────────────────────────────────
        # Can additional nonlinearity in the readout help?
        ("factored3(4,4,4) mlp=256+128 GELU +noise Muon",
         *make_factored3_gelu_deep(4, 4, 4, 256, 128), AUG_NOISE, 30, "muon",
         "3-layer GELU MLP: state→256→128→10, Muon"),
        ("factored3(4,4,4) mlp=512+128 GELU +noise Muon",
         *make_factored3_gelu_deep(4, 4, 4, 512, 128), AUG_NOISE, 30, "muon",
         "3-layer GELU MLP: state→512→128→10, Muon"),

        # ── 4. Two-head Gram (complementary feature views) ─────────────────────
        # Two independent row⊗col⊗val Gram matrices concatenated before MLP.
        # Non-emb ≈ 4*(state_dim) + mlp*2*state_dim ≈ 2× larger W1.
        # At mlp=64 the non-emb is 2*4096*(2+64)+64+10*64 = ~546K
        ("factored3(4,4,4) 2head mlp=64 GELU +noise Muon",
         *make_factored3_gelu_2head(4, 4, 4, 64), AUG_NOISE, 30, "muon",
         "2 independent Gram heads concatenated, mlp=64"),
        ("factored3(4,4,4) 2head mlp=128 GELU +noise Muon",
         *make_factored3_gelu_2head(4, 4, 4, 128), AUG_NOISE, 30, "muon",
         "2 independent Gram heads concatenated, mlp=128"),

        # ── 5. GELU + Muon at smaller Pareto points ────────────────────────────
        ("factored3(2,4,4) mlp=128 GELU +noise Muon",
         *make_factored3_gelu(2, 4, 4, 128), AUG_NOISE, 30, "muon",
         "GELU+Muon at 135K — prev Adam 96.80%"),
        ("factored3(2,4,4) mlp=256 GELU +noise Muon",
         *make_factored3_gelu(2, 4, 4, 256), AUG_NOISE, 30, "muon",
         "GELU+Muon at 267K — new Pareto point"),
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
            "name": label, "experiment": "exp10",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, epochs, elapsed))
        logging.info(f"  → saved")

    print(f"\n{'='*96}")
    print(f"{'Variant':<50} {'NonEmb':>10} {'Train':>8} {'Test':>8} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*96}")
    prev_best = {
        134_538: 0.9680, 267_018: 0.9510,
        533_898: 0.9836, 1_059_594: 0.9860, 2_110_986: 0.9856,
    }
    for label, nonemb, tr, te, ep, t in results:
        flag = " ★" if te > prev_best.get(nonemb, 0.985) else ("✓" if te >= 0.986 else " ")
        print(f"{label:<50} {nonemb:>10,} {tr:>7.4f} {te:>7.4f}{flag:2} {ep:>4} {t:>7.1f}s")
    print(f"{'='*96}")

    logging.info("Updating Pareto plot...")
    replot()
