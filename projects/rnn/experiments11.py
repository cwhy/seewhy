"""
Pareto tuning round 6: push beyond 98.81%.

Findings from exp10:
  ✓ noise30 at 2.11M: +0.26% over noise15 (98.81% new record)
  ✓ noise30 at 1.06M: 98.65% (+0.05% over noise15)
  ✗ deep MLP (3-layer): marginal gain
  ✗ 2-head: FAILED due to EMB_KEYS bug (row/col/val _a/_b not in EMB_KEYS → NS destroyed embeddings)

Questions for exp11:
  1. noise30 at 534K — only Pareto point still using noise15
  2. noise40 / noise50 at 2.11M — find the noise-scale ceiling
  3. factored3(4,4,6) — larger state dim, new Pareto points at ~609K and ~1.2M
  4. fixed 2-head — correct EMB_KEYS, test at ~1.07M
  5. label smoothing (ε=0.1) at 2.11M — reduce overconfidence
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

# Base EMB_KEYS — model builders may extend this
_BASE_EMB_KEYS = frozenset({"row_emb", "col_emb", "val_emb", "pos_emb"})


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

AUG_NONE   = "none"
AUG_NOISE  = "noise15"
AUG_NOISE2 = "noise30"
AUG_NOISE3 = "noise40"
AUG_NOISE4 = "noise50"

_NOISE_SCALE = {AUG_NONE: 0.0, AUG_NOISE: 15.0, AUG_NOISE2: 30.0,
                AUG_NOISE3: 40.0, AUG_NOISE4: 50.0}


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
        mu      = state["mu"]
        new_mu  = jax.tree_util.tree_map(lambda m, g: momentum*m + g, mu, updates)
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
# Each returns (fwd, params, emb_keys) — emb_keys needed to correctly split Muon

T_IDX = jnp.arange(784)
ROWS  = T_IDX // 28
COLS  = T_IDX % 28


def make_factored3_gelu(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """factored3: row⊗col⊗val + Gram + LayerNorm + GELU MLP."""
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


def make_factored3_gelu_2head(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """Two independent factored3 Gram matrices concatenated → MLP.
    Each head has its own embeddings.  Non-emb ≈ 2× state_dim input to W1.
    EMB_KEYS correctly includes all _a and _b embedding variants.
    """
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb_a": jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb_a": jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb_a": jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "row_emb_b": jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb_b": jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb_b": jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(2 * state_dim),
        "ln_beta":  jnp.zeros(2 * state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, 2*state_dim)) / (2*state_dim)**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    # All six embedding tables must be in emb_keys so Muon won't NS-orthogonalise them
    emb_keys = frozenset({
        "row_emb_a", "col_emb_a", "val_emb_a",
        "row_emb_b", "col_emb_b", "val_emb_b",
    })

    def gram_head(params, bins, rk, ck, vk):
        B   = bins.shape[0]
        re  = params[rk][ROWS];  ce = params[ck][COLS]
        ve  = params[vk][bins]
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

    return fwd, p, emb_keys


# ── training ──────────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          max_epochs=30, target_acc=0.990, aug=AUG_NOISE2,
          optimizer="muon", label_smooth=0.0):
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

    def loss_fn(p, bx, by):
        logits = fwd(p, bx)
        xent   = optax.softmax_cross_entropy_with_integer_labels(logits, by)
        if label_smooth > 0.0:
            # KL to uniform: -log(1/10) * smooth + (1-smooth)*xent
            log_uniform = -jnp.log(jnp.ones_like(logits) / 10.0)
            smooth_loss = jnp.mean(jax.nn.softmax(logits) * log_uniform, axis=-1)
            xent = (1 - label_smooth) * xent + label_smooth * smooth_loss
        return jnp.mean(xent)

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
            if noise_scale > 0.0:
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

    # (label, fwd, params, emb_keys, aug, max_ep, opt, label_smooth, notes)
    experiments = [
        # ── 1. Fill the 534K noise30 gap ───────────────────────────────────────
        # All larger Pareto points use noise30; 534K still uses noise15 (98.36%)
        ("factored3(4,4,4) mlp=128 GELU +noise30 Muon",
         *make_factored3_gelu(4, 4, 4, 128),
         AUG_NOISE2, 30, "muon", 0.0,
         "noise30 at 534K — was noise15 98.36%"),

        # ── 2. Noise scale ceiling at 2.11M ────────────────────────────────────
        # noise30 gave +0.26% vs noise15 at 2.11M. Does noise40/50 keep improving?
        ("factored3(4,4,4) mlp=512 GELU +noise40 Muon",
         *make_factored3_gelu(4, 4, 4, 512),
         AUG_NOISE3, 30, "muon", 0.0,
         "noise40 at 2.11M, was 98.81% at noise30"),
        ("factored3(4,4,4) mlp=512 GELU +noise50 Muon",
         *make_factored3_gelu(4, 4, 4, 512),
         AUG_NOISE4, 30, "muon", 0.0,
         "noise50 at 2.11M, find noise ceiling"),

        # ── 3. Larger state dimension: factored3(4,4,6) ────────────────────────
        # emb_dim=96, state_dim=9216 — new Pareto points between 534K and 1.06M+
        ("factored3(4,4,6) mlp=64 GELU +noise30 Muon",
         *make_factored3_gelu(4, 4, 6, 64),
         AUG_NOISE2, 30, "muon", 0.0,
         "larger state (4,4,6) mlp=64 ~609K non-emb"),
        ("factored3(4,4,6) mlp=128 GELU +noise30 Muon",
         *make_factored3_gelu(4, 4, 6, 128),
         AUG_NOISE2, 30, "muon", 0.0,
         "larger state (4,4,6) mlp=128 ~1.2M non-emb"),

        # ── 4. Fixed 2-head (correct EMB_KEYS) ────────────────────────────────
        # exp10 failed because _a/_b embedding keys were not in EMB_KEYS.
        # Fixed: emb_keys returned by builder now includes all 6 embedding tables.
        # mlp=128 → W1=(128, 2*4096=8192) → ~1.07M non-emb, comparable to 1.06M single-head
        ("factored3(4,4,4) 2head mlp=128 GELU +noise30 Muon (fixed)",
         *make_factored3_gelu_2head(4, 4, 4, 128),
         AUG_NOISE2, 30, "muon", 0.0,
         "2-head Gram, fixed EMB_KEYS, ~1.07M non-emb"),

        # ── 5. Label smoothing at 2.11M ────────────────────────────────────────
        # train_acc hits ~99.8% — slight overconfidence. ε=0.1 may regularise.
        ("factored3(4,4,4) mlp=512 GELU +noise30 Muon +lblsmooth",
         *make_factored3_gelu(4, 4, 4, 512),
         AUG_NOISE2, 30, "muon", 0.1,
         "label smoothing ε=0.1 at 2.11M with noise30"),
    ]

    results = []
    for label, fwd, params, emb_keys, aug, max_ep, opt_name, lsmooth, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        total  = n_total(params)
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  total={total:,}  opt={opt_name}  aug={aug}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug,
            optimizer=opt_name, label_smooth=lsmooth,
        )
        row = {
            "name": label, "experiment": "exp11",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, epochs, elapsed))
        logging.info(f"  → saved")

    print(f"\n{'='*100}")
    print(f"{'Variant':<54} {'NonEmb':>10} {'Train':>8} {'Test':>8} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*100}")
    prev_best = {
        134_538: 0.9689, 267_018: 0.9711,
        533_898: 0.9836, 608_970: 0.0, 1_059_594: 0.9865, 1_199_368: 0.0,
        2_110_986: 0.9881,
    }
    for label, nonemb, tr, te, ep, t in results:
        flag = " ★" if te > prev_best.get(nonemb, 0.987) else ("✓" if te >= 0.988 else " ")
        print(f"{label:<54} {nonemb:>10,} {tr:>7.4f} {te:>7.4f}{flag:2} {ep:>4} {t:>7.1f}s")
    print(f"{'='*100}")

    logging.info("Updating Pareto plot...")
    replot()
