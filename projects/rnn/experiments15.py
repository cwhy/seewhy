"""
Pareto tuning round 10: GLU variants for the MLP head.

Current head: x = LN(state); h = GELU(W1 @ x + b1); logits = W2 @ h + b2

GLU variants split W1 into two branches (val + gate):
  h = (W_val @ x) * act(W_gate @ x)   — "split-projection" GLU

Use g = mlp//2 so non-emb params match GELU baseline:
  GELU  mlp=M:  W1 ∈ R^{M×state} + W2 ∈ R^{10×M}  →  state*M + 11*M + 10 + 2*state
  GEGLU g=M/2: W1 ∈ R^{2g×state} + W2 ∈ R^{10×g} →  state*2g + 11*g + 10 + 2*state
                                                    = state*M + 5.5*M + ... ≈ same

Variants:
  GEGLU   — gate activation = gelu   (PaLM, T5 variant)
  SwiGLU  — gate activation = silu   (LLaMA, Mistral)
  Bilinear — gate activation = identity (pure multiplicative gating, no nonlinearity)

Baselines to beat:
  534K:   f3(4,4,4) GELU mlp=128   → 98.46%
  1.06M:  f3(4,4,4) GELU mlp=256   → 98.60%
  2.11M:  f3(4,4,4) GELU mlp=512   → 98.81%
  68K:    joint(d=32) GELU mlp=64  → 97.29%
  267K:   joint(d=32) GELU mlp=256 → 97.91%

Questions:
  1. Does GEGLU/SwiGLU beat GELU at the same non-emb budget?
  2. Is the gating itself useful (bilinear > GELU)?
  3. Does the answer depend on scale (small 68K vs large 2.11M)?
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


# ── utils ──────────────────────────────────────────────────────────────────────

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


# ── augmentation ───────────────────────────────────────────────────────────────

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


# ── GLU head helper ────────────────────────────────────────────────────────────
# W1 ∈ R^{2g × state_dim}, b1 ∈ R^{2g}  (split evenly into val, gate)
# h  = W1_val @ x * act(W1_gate @ x)     ∈ R^g
# out = W2 @ h + b2                       ∈ R^{10}

def _glu_forward(params, x_norm, act_fn):
    """GLU head forward pass (expects x_norm already layer-normed)."""
    h  = x_norm @ params["W1"].T + params["b1"]   # (B, 2g)
    g  = h.shape[-1] // 2
    val, gate = h[:, :g], h[:, g:]
    h  = val * act_fn(gate)                         # (B, g)
    return h @ params["W2"].T + params["b2"]

def _glu_init_params(state_dim, g, kg):
    """Return W1 (2g×state), b1 (2g), W2 (10×g), b2 (10) for a GLU head."""
    return {
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (2 * g, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(2 * g),
        "W2": jax.random.normal(next(kg).get(), (10, g)) * 0.01,
        "b2": jnp.zeros(10),
    }


# ── model builders ─────────────────────────────────────────────────────────────
# All return (fwd, params, emb_keys).
# head_act: "gelu" | "swiglu" | "bilinear"
# g = mlp_dim // 2  (so param count ≈ GELU with mlp_dim)

_ACT = {
    "gelu":     jax.nn.gelu,
    "swiglu":   jax.nn.silu,
    "bilinear": lambda x: x,   # pure multiplicative gating
}


def make_factored3_glu(d_r, d_c, d_v, mlp_dim, head_act="geglu",
                       n_val_bins=N_VAL_BINS_DEFAULT):
    """Factored3 embedding + GLU head.

    emb_t = row[r] ⊗ col[c] ⊗ val[bin]
    M     = Σ_t emb_t ⊗ emb_t  ∈ R^{state_dim}
    head  = GLU (split W1 into val+gate branches)

    g = mlp_dim // 2  (W1 has shape 2g×state, so non-emb ≈ GELU mlp=mlp_dim)
    """
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim * emb_dim
    g         = mlp_dim // 2
    act_fn    = _ACT[head_act]

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        **_glu_init_params(state_dim, g, kg),
    }
    emb_keys = frozenset({"row_emb", "col_emb", "val_emb"})

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS]
        ce   = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, emb_dim)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        return _glu_forward(params, M, act_fn)

    return fwd, p, emb_keys


def make_joint_glu(d, mlp_dim, head_act="geglu", n_val_bins=N_VAL_BINS_DEFAULT):
    """Joint (pos×val) embedding + GLU head.

    emb_t = E[t, bin_t]    E ∈ R^{784 × K × d}
    M     = Σ_t emb_t ⊗ emb_t  ∈ R^{d^2}
    head  = GLU (g = mlp_dim // 2)
    """
    state_dim = d * d
    g         = mlp_dim // 2
    act_fn    = _ACT[head_act]

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "joint_emb": jax.random.normal(
            next(kg).get(), (784, n_val_bins, d)) * 0.02,
        **_glu_init_params(state_dim, g, kg),
    }
    emb_keys = frozenset({"joint_emb"})

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["joint_emb"].shape[1]
        bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
        emb  = params["joint_emb"][T_IDX, bins]              # (B, 784, d)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        return _glu_forward(params, M, act_fn)

    return fwd, p, emb_keys


# ── training ───────────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          max_epochs=30, target_acc=0.995, aug=AUG_NOISE30):
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
        if test_acc >= target_acc:
            break

    return train_acc, test_acc, time.perf_counter() - t0, epoch + 1


# ── param budget reference ──────────────────────────────────────────────────────
# factored3(4,4,4): emb_dim=64, state_dim=4096
#   GELU mlp=M:   state*M + 11M + 10 + 2*state = 4096M + 11M + 8202 = 4107M + 8202
#   GLU  g=M//2:  state*M + 11*(M//2) + 10 + 2*state ≈ 4096M + 5.5M + 8202
#   (slightly fewer due to smaller W2; close enough to compare)
#
# ~534K non-emb:  GELU mlp=128  → GLU g=64
# ~1.06M non-emb: GELU mlp=256  → GLU g=128
# ~2.11M non-emb: GELU mlp=512  → GLU g=256
#
# joint(d=32): state_dim=1024
#   ~68K:  GELU mlp=64  → GLU g=32
#   ~267K: GELU mlp=256 → GLU g=128

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    prev_best = {
        68_298:    ("joint(d=32) GELU mlp=64",  0.9729),
        267_018:   ("joint(d=32) GELU mlp=256", 0.9791),
        533_898:   ("f3(4,4,4) GELU mlp=128",   0.9846),
        1_059_594: ("f3(4,4,4) GELU mlp=256",   0.9860),
        2_110_986: ("f3(4,4,4) GELU mlp=512",   0.9881),
    }

    experiments = [
        # ── factored3(4,4,4) + noise30, GLU heads at each budget ──────────────
        # g = mlp//2: two 64-wide projections ≈ same non-emb as GELU mlp=128
        ("f3(4,4,4) GEGLU g=64 +noise30 Muon",
         *make_factored3_glu(4, 4, 4, 128, "gelu"),
         AUG_NOISE30, 30,
         "GEGLU g=64, same budget as GELU mlp=128 534K; does gating+gelu > gelu?"),

        ("f3(4,4,4) SwiGLU g=64 +noise30 Muon",
         *make_factored3_glu(4, 4, 4, 128, "swiglu"),
         AUG_NOISE30, 30,
         "SwiGLU g=64, same budget as GELU mlp=128 534K"),

        ("f3(4,4,4) Bilinear g=64 +noise30 Muon",
         *make_factored3_glu(4, 4, 4, 128, "bilinear"),
         AUG_NOISE30, 30,
         "Bilinear g=64 (pure gating, no nonlinearity), same budget 534K"),

        ("f3(4,4,4) GEGLU g=128 +noise30 Muon",
         *make_factored3_glu(4, 4, 4, 256, "gelu"),
         AUG_NOISE30, 30,
         "GEGLU g=128 ≈ 1.06M; vs GELU mlp=256 98.60%"),

        ("f3(4,4,4) SwiGLU g=128 +noise30 Muon",
         *make_factored3_glu(4, 4, 4, 256, "swiglu"),
         AUG_NOISE30, 30,
         "SwiGLU g=128 ≈ 1.06M; vs GELU mlp=256 98.60%"),

        ("f3(4,4,4) GEGLU g=256 +noise30 Muon",
         *make_factored3_glu(4, 4, 4, 512, "gelu"),
         AUG_NOISE30, 30,
         "GEGLU g=256 ≈ 2.11M; vs GELU mlp=512 98.81%"),

        ("f3(4,4,4) SwiGLU g=256 +noise30 Muon",
         *make_factored3_glu(4, 4, 4, 512, "swiglu"),
         AUG_NOISE30, 30,
         "SwiGLU g=256 ≈ 2.11M; vs GELU mlp=512 98.81%"),

        # ── joint(d=32) + noise30, GLU head ───────────────────────────────────
        ("joint(d=32) GEGLU g=32 +noise30 Muon",
         *make_joint_glu(32, 64, "gelu"),
         AUG_NOISE30, 30,
         "joint GEGLU g=32 ≈ 68K; vs GELU mlp=64 97.29%"),

        ("joint(d=32) GEGLU g=128 +noise30 Muon",
         *make_joint_glu(32, 256, "gelu"),
         AUG_NOISE30, 30,
         "joint GEGLU g=128 ≈ 267K; vs GELU mlp=256 97.91%"),
    ]

    results = []
    for label, fwd, params, emb_keys, aug, max_ep, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        total  = n_total(params)
        emb    = total - nonemb
        closest = min(prev_best.keys(), key=lambda x: abs(x - nonemb))
        base_name, base_acc = prev_best[closest]
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  emb={emb:,}  total={total:,}  aug={aug}\n"
            f"  baseline: {base_name} = {base_acc:.4f}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug,
        )
        delta = te_acc - base_acc
        row = {
            "name": label, "experiment": "exp15",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, emb, tr_acc, te_acc, epochs, elapsed,
                        base_acc, base_name))
        logging.info(f"  → {te_acc:.4f}  vs {base_name} {base_acc:.4f}: {delta:+.4f}")

    print(f"\n{'='*120}")
    print(f"{'Variant':<52} {'NonEmb':>8} {'Emb':>10} {'Train':>8} {'Test':>8} {'Delta':>7} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*120}")
    for label, nonemb, emb, tr, te, ep, t, base, bname in results:
        flag  = " ★" if te > base else " ✗"
        delta = f"{te-base:+.4f}"
        print(f"{label:<52} {nonemb:>8,} {emb:>10,} {tr:>7.4f} {te:>7.4f}{flag} {delta:>7} {ep:>4} {t:>7.1f}s")
    print(f"{'='*120}")
    print(f"\nBaselines: ", "  |  ".join(f"{v[0]}={v[1]:.4f}" for v in prev_best.values()))

    logging.info("Updating Pareto plot...")
    replot()
