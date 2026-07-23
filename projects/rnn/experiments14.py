"""
Pareto tuning round 9: decouple embedding dim from Gram state dim.

Current constraint: state_dim = emb_dim^2
  (because M = Σ emb_t ⊗ emb_t, so M ∈ R^{emb_dim × emb_dim})

New architecture: add a learned projection P ∈ R^{k × emb_dim} before the Gram:
  proj_t = P @ emb_t  ∈ R^k
  M = Σ_t proj_t ⊗ proj_t  ∈ R^{k^2}

Now k (Gram size) is independent of emb_dim. P is a 2D weight matrix → goes to Muon.

Non-emb formula:
  k * emb_dim      (projection P)
  + k^2 * (2 + mlp) + 11*mlp + 10   (LayerNorm + MLP)

Key use cases:
  A) joint(large d) + project DOWN to small k
     → rich embedding, small state; avoids the 1.6M emb table at d=64
  B) joint(small d) + project to k > d
     → tiny emb table, large state
  C) factored3(small dims) + project UP to large k
     → same state as factored3(big dims) but with cheaper embeddings

Questions:
  1. Can joint(d=8, k=64) + Muon@P match factored3(4,4,4) at 534K?
     (emb table: 200K instead of 1.6M for joint(d=64) — was -0.92%)
  2. Does joint(d=32, k=64) at 536K beat joint(d=64) 97.54%?
     (same embedding as best joint, but now state=4096 instead of 1024)
  3. Can factored3(2,4,4) projected to k=64 match factored3(4,4,4) at 534K?
     (much smaller embedding table, same state size)
  4. Is joint(d=8, k=32) at 35K as good as direct joint(d=32)?
     (emb table 200K vs 802K, same state size 1024)
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


# ── model builders ─────────────────────────────────────────────────────────────
# All return (fwd, params, emb_keys).
# 'proj' (if present) is a 2D weight matrix → goes to Muon automatically.

def make_joint_proj_gelu(d, k, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """Joint (pos×val) embedding, projected to k before Gram.

    emb_t = joint_emb[t, bin_t]  ∈ R^d
    proj_t = P @ emb_t             ∈ R^k     (P ∈ R^{k×d}, non-emb, Muon)
    M = Σ_t proj_t ⊗ proj_t        ∈ R^{k^2}

    emb params: 784 * n_bins * d
    non-emb: k*d (proj) + k^2*(2+mlp) + 11*mlp + 10
    """
    state_dim = k * k
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "joint_emb": jax.random.normal(
            next(kg).get(), (784, n_val_bins, d)) * 0.02,
        "proj":      jax.random.normal(
            next(kg).get(), (k, d)) / d**0.5,
        "ln_gamma":  jnp.ones(state_dim),
        "ln_beta":   jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    emb_keys = frozenset({"joint_emb"})

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["joint_emb"].shape[1]
        bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
        emb  = params["joint_emb"][T_IDX, bins]              # (B, 784, d)
        proj = emb @ params["proj"].T                         # (B, 784, k)
        M    = jnp.einsum("bti,btj->bij", proj, proj).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h    = jax.nn.gelu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, emb_keys


def make_factored3_proj_gelu(d_r, d_c, d_v, k, mlp_dim,
                              n_val_bins=N_VAL_BINS_DEFAULT):
    """Factored3 embedding, projected to k before Gram.

    emb_t = row[r] ⊗ col[c] ⊗ val[bin]  ∈ R^{d_r*d_c*d_v}
    proj_t = P @ emb_t                    ∈ R^k
    M = Σ_t proj_t ⊗ proj_t

    emb params: 28*d_r + 28*d_c + n_bins*d_v
    non-emb: k*emb_dim (proj) + k^2*(2+mlp) + 11*mlp + 10
    """
    emb_dim   = d_r * d_c * d_v
    state_dim = k * k
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "proj":    jax.random.normal(next(kg).get(), (k, emb_dim)) / emb_dim**0.5,
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
        re   = params["row_emb"][ROWS]; ce = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:,:,None] * ce[:,None,:]).reshape(784, d_r*d_c)
        emb  = (pos[None,:,:,None] * ve[:,:,None,:]).reshape(B, 784, emb_dim)
        proj = emb @ params["proj"].T                          # (B, 784, k)
        M    = jnp.einsum("bti,btj->bij", proj, proj).reshape(B, -1)
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
        tx = make_muon_tx(params, emb_keys, adam_sched, muon_sched)  # always Muon

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


# ── param budget reference ─────────────────────────────────────────────────────
# joint_proj(d, k, mlp):  non-emb = k*d + k^2*(2+mlp) + 11*mlp + 10
# f3_proj(d_r,d_c,d_v, k, mlp): non-emb = k*(d_r*d_c*d_v) + k^2*(2+mlp) + 11*mlp + 10
#
# ~35K:
#   joint_proj(d=8,  k=32, mlp=32):  32*8  + 1024*34 + 362 =  35,690
#   joint_proj(d=4,  k=32, mlp=32):  32*4  + 1024*34 + 362 =  35,306
#
# ~534K:
#   joint_proj(d=8,  k=64, mlp=128): 64*8  + 4096*130 + 1418 = 534,426
#   joint_proj(d=16, k=64, mlp=128): 64*16 + 4096*130 + 1418 = 535,450
#   joint_proj(d=32, k=64, mlp=128): 64*32 + 4096*130 + 1418 = 537,498
#   f3_proj(2,4,4,   k=64, mlp=128): 64*32 + 4096*130 + 1418 = 537,498
#   f3_proj(4,4,4,   k=64, mlp=128): 64*64 + 4096*130 + 1418 = 541,594 (k=emb, redundant)
#
# ~267K:
#   joint_proj(d=8,  k=32, mlp=256): 32*8  + 1024*258 + 2826 = 267,034
#   joint_proj(d=16, k=32, mlp=256): 32*16 + 1024*258 + 2826 = 267,546

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # best known results at each non-emb budget
    prev_best = {
        35_178:    ("joint(d=32) mlp=32",   0.9660),
        68_298:    ("joint(d=32) mlp=64",   0.9729),
        134_538:   ("joint(d=32) mlp=128",  0.9769),
        267_018:   ("joint(d=32) mlp=256",  0.9791),
        533_898:   ("f3(4,4,4) mlp=128",    0.9846),
        1_059_594: ("f3(4,4,4) mlp=256",    0.9865),
        2_110_986: ("f3(4,4,4) mlp=512",    0.9881),
    }

    experiments = [
        # ── Q1: joint small d + proj UP — same state as factored3(4,4,4) ──────
        # Hypothesis: tiny emb table (200K) + proj avoids joint(d=64) overfitting
        ("joint_proj(d=8,k=64) mlp=128 GELU +noise30 Muon",
         *make_joint_proj_gelu(8, 64, 128),
         AUG_NOISE30, 30,
         "joint d=8→k=64, state=4096, emb=200K, ~534K non-emb; vs f3 98.46%"),

        ("joint_proj(d=16,k=64) mlp=128 GELU +noise30 Muon",
         *make_joint_proj_gelu(16, 64, 128),
         AUG_NOISE30, 30,
         "joint d=16→k=64, state=4096, emb=401K, ~535K non-emb; vs f3 98.46%"),

        # ── Q2: joint d=32 + proj to k=64 — bigger state from same emb ────────
        # Same emb table as best joint(d=32) but now state=4096 instead of 1024
        ("joint_proj(d=32,k=64) mlp=128 GELU +noise30 Muon",
         *make_joint_proj_gelu(32, 64, 128),
         AUG_NOISE30, 30,
         "joint d=32→k=64, state=4096, emb=802K, ~537K non-emb; can bigger state help?"),

        # ── Q3: factored3(2,4,4) + proj UP to k=64 — cheap emb, large state ──
        # factored3(2,4,4) emb table: 56+112+128=296 params (tiny!)
        # Projected to k=64 → same state=4096 as factored3(4,4,4) mlp=128
        ("f3_proj(2,4,4,k=64) mlp=128 GELU +noise30 Muon",
         *make_factored3_proj_gelu(2, 4, 4, 64, 128),
         AUG_NOISE30, 30,
         "f3(2,4,4) emb_dim=32→k=64, tiny emb, state=4096, ~537K; vs f3(4,4,4) 98.46%"),

        # ── Q4: same at 1.06M budget ───────────────────────────────────────────
        ("joint_proj(d=8,k=64) mlp=256 GELU +noise30 Muon",
         *make_joint_proj_gelu(8, 64, 256),
         AUG_NOISE30, 30,
         "joint d=8→k=64, state=4096, mlp=256, ~1.06M; vs f3 98.65%"),

        ("f3_proj(2,4,4,k=64) mlp=256 GELU +noise30 Muon",
         *make_factored3_proj_gelu(2, 4, 4, 64, 256),
         AUG_NOISE30, 30,
         "f3(2,4,4)→k=64, state=4096, mlp=256, ~1.06M; vs f3(4,4,4) 98.65%"),

        # ── Q5: small budget — can joint_proj(d=8,k=32) match joint(d=32)? ────
        # joint(d=32) mlp=32: 96.60% with 802K emb
        # joint_proj(d=8,k=32) mlp=32: same state=1024, but emb=200K
        ("joint_proj(d=8,k=32) mlp=32 GELU +noise30 Muon",
         *make_joint_proj_gelu(8, 32, 32),
         AUG_NOISE30, 30,
         "joint d=8→k=32, state=1024, emb=200K, ~35K; vs joint(d=32) 96.60%"),

        ("joint_proj(d=8,k=32) mlp=256 GELU +noise30 Muon",
         *make_joint_proj_gelu(8, 32, 256),
         AUG_NOISE30, 30,
         "joint d=8→k=32, state=1024, mlp=256, ~267K; vs joint(d=32) 97.91%"),
    ]

    results = []
    for label, fwd, params, emb_keys, aug, max_ep, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        total  = n_total(params)
        emb    = total - nonemb
        # find closest prev_best
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
            "name": label, "experiment": "exp14",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, emb, tr_acc, te_acc, epochs, elapsed,
                        base_acc, base_name))
        logging.info(f"  → {te_acc:.4f}  vs {base_name} {base_acc:.4f}: {delta:+.4f}")

    print(f"\n{'='*116}")
    print(f"{'Variant':<48} {'NonEmb':>8} {'Emb':>10} {'Train':>8} {'Test':>8} {'Delta':>7} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*116}")
    for label, nonemb, emb, tr, te, ep, t, base, bname in results:
        flag  = " ★" if te > base else " ✗"
        delta = f"{te-base:+.4f}"
        print(f"{label:<48} {nonemb:>8,} {emb:>10,} {tr:>7.4f} {te:>7.4f}{flag} {delta:>7} {ep:>4} {t:>7.1f}s")
    print(f"{'='*116}")
    print(f"\nBaselines: ", "  |  ".join(f"{v[0]}={v[1]:.4f}" for v in prev_best.values()))

    logging.info("Updating Pareto plot...")
    replot()
