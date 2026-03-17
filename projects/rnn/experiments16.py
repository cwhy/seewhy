"""
Pareto tuning round 11: matrix-structured readout of M.

Current head:  flatten(M) ∈ R^{d²} → MLP → 10 logits   (533K non-emb for d=64)
               treats M as a long vector, ignoring its matrix structure.

New ideas: exploit M as a d×d matrix.

─── Architecture A: Bilinear readout ───────────────────────────────────────────
  h_answer ∈ R^d  (learned "query" vector)
  H_y      ∈ R^{10 × d}  (one embedding per class)

  h_yhat  = M @ h_answer       ∈ R^d      (matrix-vector product)
  h_yhat  = LN(h_yhat)         (normalize; M scale grows with #pixels)
  logit_c = h_yc · h_yhat      ∈ R       (dot-product classifier)

  Interpretation: h_yhat is a "glimpse" of the Gram matrix along h_answer.
  logit_c = h_yc^T M h_answer = Σ_t (h_yc·emb_t)(h_answer·emb_t)
             = sum over pixels of (class-alignment × answer-alignment)

  Variant: replace dot-product head with a small MLP on the d-dim h_yhat.
  This is a huge compression: d=64 input to MLP vs d²=4096 input currently.

─── Architecture B: Projected Gram readout ─────────────────────────────────────
  H_proj ∈ R^{k × d}  (project embeddings to k-dim subspace before Gram)

  h_t = H_proj @ emb_t              ∈ R^k
  G   = Σ_t h_t ⊗ h_t              ∈ R^{k × k}   (k×k Gram in subspace)
  logits = W_out @ LN(flatten(G))   ∈ R^{10}

  This is a learned compression of the full Gram: captures all pairwise
  correlations within the k-dimensional projection. H_proj is 2D → Muon.

  Non-emb: k*d (H_proj) + 2*k² (LN) + 10*k² (W_out)
  For k=8, d=64:  512  + 128 + 640  = 1,280
  For k=16, d=64: 1024 + 512 + 2560 = 4,096
  For k=32, d=64: 2048 + 2048+10240 = 14,336
  For k=64, d=64: 4096 + 8192+40960 = 53,248

─── Baselines ────────────────────────────────────────────────────────────────
  534K:  f3(4,4,4) GELU mlp=128   → 98.46%
  68K:   joint(d=32) GELU mlp=64  → 97.29%
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


# ── embedding helpers ──────────────────────────────────────────────────────────

def _factored3_emb(params, x, d_r, d_c, d_v):
    """Return (B, 784, emb_dim) factored3 embeddings."""
    B    = x.shape[0]
    K    = params["val_emb"].shape[0]
    bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
    re   = params["row_emb"][ROWS]                          # (784, d_r)
    ce   = params["col_emb"][COLS]                          # (784, d_c)
    ve   = params["val_emb"][bins]                          # (B, 784, d_v)
    pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
    emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, d_r * d_c * d_v)
    return emb

def _joint_emb(params, x):
    """Return (B, 784, d) joint embeddings."""
    B    = x.shape[0]
    K    = params["joint_emb"].shape[1]
    bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
    return params["joint_emb"][T_IDX, bins]                 # (B, 784, d)

def _f3_init(d_r, d_c, d_v, kg, n_val_bins):
    return {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))         * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))         * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
    }

def _joint_init(d, kg, n_val_bins):
    return {
        "joint_emb": jax.random.normal(next(kg).get(), (784, n_val_bins, d)) * 0.02,
    }


# ── Architecture A: Bilinear readout ──────────────────────────────────────────

def make_bilinear(emb_type, emb_dims, mlp_dim=0, n_val_bins=N_VAL_BINS_DEFAULT):
    """
    Bilinear readout: h_yhat = M @ h_answer, then LN, then dot-product or MLP.

    emb_type: "factored3" or "joint"
    emb_dims: (d_r, d_c, d_v) for factored3, or (d,) for joint
    mlp_dim:  0 → dot-product head; >0 → small MLP on d-dim h_yhat

    non-emb (factored3 d=64):
      dot head:   d + 10*d + 2*d = 13*d = 832
      mlp head:   d + 2*d + d*mlp + mlp + mlp*10 + 10 = d*(3+mlp) + 11*mlp + 10
      mlp=32:     64*35 + 352 + 10 = 2592
      mlp=128:    64*131 + 1418 = 9802
      mlp=512:    64*515 + 5642 = 34602
    """
    if emb_type == "factored3":
        d_r, d_c, d_v = emb_dims
        emb_dim = d_r * d_c * d_v
        emb_keys = frozenset({"row_emb", "col_emb", "val_emb"})
    else:
        d = emb_dims[0]
        emb_dim = d
        emb_keys = frozenset({"joint_emb"})

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = (_f3_init(*emb_dims, kg, n_val_bins) if emb_type == "factored3"
          else _joint_init(emb_dims[0], kg, n_val_bins))

    # readout params (all non-emb)
    p["h_answer"] = jax.random.normal(next(kg).get(), (emb_dim,)) * 0.02      # 1D → Adam
    p["ln_gamma"] = jnp.ones(emb_dim)
    p["ln_beta"]  = jnp.zeros(emb_dim)

    if mlp_dim == 0:
        # dot-product classifier: H_y ∈ R^{10×d} (2D → Muon)
        p["H_y"] = jax.random.normal(next(kg).get(), (10, emb_dim)) * 0.01
    else:
        # small MLP on d-dim h_yhat
        p["W1"] = jax.random.normal(next(kg).get(), (mlp_dim, emb_dim)) / emb_dim**0.5
        p["b1"] = jnp.zeros(mlp_dim)
        p["W2"] = jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01
        p["b2"] = jnp.zeros(10)

    def fwd(params, x):
        emb = (_factored3_emb(params, x, d_r, d_c, d_v) if emb_type == "factored3"
               else _joint_emb(params, x))                # (B, 784, emb_dim)
        M   = jnp.einsum("bti,btj->bij", emb, emb)        # (B, emb_dim, emb_dim) — matrix!
        h   = M @ params["h_answer"]                       # (B, emb_dim)
        h   = layer_norm(h, params["ln_gamma"], params["ln_beta"])
        if mlp_dim == 0:
            return h @ params["H_y"].T                     # (B, 10)
        else:
            h = jax.nn.gelu(h @ params["W1"].T + params["b1"])
            return h @ params["W2"].T + params["b2"]

    return fwd, p, emb_keys


# ── Architecture B: Projected Gram readout ────────────────────────────────────

def make_proj_gram(emb_type, emb_dims, k, n_val_bins=N_VAL_BINS_DEFAULT):
    """
    Project each emb_t to k-dim, build k×k Gram, linear classifier.

    H_proj ∈ R^{k×d}  (2D → Muon)
    h_t = H_proj @ emb_t              ∈ R^k
    G   = Σ_t h_t ⊗ h_t              ∈ R^{k×k}
    logits = W_out @ LN(flatten(G))

    Interpretation: G captures all pairwise correlations between the k learned
    "feature" directions. Unlike bilinear (rank-1 glimpse), this reads k² features.

    non-emb (factored3 d=64):
      k=8:  8*64 + 2*64 + 10*64    = 1,280
      k=16: 16*64 + 2*256 + 10*256 = 4,096
      k=32: 32*64 + 2*1024+10*1024 = 14,336
      k=64: 64*64 + 2*4096+10*4096 = 53,248
    """
    if emb_type == "factored3":
        d_r, d_c, d_v = emb_dims
        emb_dim = d_r * d_c * d_v
        emb_keys = frozenset({"row_emb", "col_emb", "val_emb"})
    else:
        d = emb_dims[0]
        emb_dim = d
        emb_keys = frozenset({"joint_emb"})

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = (_f3_init(*emb_dims, kg, n_val_bins) if emb_type == "factored3"
          else _joint_init(emb_dims[0], kg, n_val_bins))

    p["H_proj"]   = jax.random.normal(next(kg).get(), (k, emb_dim)) / emb_dim**0.5  # 2D → Muon
    p["ln_gamma"] = jnp.ones(k * k)
    p["ln_beta"]  = jnp.zeros(k * k)
    p["W_out"]    = jax.random.normal(next(kg).get(), (10, k * k)) * 0.01           # 2D → Muon
    p["b_out"]    = jnp.zeros(10)

    def fwd(params, x):
        emb = (_factored3_emb(params, x, d_r, d_c, d_v) if emb_type == "factored3"
               else _joint_emb(params, x))                # (B, 784, emb_dim)
        h   = emb @ params["H_proj"].T                    # (B, 784, k)
        G   = jnp.einsum("bti,btj->bij", h, h)           # (B, k, k)
        G   = G.reshape(G.shape[0], -1)                   # (B, k²)
        G   = layer_norm(G, params["ln_gamma"], params["ln_beta"])
        return G @ params["W_out"].T + params["b_out"]    # (B, 10)

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


if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    F3 = ("factored3", (4, 4, 4))  # emb_dim=64, state_dim=4096
    J  = ("joint",     (32,))      # emb_dim=32, state_dim=1024

    experiments = [
        # ── Architecture A: Bilinear readout (factored3) ─────────────────────
        # User proposal: h_yhat = M @ h_answer; logit_c = h_yc · h_yhat
        ("f3(4,4,4) bilinear+LN +noise30",
         *make_bilinear(*F3, mlp_dim=0), AUG_NOISE30, 30,
         "bilinear dot-product head; 832 non-emb; rank-1 glimpse into M"),

        # Bottleneck: compress d²=4096 state to d=64 via M@h_answer, then MLP
        ("f3(4,4,4) bilinear+LN+mlp32 +noise30",
         *make_bilinear(*F3, mlp_dim=32), AUG_NOISE30, 30,
         "bilinear → MLP(32); ~2.6K non-emb; 64-dim bottleneck"),

        ("f3(4,4,4) bilinear+LN+mlp128 +noise30",
         *make_bilinear(*F3, mlp_dim=128), AUG_NOISE30, 30,
         "bilinear → MLP(128); ~9.8K non-emb; does bottleneck hurt?"),

        ("f3(4,4,4) bilinear+LN+mlp512 +noise30",
         *make_bilinear(*F3, mlp_dim=512), AUG_NOISE30, 30,
         "bilinear → MLP(512); ~34.5K non-emb; large MLP after bottleneck"),

        # ── Architecture B: Projected Gram (factored3) ───────────────────────
        # H_proj ∈ R^{k×64}; G = k×k Gram in subspace; W_out ∈ R^{10×k²}
        ("f3(4,4,4) proj_gram(k=8) +noise30",
         *make_proj_gram(*F3, k=8), AUG_NOISE30, 30,
         "k²=64 features; ~1.3K non-emb; captures 8×8 pairwise correlations"),

        ("f3(4,4,4) proj_gram(k=16) +noise30",
         *make_proj_gram(*F3, k=16), AUG_NOISE30, 30,
         "k²=256 features; ~4.1K non-emb"),

        ("f3(4,4,4) proj_gram(k=32) +noise30",
         *make_proj_gram(*F3, k=32), AUG_NOISE30, 30,
         "k²=1024 features; ~14.3K non-emb"),

        ("f3(4,4,4) proj_gram(k=64) +noise30",
         *make_proj_gram(*F3, k=64), AUG_NOISE30, 30,
         "k²=4096 features; ~53K non-emb; approaches full-Gram linear classifier"),

        # ── joint(d=32) versions ─────────────────────────────────────────────
        ("joint(d=32) bilinear+LN +noise30",
         *make_bilinear(*J, mlp_dim=0), AUG_NOISE30, 30,
         "joint bilinear; 416 non-emb; baseline for ultra-small readout"),

        ("joint(d=32) proj_gram(k=8) +noise30",
         *make_proj_gram(*J, k=8), AUG_NOISE30, 30,
         "joint proj_gram k=8; ~896 non-emb"),

        ("joint(d=32) proj_gram(k=16) +noise30",
         *make_proj_gram(*J, k=16), AUG_NOISE30, 30,
         "joint proj_gram k=16; ~2.8K non-emb"),
    ]

    prev_best = {
        533_898:   ("f3(4,4,4) GELU mlp=128", 0.9846),
        68_298:    ("joint(d=32) GELU mlp=64", 0.9729),
    }

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
            f"  (!) only {nonemb/533898*100:.1f}% of 534K baseline params\n"
            f"  baseline: {base_name} = {base_acc:.4f}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug,
        )
        delta = te_acc - base_acc
        row = {
            "name": label, "experiment": "exp16",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, emb, tr_acc, te_acc, epochs, elapsed,
                        base_acc, base_name))
        logging.info(f"  → {te_acc:.4f}  delta vs closest baseline: {delta:+.4f}")

    print(f"\n{'='*124}")
    print(f"{'Variant':<50} {'NonEmb':>8} {'%534K':>7} {'Train':>8} {'Test':>8} {'Delta':>7} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*124}")
    for label, nonemb, emb, tr, te, ep, t, base, bname in results:
        pct   = f"{nonemb/533898*100:.1f}%"
        flag  = " ★" if te > base else "  "
        delta = f"{te-base:+.4f}"
        print(f"{label:<50} {nonemb:>8,} {pct:>7} {tr:>7.4f} {te:>7.4f}{flag} {delta:>7} {ep:>4} {t:>7.1f}s")
    print(f"{'='*124}")
    print(f"\nBaselines: f3(4,4,4) GELU mlp=128 = 98.46%  |  joint(d=32) GELU mlp=64 = 97.29%")

    logging.info("Updating Pareto plot...")
    replot()
