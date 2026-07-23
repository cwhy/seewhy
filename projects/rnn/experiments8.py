"""
Pareto tuning round 3: norms, optimizers, gating.

Ideas from DeltaNet / RWKV7 / recent norm papers:
  1. Muon-style optimizer  — Newton-Schulz orthogonalised gradients for W1/W2,
                             Adam for embeddings.  More isotropic updates on the
                             large matrix weights.
  2. Dynamic Tanh (DyT)   — γ·tanh(α·x) replaces LayerNorm.  No mean/var stats;
                             α is a learnable scalar (Zhu et al. 2025).
  3. RMSNorm              — no mean subtraction, only RMS scaling.
  4. GELU activation      — smooth gating in MLP vs hard ReLU threshold.
  5. Spatial gate         — learnable per-position sigmoid weight on each emb_t⊗emb_t
                             term (row + col additive gate, 56 extra params).
                             Inspired by RWKV7 token-mixing gates.
  6. bins=64 + noise      — combine finer pixel quantisation with noise aug.

All experiments use factored3(4,4,4) as the base architecture because it is
currently on the Pareto front at both 534K and 1.06M non-embedding params.
"""

import json
import time
from pathlib import Path
import sys
import subprocess
from functools import partial
from typing import NamedTuple

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

def rms_norm(x, gamma, eps=1e-5):
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return gamma * x / rms

def dyt_norm(x, alpha, gamma):
    """Dynamic Tanh: γ · tanh(α · x).  α is a learnable scalar."""
    return gamma * jnp.tanh(alpha * x)

def n_nonemb(p): return sum(v.size for k,v in p.items() if k not in EMB_KEYS)
def n_total(p):  return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")

def replot():
    subprocess.run([sys.executable, str(PLOT)], check=False)


# ── augmentation ──────────────────────────────────────────────────────────────

def random_shift_batch(imgs_flat, rng, max_shift=2):
    B   = imgs_flat.shape[0]
    pad = max_shift
    imgs = imgs_flat.reshape(B, 28, 28)
    imgs_padded = jnp.pad(imgs, ((0,0),(pad,pad),(pad,pad)))
    rng1, rng2  = jax.random.split(rng)
    row_off = jax.random.randint(rng1, (B,), 0, 2*pad+1)
    col_off = jax.random.randint(rng2, (B,), 0, 2*pad+1)
    @jax.vmap
    def crop(img, r, c): return jax.lax.dynamic_slice(img, (r,c), (28,28))
    return crop(imgs_padded, row_off, col_off).reshape(B, 784)

def random_noise_batch(imgs, rng, scale=15.0):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)

AUG_NONE  = "none"
AUG_NOISE = "noise"


# ── Muon (Newton-Schulz preconditioned Adam) ──────────────────────────────────

def _newton_schulz(G, steps=5):
    """Quintic Newton-Schulz iteration to (approx.) orthogonalise G."""
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = jnp.linalg.norm(G)
    X = G / (norm + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a*X + b*(A@X) + c*(A@(A@X))
    return X * norm   # preserve gradient scale

def apply_ns_to_2d():
    """Optax GradientTransformation: NS-orthogonalise each 2-D gradient leaf."""
    def init(params): return optax.EmptyState()
    def update(updates, state, params=None):
        return jax.tree_util.tree_map(
            lambda g: _newton_schulz(g) if g.ndim == 2 else g,
            updates,
        ), state
    return optax.GradientTransformation(init, update)

def make_muon_tx(params, schedule):
    """NS preconditioning for 2-D weight matrices, plain Adam for everything else."""
    mat_keys = frozenset(k for k, v in params.items()
                         if v.ndim == 2 and k not in EMB_KEYS)
    labels = {k: "muon" if k in mat_keys else "adam" for k in params}

    muon_branch = optax.chain(apply_ns_to_2d(), optax.adam(schedule))
    adam_branch = optax.adam(schedule)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform({"muon": muon_branch, "adam": adam_branch}, labels),
    )

def make_adam_tx(schedule):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))


# ── model: factored3 variants ─────────────────────────────────────────────────

T_IDX = jnp.arange(784)
ROWS  = T_IDX // 28
COLS  = T_IDX % 28


def _factored3_base_params(d_r, d_c, d_v, mlp_dim, n_val_bins, extra=None):
    state_dim = (d_r * d_c * d_v) ** 2
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    emb_dim = d_r * d_c * d_v
    p = {
        "row_emb":  jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb":  jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    if extra:
        p.update(extra(state_dim, d_r, d_c, d_v, next(kg)))
    return p, state_dim, emb_dim


def _gram(params, x, d_r, d_c, d_v, emb_dim):
    B    = x.shape[0]
    K    = params["val_emb"].shape[0]
    bins = jnp.floor(x / 255.0 * (K-1)).astype(jnp.int32)
    re   = params["row_emb"][ROWS]
    ce   = params["col_emb"][COLS]
    ve   = params["val_emb"][bins]
    pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r*d_c)
    emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, emb_dim)
    M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
    return M


def make_factored3(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """Standard factored3 with LayerNorm + ReLU (matches exp7)."""
    p, state_dim, emb_dim = _factored3_base_params(d_r, d_c, d_v, mlp_dim, n_val_bins, extra=
        lambda sd, *_a, **_k: {"ln_gamma": jnp.ones(sd), "ln_beta": jnp.zeros(sd)})

    def fwd(params, x):
        M = _gram(params, x, d_r, d_c, d_v, emb_dim)
        M = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.relu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


def make_factored3_rms(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """factored3 with RMSNorm (no mean subtraction) + ReLU."""
    p, state_dim, emb_dim = _factored3_base_params(d_r, d_c, d_v, mlp_dim, n_val_bins, extra=
        lambda sd, *_a, **_k: {"rms_gamma": jnp.ones(sd)})

    def fwd(params, x):
        M = _gram(params, x, d_r, d_c, d_v, emb_dim)
        M = rms_norm(M, params["rms_gamma"])
        h = jax.nn.relu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


def make_factored3_dyt(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT,
                       alpha_init=0.01):
    """factored3 with Dynamic Tanh norm: γ · tanh(α · x)."""
    p, state_dim, emb_dim = _factored3_base_params(d_r, d_c, d_v, mlp_dim, n_val_bins, extra=
        lambda sd, *_a, **_k: {
            "dyt_alpha": jnp.array(alpha_init),   # scalar
            "dyt_gamma": jnp.ones(sd),
        })

    def fwd(params, x):
        M = _gram(params, x, d_r, d_c, d_v, emb_dim)
        M = dyt_norm(M, params["dyt_alpha"], params["dyt_gamma"])
        h = jax.nn.relu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


def make_factored3_gelu(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """factored3 with LayerNorm + GELU (smooth activation)."""
    p, state_dim, emb_dim = _factored3_base_params(d_r, d_c, d_v, mlp_dim, n_val_bins, extra=
        lambda sd, *_a, **_k: {"ln_gamma": jnp.ones(sd), "ln_beta": jnp.zeros(sd)})

    def fwd(params, x):
        M = _gram(params, x, d_r, d_c, d_v, emb_dim)
        M = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h = jax.nn.gelu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


def make_factored3_sgate(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """factored3 + spatial gate: M = Σ_t σ(a_row[r] + a_col[c]) · emb_t⊗emb_t.

    Inspired by RWKV7 per-token gating: learn which pixel positions matter most.
    Extra params: 28 (row scores) + 28 (col scores) = 56.  Non-emb count unchanged.
    """
    p, state_dim, emb_dim = _factored3_base_params(d_r, d_c, d_v, mlp_dim, n_val_bins, extra=
        lambda sd, *_a, **_k: {
            "ln_gamma":   jnp.ones(sd),
            "ln_beta":    jnp.zeros(sd),
            "gate_row":   jnp.zeros(28),   # additive gate score per row
            "gate_col":   jnp.zeros(28),   # additive gate score per col
        })

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K-1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS]
        ce   = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r*d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, emb_dim)
        # RWKV7-style gate: sigmoid(score_row + score_col) per pixel, (784,)
        gate = jax.nn.sigmoid(
            params["gate_row"][ROWS] + params["gate_col"][COLS]
        )  # (784,)
        # weighted Gram: M_ij = Σ_t gate_t · emb_t_i · emb_t_j
        M    = jnp.einsum("t,bti,btj->bij", gate, emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        h    = jax.nn.relu(M @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


# ── training ──────────────────────────────────────────────────────────────────

def train(name, fwd, params, X_tr, y_tr, X_te, y_te,
          max_epochs=20, target_acc=0.985, aug=AUG_NONE, optimizer="adam"):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    schedule    = optax.cosine_decay_schedule(LR, total_steps)

    if optimizer == "muon":
        tx = make_muon_tx(params, schedule)
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

    # (label, fwd, params, aug, max_epochs, optimizer, notes)
    experiments = [
        # ── 1. Muon: NS-orthogonalised gradients for W1/W2 ────────────────────
        # hypothesis: more isotropic matrix updates → better convergence on 534K+ models
        ("factored3(4,4,4) mlp=128 +noise Muon",
         *make_factored3(4, 4, 4, 128), AUG_NOISE, 30, "muon",
         "NS-preconditioned Adam for W1/W2, Adam for emb"),
        ("factored3(4,4,4) mlp=256 +noise Muon",
         *make_factored3(4, 4, 4, 256), AUG_NOISE, 30, "muon",
         "NS-preconditioned Adam for W1/W2, 1.06M"),

        # ── 2. Dynamic Tanh norm (Zhu et al. 2025) ────────────────────────────
        # replaces LayerNorm stats computation with learnable tanh(α·x)
        ("factored3(4,4,4) mlp=128 +noise DyT",
         *make_factored3_dyt(4, 4, 4, 128), AUG_NOISE, 30, "adam",
         "DyT norm: gamma*tanh(alpha*x), alpha_init=0.01"),
        ("factored3(4,4,4) mlp=256 +noise DyT",
         *make_factored3_dyt(4, 4, 4, 256), AUG_NOISE, 30, "adam",
         "DyT norm, 1.06M"),

        # ── 3. RMSNorm (no mean subtraction) ──────────────────────────────────
        ("factored3(4,4,4) mlp=128 +noise RMSNorm",
         *make_factored3_rms(4, 4, 4, 128), AUG_NOISE, 30, "adam",
         "RMSNorm on gram matrix"),

        # ── 4. GELU activation (smooth gate vs hard ReLU) ─────────────────────
        ("factored3(4,4,4) mlp=128 +noise GELU",
         *make_factored3_gelu(4, 4, 4, 128), AUG_NOISE, 30, "adam",
         "GELU in MLP, otherwise same as +noise"),
        ("factored3(4,4,4) mlp=256 +noise GELU",
         *make_factored3_gelu(4, 4, 4, 256), AUG_NOISE, 30, "adam",
         "GELU, 1.06M"),

        # ── 5. Spatial gate (RWKV7-style per-position gating) ─────────────────
        # σ(row_score + col_score) gates each pixel's contribution to the Gram
        ("factored3(4,4,4) mlp=128 +noise +sgate",
         *make_factored3_sgate(4, 4, 4, 128), AUG_NOISE, 30, "adam",
         "spatial sigmoid gate per pixel (56 extra params)"),
        ("factored3(4,4,4) mlp=256 +noise +sgate",
         *make_factored3_sgate(4, 4, 4, 256), AUG_NOISE, 30, "adam",
         "spatial gate, 1.06M"),

        # ── 6. bins=64 + noise (combine both ideas that helped individually) ───
        ("factored3(4,4,4) mlp=128 +noise bins=64",
         *make_factored3(4, 4, 4, 128, n_val_bins=64), AUG_NOISE, 30, "adam",
         "finer pixel quantisation + noise aug"),
        ("factored3(4,4,4) mlp=256 +noise bins=64",
         *make_factored3(4, 4, 4, 256, n_val_bins=64), AUG_NOISE, 30, "adam",
         "finer pixel quantisation + noise aug, 1.06M"),
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
            "name": label, "experiment": "exp8",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, epochs, elapsed))
        logging.info(f"  → saved to JSONL")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"{'Variant':<46} {'NonEmb':>10} {'Train':>8} {'Test':>8} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*92}")
    # previous bests at these param counts
    prev_best = {533_898: 0.9804, 1_059_594: 0.9801}
    for label, nonemb, tr, te, ep, t in results:
        flag = " ★" if te > prev_best.get(nonemb, 0.98) else ("✓" if te >= 0.98 else " ")
        print(f"{label:<46} {nonemb:>10,} {tr:>7.4f} {te:>7.4f}{flag:2} {ep:>4} {t:>7.1f}s")
    print(f"{'='*92}")

    logging.info("Updating Pareto plot...")
    replot()
