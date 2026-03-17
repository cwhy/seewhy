"""
Pareto tuning round 2.

Insights from exp6:
  • factored3 > outer at same non-emb (2D structure: row⊗col⊗val)
  • outer+shift has near-zero train/test gap → safe to train much longer
  • factored3 + shift hurts (absolute position encoding ≠ shift-invariant)
  • bins=64 / more epochs overfit outer but NOT factored3 (position-anchored)

Strategy:
  1. More epochs for shift-aug outer models (exploit near-zero gap)
  2. factored3 + bins=64 (finer pixel encoding; position anchored so no overfit risk)
  3. factored3 + pixel-noise aug (preserves positions, adds robustness to val_emb)
  4. factored3(4,4,6): bigger value dimension (emb=96, state=9216) — does larger state help?
  5. factored3(4,4,4) mlp=512/1024: factored features may use more MLP capacity than outer
  6. factored3(2,2,4) mlp=128: 2D structure at the 34K Pareto point
  7. factored3(2,4,4) mlp=128 +shift: can shift help the 135K point?
  8. outer(8,8) mlp=256 +shift+noise (double aug): push 1.06M point further
"""

import json
import time
from pathlib import Path
import sys
import subprocess

import jax
import jax.numpy as jnp
import optax
import logging
from functools import partial

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
EMB_KEYS = {"pos_emb", "val_emb", "row_emb", "col_emb"}


# ── utils ─────────────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

def mlp2(params, x):
    h = jax.nn.relu(x @ params["W1"].T + params["b1"])
    return h @ params["W2"].T + params["b2"]

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
    """Uniform pixel noise ±scale (≈6% of 255). Preserves positions → safe for factored3."""
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)


# ── model: outer (2-way) ──────────────────────────────────────────────────────

def make_outer(d_p, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    emb_dim   = d_p * d_v
    state_dim = emb_dim ** 2
    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K-1)).astype(jnp.int32)
        pe   = params["pos_emb"][None]
        ve   = params["val_emb"][bins]
        emb  = (pe[...,:,None] * ve[...,None,:]).reshape(B, 784, emb_dim)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        return mlp2(params, M)
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "pos_emb":  jax.random.normal(next(kg).get(), (784, d_p)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim), "ln_beta": jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    return fwd, p


# ── model: factored3 (row ⊗ col ⊗ val) ───────────────────────────────────────

def make_factored3(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2
    T    = jnp.arange(784)
    rows = T // 28
    cols = T % 28
    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K-1)).astype(jnp.int32)
        re   = params["row_emb"][rows]
        ce   = params["col_emb"][cols]
        ve   = params["val_emb"][bins]
        pos  = (re[:,:,None] * ce[:,None,:]).reshape(784, d_r*d_c)
        emb  = (pos[None,:,:,None] * ve[:,:,None,:]).reshape(B, 784, emb_dim)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        return mlp2(params, M)
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "row_emb":  jax.random.normal(next(kg).get(), (28, d_r)) * 0.02,
        "col_emb":  jax.random.normal(next(kg).get(), (28, d_c)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim), "ln_beta": jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    return fwd, p


# ── training ──────────────────────────────────────────────────────────────────

AUG_NONE  = "none"
AUG_SHIFT = "shift"
AUG_NOISE = "noise"
AUG_BOTH  = "shift+noise"

def train(name, fwd, params, X_tr, y_tr, X_te, y_te,
          max_epochs=20, target_acc=0.98, aug=AUG_NONE):
    num_batches = len(X_tr) // BATCH_SIZE
    schedule    = optax.cosine_decay_schedule(LR, max_epochs * num_batches)
    optimizer   = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state   = optimizer.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    loss_fn = lambda p, bx, by: jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))

    def acc_fn(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i+BATCH_SIZE]), 1)
            for i in range(0, len(x), BATCH_SIZE)])
        return float(jnp.mean(preds == y))

    @partial(jax.jit, static_argnums=(0,))
    def step(opt, p, s, bx, by):
        loss, g = jax.value_and_grad(loss_fn)(p, bx, by)
        u, s2   = opt.update(g, s, p)
        return optax.apply_updates(p, u), s2, loss

    t0 = time.perf_counter()
    train_acc = test_acc = 0.0
    for epoch in range(max_epochs):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        for b in range(num_batches):
            s, e  = b * BATCH_SIZE, (b+1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            if aug in (AUG_SHIFT, AUG_BOTH):
                aug_rng, sub = jax.random.split(aug_rng)
                bx = random_shift_batch(bx, sub)
            if aug in (AUG_NOISE, AUG_BOTH):
                aug_rng, sub = jax.random.split(aug_rng)
                bx = random_noise_batch(bx, sub)
            params, opt_state, _ = step(optimizer, params, opt_state, bx, by)

        train_acc = acc_fn(params, X_tr, y_tr)
        test_acc  = acc_fn(params, X_te, y_te)
        logging.info(f"[{name}] epoch {epoch:2d}  train={train_acc:.4f}  test={test_acc:.4f}  {time.perf_counter()-t0:.1f}s")
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

    # (label, fwd, params, aug, max_epochs, notes)
    experiments = [
        # ── 1. Longer training for shift-aug outer (near-zero gap → safe) ─────
        ("outer(8,8) mlp=128 +shift 50ep",
         *make_outer(8, 8, 128), AUG_SHIFT, 50,
         "534K non-emb, more epochs on well-generalised model"),
        ("outer(8,8) mlp=256 +shift 50ep",
         *make_outer(8, 8, 256), AUG_SHIFT, 50,
         "1.06M non-emb, more epochs on well-generalised model"),

        # ── 2. factored3 + finer bins (position-anchored → no overfit from bins) ─
        ("factored3(4,4,4) mlp=256 bins=64",
         *make_factored3(4, 4, 4, 256, n_val_bins=64), AUG_NONE, 20,
         "finer pixel quantisation, state=4096"),
        ("factored3(4,4,4) mlp=128 bins=64",
         *make_factored3(4, 4, 4, 128, n_val_bins=64), AUG_NONE, 20,
         "finer pixel quantisation, state=4096"),

        # ── 3. factored3 + pixel noise (keeps positions, robustifies val_emb) ───
        ("factored3(4,4,4) mlp=256 +noise",
         *make_factored3(4, 4, 4, 256), AUG_NOISE, 30,
         "noise aug ±15px, doesn't hurt position encoding"),
        ("factored3(4,4,4) mlp=128 +noise",
         *make_factored3(4, 4, 4, 128), AUG_NOISE, 30,
         "noise aug ±15px"),

        # ── 4. Larger factored3 value dim (richer pixel features in state) ──────
        # factored3(4,4,6): emb=96, state=9216, non-emb≈609K (between 534K & 1.06M)
        ("factored3(4,4,6) mlp=64",
         *make_factored3(4, 4, 6, 64), AUG_NONE, 20,
         "state=9216 emb=96, non-emb=608970"),
        ("factored3(4,4,6) mlp=64 +noise",
         *make_factored3(4, 4, 6, 64), AUG_NOISE, 30,
         "state=9216, noise aug"),

        # ── 5. factored3 with more MLP capacity (can factored features use it?) ─
        ("factored3(4,4,4) mlp=512",
         *make_factored3(4, 4, 4, 512), AUG_NONE, 20,
         "2.11M non-emb, test if factored features saturate at mlp=256"),
        ("factored3(4,4,4) mlp=512 +noise",
         *make_factored3(4, 4, 4, 512), AUG_NOISE, 30,
         "more MLP + noise aug"),

        # ── 6. Small-model Pareto: factored3(2,2,4) at the 34K non-emb point ───
        ("factored3(2,2,4) mlp=128",
         *make_factored3(2, 2, 4, 128), AUG_NONE, 30,
         "state=256, non-emb=34698, 2D structure vs outer(4,4) 94.4%"),

        # ── 7. 135K Pareto point: factored3(2,4,4) with augmentation ────────────
        ("factored3(2,4,4) mlp=128 +noise",
         *make_factored3(2, 4, 4, 128), AUG_NOISE, 30,
         "135K non-emb, noise aug for small factored model"),

        # ── 8. Best combo: outer shift + noise double aug ─────────────────────
        ("outer(8,8) mlp=256 +shift+noise 50ep",
         *make_outer(8, 8, 256), AUG_BOTH, 50,
         "double augmentation: shift + pixel noise"),
    ]

    results = []
    for label, fwd, params, aug, max_ep, notes in experiments:
        nonemb = n_nonemb(params)
        total  = n_total(params)
        logging.info(f"\n{'='*60}\n{label}  (non-emb={nonemb:,}  total={total:,})\n{'='*60}")
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug,
        )
        row = {
            "name": label, "experiment": "exp7",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, epochs, elapsed))
        logging.info(f"  → saved")

    print(f"\n{'='*88}")
    print(f"{'Variant':<42} {'NonEmb':>10} {'Train':>8} {'Test':>8} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*88}")
    prev_best = {534_898: 0.9715, 1_059_594: 0.9727}
    for label, nonemb, tr, te, ep, t in results:
        flag = " ★" if te > prev_best.get(nonemb, 0.97) else ("✓" if te >= 0.97 else "")
        print(f"{label:<42} {nonemb:>10,} {tr:>7.4f} {te:>7.4f}{flag:3} {ep:>4} {t:>7.1f}s")
    print(f"{'='*88}")

    logging.info("Updating Pareto plot...")
    replot()
