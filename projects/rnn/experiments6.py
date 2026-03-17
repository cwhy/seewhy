"""
Push accuracy and Pareto front.

Since training accuracy is not saturated (~97.3% train vs 96.5% test), the model is
capacity-limited, not overfitting. New ideas:

  factored3   — 3-way outer product embedding: row_emb[r] ⊗ col_emb[c] ⊗ val_emb[bin]
                Encodes 2D spatial structure explicitly. M_T captures 6th-order stats:
                products of (row-pair) × (col-pair) × (val-pair) across all positions.
                Far fewer embedding params (28·d_r + 28·d_c + 32·d_v vs 784·d).

  shift_aug   — Random 2D shift during training: pad 28×28→32×32, random crop 28×28.
                Gives translational invariance; standard trick for pushing MNIST above 97%.

  val_bins=64 — Finer pixel quantisation (64 instead of 32 bins).
                val_emb doubles but is negligible; model can distinguish finer intensity.

  longer      — More epochs (40) for the best current model; training acc not plateaued.

  larger_state — outer(8,12): emb_dim=96, state=9216 with mlp=64 (similar total non-emb).
                 Tests whether a richer state (more capacity in M_T) beats a bigger MLP.

Results are appended to results_mnist.jsonl and plot_pareto_mnist.py is called at the end.
"""

import json
import time
from pathlib import Path
import sys

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

JSONL      = Path(__file__).parent / "results_mnist.jsonl"
EMB_KEYS   = {"pos_emb", "val_emb", "row_emb", "col_emb"}


# ── utils ─────────────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def mlp2(params, x):
    h = jax.nn.relu(x @ params["W1"].T + params["b1"])
    return h @ params["W2"].T + params["b2"]


def n_nonemb(params):
    return sum(v.size for k, v in params.items() if k not in EMB_KEYS)


def n_total(params):
    return sum(v.size for v in jax.tree_util.tree_leaves(params))


# ── augmentation ──────────────────────────────────────────────────────────────

def random_shift_batch(imgs_flat, rng, max_shift=2):
    """Randomly shift each 28×28 image by up to max_shift pixels in each direction."""
    B   = imgs_flat.shape[0]
    pad = max_shift
    imgs = imgs_flat.reshape(B, 28, 28)
    imgs_padded = jnp.pad(imgs, ((0, 0), (pad, pad), (pad, pad)))   # (B, 32, 32)

    rng1, rng2 = jax.random.split(rng)
    row_off = jax.random.randint(rng1, (B,), 0, 2 * pad + 1)
    col_off = jax.random.randint(rng2, (B,), 0, 2 * pad + 1)

    @jax.vmap
    def crop(img, r, c):
        return jax.lax.dynamic_slice(img, (r, c), (28, 28))

    return crop(imgs_padded, row_off, col_off).reshape(B, 784)


# ── model: outer product (2-way) ─────────────────────────────────────────────

def make_outer(d_p, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    emb_dim   = d_p * d_v
    state_dim = emb_dim ** 2

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
        pe   = params["pos_emb"][None]
        ve   = params["val_emb"][bins]
        emb  = (pe[..., :, None] * ve[..., None, :]).reshape(B, 784, emb_dim)
        M    = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M    = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        return mlp2(params, M)

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
        "pos_emb":  jax.random.normal(next(kg).get(), (784, d_p)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (n_val_bins, d_v)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2": jnp.zeros(10),
    }
    return fwd, p


# ── model: 3-way factored outer (row ⊗ col ⊗ val) ────────────────────────────

def make_factored3(d_r, d_c, d_v, mlp_dim, n_val_bins=N_VAL_BINS_DEFAULT):
    """
    emb_t = row_emb[t//28] ⊗ col_emb[t%28] ⊗ val_emb[bin(pixel_t)]
    shape: (d_r * d_c * d_v,)   — same state_dim formula as outer product
    embedding params: 28*d_r + 28*d_c + n_val_bins*d_v  (tiny vs 784*d)
    """
    emb_dim   = d_r * d_c * d_v
    state_dim = emb_dim ** 2

    T    = jnp.arange(784)
    rows = T // 28
    cols = T % 28

    def fwd(params, x):
        B    = x.shape[0]
        K    = params["val_emb"].shape[0]
        bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)   # (B, 784)

        re = params["row_emb"][rows]   # (784, d_r)
        ce = params["col_emb"][cols]   # (784, d_c)
        ve = params["val_emb"][bins]   # (B, 784, d_v)

        # pos part: (784, d_r*d_c) — outer product of row and col
        pos = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
        # full 3-way outer: (B, 784, d_r*d_c, d_v) → (B, 784, emb_dim)
        emb = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, emb_dim)

        M = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)
        M = layer_norm(M, params["ln_gamma"], params["ln_beta"])
        return mlp2(params, M)

    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p  = {
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
    return fwd, p


# ── training harness ─────────────────────────────────────────────────────────

def train(name, fwd, params, X_tr, y_tr, X_te, y_te,
          max_epochs=20, target_acc=0.97, use_aug=False):
    num_batches = len(X_tr) // BATCH_SIZE
    schedule    = optax.cosine_decay_schedule(LR, max_epochs * num_batches)
    optimizer   = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state   = optimizer.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    loss_fn = lambda p, bx, by: jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by)
    )

    def acc_fn(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i+BATCH_SIZE]), 1)
            for i in range(0, len(x), BATCH_SIZE)
        ])
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
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            if use_aug:
                aug_rng, sub = jax.random.split(aug_rng)
                bx = random_shift_batch(bx, sub)
            params, opt_state, _ = step(optimizer, params, opt_state, bx, by)

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


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


# ── experiment grid ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # (label, fwd, params, use_aug, max_epochs, notes)
    experiments = [
        # ── factored3: same non-emb as outer(8,8), test 2D structure ──────────
        ("factored3(4,4,4) mlp=128",
         *make_factored3(4, 4, 4, 128), False, 20,
         "3-way outer, state=4096, non-emb=533898"),
        ("factored3(4,4,4) mlp=256",
         *make_factored3(4, 4, 4, 256), False, 20,
         "3-way outer, state=4096, non-emb=1059594"),

        # ── factored3: same non-emb as outer(4,8), small model ────────────────
        ("factored3(2,4,4) mlp=128",
         *make_factored3(2, 4, 4, 128), False, 20,
         "3-way outer, state=1024, non-emb=134538"),

        # ── shift augmentation on best outer model ────────────────────────────
        ("outer(8,8) mlp=256 +shift",
         *make_outer(8, 8, 256), True, 30,
         "2D random shift aug, max_shift=2"),
        ("outer(8,8) mlp=128 +shift",
         *make_outer(8, 8, 128), True, 30,
         "2D random shift aug, max_shift=2"),

        # ── factored3 + augmentation ──────────────────────────────────────────
        ("factored3(4,4,4) mlp=256 +shift",
         *make_factored3(4, 4, 4, 256), True, 30,
         "3-way outer + shift aug"),
        ("factored3(4,4,4) mlp=128 +shift",
         *make_factored3(4, 4, 4, 128), True, 30,
         "3-way outer + shift aug"),

        # ── larger val bins: finer pixel quantization ─────────────────────────
        ("outer(8,8) mlp=256 bins=64",
         *make_outer(8, 8, 256, n_val_bins=64), False, 20,
         "64 value bins instead of 32"),

        # ── larger state: outer(8,12)=96-dim emb, state=9216, mlp=64 ─────────
        ("outer(8,12) mlp=64",
         *make_outer(8, 12, 64), False, 20,
         "state=9216, non-emb=608906"),
        ("outer(8,12) mlp=128",
         *make_outer(8, 12, 128), False, 20,
         "state=9216, non-emb=1208714"),

        # ── long training: 40 epochs on best model ────────────────────────────
        ("outer(8,8) mlp=256 40ep",
         *make_outer(8, 8, 256), False, 40,
         "40 epochs, training not yet saturated"),
    ]

    results = []
    for label, fwd, params, use_aug, max_ep, notes in experiments:
        nonemb = n_nonemb(params)
        total  = n_total(params)
        logging.info(f"\n{'='*60}\n{label}  (non-emb={nonemb:,}  total={total:,})\n{'='*60}")
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, X_train, y_train, X_test, y_test,
            max_epochs=max_ep, use_aug=use_aug,
        )
        row = {
            "name": label,
            "experiment": "exp6",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6),
            "test_acc":  round(te_acc, 6),
            "epochs":    epochs,
            "time_s":    round(elapsed, 1),
            "notes":     notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, epochs, elapsed))
        logging.info(f"  → appended to {JSONL.name}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*82}")
    print(f"{'Variant':<35} {'NonEmb':>10} {'Train':>8} {'Test':>8} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*82}")
    TARGET = 0.97
    for label, nonemb, tr, te, ep, t in results:
        flag = "✓" if te >= TARGET else (" ✓96" if te >= 0.96 else "")
        print(f"{label:<35} {nonemb:>10,} {tr:>7.4f} {te:>7.4f}{flag:4} {ep:>4} {t:>7.1f}s")
    print(f"{'='*82}")

    # ── update Pareto plot ────────────────────────────────────────────────────
    logging.info("Updating Pareto plot...")
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, str(Path(__file__).parent / "plot_pareto_mnist.py")])
