"""
Simplification sweep: find the smallest S2 model (no projection) that still hits ~95%.

The S2 model has 4.27M params — 98% lives in MLP W1 (1024×4096 = 4.2M).
Two independent knobs:
  d       — embedding dim → state is d×d entries
  mlp_dim — hidden units in the 2-layer MLP readout

We also test linear readout (mlp_dim=0) as a lower bound.

Grid tested:
  d=64: mlp ∈ {0, 32, 64, 128, 256}
  d=32: mlp ∈ {0, 64, 128, 256}
  d=48: mlp ∈ {64, 128}           — filling the gap if needed

write_gate (+64 params) is included for the smallest configs to check
if gating recovers accuracy at small d.
"""

import time
import jax
import jax.numpy as jnp
import optax
import logging
from functools import partial

from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_VAL_BINS = 32
BATCH_SIZE = 128
LR         = 1e-3
MAX_EPOCHS = 10
TARGET_ACC = 0.95
SEED       = 42


# ── shared utils ─────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def embeddings(params, x, d):
    bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    return params["pos_emb"][None] + params["val_emb"][bins]   # (B, 784, d)


def make_params(d, mlp_dim, use_write_gate=False, seed=SEED):
    """Build S2-style params for given (d, mlp_dim). mlp_dim=0 → linear readout."""
    state_dim = d * d
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(seed))
    p = {
        "pos_emb":  jax.random.normal(next(kg).get(), (784, d)) * 0.02,
        "val_emb":  jax.random.normal(next(kg).get(), (N_VAL_BINS, d)) * 0.02,
        "ln_gamma": jnp.ones(state_dim),
        "ln_beta":  jnp.zeros(state_dim),
    }
    if mlp_dim > 0:
        p["W1"] = jax.random.normal(next(kg).get(), (mlp_dim, state_dim)) / state_dim**0.5
        p["b1"] = jnp.zeros(mlp_dim)
        p["W2"] = jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01
        p["b2"] = jnp.zeros(10)
    else:
        p["W_out"] = jax.random.normal(next(kg).get(), (10, state_dim)) * 0.01
        p["b_out"] = jnp.zeros(10)
    if use_write_gate:
        p["w_gate"] = jax.random.normal(next(kg).get(), (d,)) * 0.02
    return p


def make_fwd(d, mlp_dim, use_write_gate=False):
    """Return a forward function closed over (d, mlp_dim)."""
    def fwd(params, x):
        B   = x.shape[0]
        emb = embeddings(params, x, d)          # (B, 784, d)

        if use_write_gate:
            w = jax.nn.sigmoid(emb @ params["w_gate"])   # (B, 784)
            M = jnp.einsum("bt,bti,btj->bij", w, emb, emb).reshape(B, -1)
        else:
            M = jnp.einsum("bti,btj->bij", emb, emb).reshape(B, -1)

        M = layer_norm(M, params["ln_gamma"], params["ln_beta"])

        if mlp_dim > 0:
            M = jax.nn.relu(M @ params["W1"].T + params["b1"])
            return M @ params["W2"].T + params["b2"]
        else:
            return M @ params["W_out"].T + params["b_out"]

    return fwd


def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))


# ── training harness ─────────────────────────────────────────────────────────

def train(name, fwd, params, X_tr, y_tr, X_te, y_te):
    num_batches = len(X_tr) // BATCH_SIZE
    schedule    = optax.cosine_decay_schedule(LR, MAX_EPOCHS * num_batches)
    optimizer   = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state   = optimizer.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    loss_fn = lambda p, bx, by: jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by)
    )

    def acc_fn(p, bx, by):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, bx[i:i+BATCH_SIZE]), 1)
            for i in range(0, len(bx), BATCH_SIZE)
        ])
        return float(jnp.mean(preds == by))

    @partial(jax.jit, static_argnums=(0,))
    def step(opt, p, s, bx, by):
        loss, g = jax.value_and_grad(loss_fn)(p, bx, by)
        u, s2   = opt.update(g, s, p)
        return optax.apply_updates(p, u), s2, loss

    t0 = time.perf_counter()
    test_acc = 0.0
    for epoch in range(MAX_EPOCHS):
        perm = jax.random.permutation(next(key_gen).get(), len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            params, opt_state, _ = step(optimizer, params, opt_state, Xs[s:e], ys[s:e])
        test_acc = acc_fn(params, X_te, y_te)
        logging.info(f"[{name}] epoch {epoch}  test={test_acc:.4f}  {time.perf_counter()-t0:.1f}s")
        if test_acc >= TARGET_ACC:
            break
    return test_acc, time.perf_counter() - t0, epoch + 1


# ── experiment grid ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # (d, mlp_dim, use_write_gate)
    configs = [
        # ── d=64: shrink MLP ──────────────────────────────────────────
        (64,   0, False),    # linear readout                 ~62K
        (64,  32, False),    # tiny MLP                      ~195K
        (64,  64, False),    # small MLP                     ~323K
        (64, 128, False),    # medium MLP                    ~585K
        (64, 256, False),    # reference-lite                ~1.1M
        # ── d=32: smaller state ───────────────────────────────────────
        (32,   0, False),    # linear readout                 ~38K
        (32,  64, False),    # tiny MLP                       ~96K
        (32, 128, False),    # small MLP                     ~163K
        (32, 256, False),    # medium MLP                    ~295K
        # ── d=32 + write_gate ────────────────────────────────────────
        (32,  64, True),     # gated tiny MLP                 ~96K
        (32, 128, True),     # gated small MLP               ~163K
        # ── d=48: middle ground ──────────────────────────────────────
        (48,  64, False),    # small MLP                     ~191K
        (48, 128, False),    # medium MLP                    ~374K
    ]

    results = []
    for d, mlp_dim, wg in configs:
        state_dim = d * d
        label = f"d={d} mlp={mlp_dim or 'lin'}"
        if wg:
            label += "+gate"
        fwd    = make_fwd(d, mlp_dim, wg)
        params = make_params(d, mlp_dim, wg)
        np_    = n_params(params)
        logging.info(f"\n{'='*60}\n{label}  ({np_:,} params)  state={state_dim}\n{'='*60}")
        acc, elapsed, epochs = train(label, fwd, params, X_train, y_train, X_test, y_test)
        results.append((label, np_, acc, epochs, elapsed))

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'Variant':<26} {'Params':>10} {'Test acc':>10} {'Epochs':>7} {'Time':>8}")
    print(f"{'-'*72}")
    for label, np_, acc, epochs, elapsed in results:
        flag = "✓" if acc >= TARGET_ACC else " "
        print(f"{label:<26} {np_:>10,} {acc:>9.4f}{flag} {epochs:>7} {elapsed:>7.1f}s")
    print(f"{'='*72}")
