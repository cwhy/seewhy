"""
Pareto tuning round 7: fill the low non-embedding parameter regime.

Current Pareto front has a big gap:
  35K  → 94.37%  outer(4,4) mlp=128
  135K → 96.89%  factored3(2,4,4) mlp=128 GELU +noise Muon

Questions for exp12 (all sub-135K):
  1. Can factored3 beat outer at the 35K budget?
     factored3(2,4,4) mlp=32 and factored3(2,2,4) mlp=128 both land near 35K.
  2. Fill the 35K→135K gap with factored3 at ~68K
     factored3(2,4,4) mlp=64 and factored3(2,2,4) mlp=256 (same W1 size, different state/MLP balance)
  3. State-vs-MLP tradeoff at ~68K budget:
     Big state + small MLP  vs  small state + big MLP
  4. Noise sensitivity at small models:
     noise15 vs noise30 at 68K — small models may not tolerate strong noise
  5. Sub-35K territory:
     factored3(2,2,2) mlp=64  ~4.5K non-emb (tiny, but can factored3 > additive?)
     factored3(2,2,4) mlp=32  ~9.6K non-emb  (between 12K and 35K)
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

def make_adam_tx(schedule):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))


# ── model builders ─────────────────────────────────────────────────────────────

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


# ── training ──────────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          max_epochs=30, target_acc=0.990, aug=AUG_NOISE15, optimizer="muon"):
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


# ── non-emb param formulae ─────────────────────────────────────────────────────
# factored3(d_r, d_c, d_v, mlp):
#   emb_dim = d_r*d_c*d_v,  state_dim = emb_dim^2
#   non-emb = state_dim*(2 + mlp) + 11*mlp + 10
#
# Computed budgets:
#   (2,4,4) mlp=32   → state=1024: 1024*34 + 352 + 10 =  35,178
#   (2,4,4) mlp=64   → state=1024: 1024*66 + 704 + 10 =  68,298
#   (2,2,4) mlp=128  → state=256 : 256*130 + 1408+ 10 =  34,698
#   (2,2,4) mlp=256  → state=256 : 256*258 + 2816+ 10 =  68,874
#   (2,2,2) mlp=64   → state=64  : 64*66   + 704 + 10 =   4,938
#   (2,2,4) mlp=32   → state=256 : 256*34  + 352 + 10 =   9,074
#   (2,4,2) mlp=128  → state=256 : 256*130 + 1408+ 10 =  34,698  (fewer val dims)

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # (label, fwd, params, emb_keys, aug, max_ep, opt, notes)
    experiments = [
        # ── Sub-35K: can factored3 beat additive at tiny budgets? ──────────────
        # Current: 12K additive 93.44%, 35K outer(4,4) 94.37%

        # (2,2,2) has state_dim=64, very small. More of a sanity check.
        ("factored3(2,2,2) mlp=64 GELU +noise15 Muon",
         *make_factored3_gelu(2, 2, 2, 64),
         AUG_NOISE15, 30, "muon",
         "~4.9K non-emb, factored3 vs additive at tiny budget"),

        # (2,2,4) mlp=32: state_dim=256, fills 12K→35K gap
        ("factored3(2,2,4) mlp=32 GELU +noise15 Muon",
         *make_factored3_gelu(2, 2, 4, 32),
         AUG_NOISE15, 30, "muon",
         "~9K non-emb, new point between 12K additive and 35K outer"),

        # ── ~35K budget: factored3 vs outer(4,4) ──────────────────────────────
        # Test two allocations: big state+tiny mlp vs small state+big mlp

        # Big state, small MLP: state=1024, mlp=32
        ("factored3(2,4,4) mlp=32 GELU +noise15 Muon",
         *make_factored3_gelu(2, 4, 4, 32),
         AUG_NOISE15, 30, "muon",
         "~35K, big state(1024) + small MLP(32), vs outer(4,4)"),

        # Small state, big MLP: state=256, mlp=128 (same emb-budget profile)
        ("factored3(2,2,4) mlp=128 GELU +noise15 Muon",
         *make_factored3_gelu(2, 2, 4, 128),
         AUG_NOISE15, 30, "muon",
         "~35K, small state(256) + big MLP(128)"),

        # ── ~68K budget: fill the 35K→135K gap ────────────────────────────────
        # Big state, small MLP: state=1024, mlp=64
        ("factored3(2,4,4) mlp=64 GELU +noise15 Muon",
         *make_factored3_gelu(2, 4, 4, 64),
         AUG_NOISE15, 30, "muon",
         "~68K, big state(1024) + small MLP(64)"),

        # Small state, big MLP: state=256, mlp=256
        ("factored3(2,2,4) mlp=256 GELU +noise15 Muon",
         *make_factored3_gelu(2, 2, 4, 256),
         AUG_NOISE15, 30, "muon",
         "~68K, small state(256) + big MLP(256)"),

        # ── Noise sensitivity at 68K: does noise30 hurt small models? ─────────
        ("factored3(2,4,4) mlp=64 GELU +noise30 Muon",
         *make_factored3_gelu(2, 4, 4, 64),
         AUG_NOISE30, 30, "muon",
         "~68K, noise30 vs noise15 at small model"),

        # ── Best 135K config with noise30 (for comparison baseline) ───────────
        # Already ran noise15 at 135K (96.89%). Does noise30 help here too?
        ("factored3(2,4,4) mlp=128 GELU +noise30 Muon",
         *make_factored3_gelu(2, 4, 4, 128),
         AUG_NOISE30, 30, "muon",
         "135K with noise30, was 96.89% at noise15"),
    ]

    results = []
    for label, fwd, params, emb_keys, aug, max_ep, opt_name, notes in experiments:
        nonemb = n_nonemb(params, emb_keys)
        total  = n_total(params)
        logging.info(
            f"\n{'='*60}\n{label}\n"
            f"  non-emb={nonemb:,}  total={total:,}  opt={opt_name}  aug={aug}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug, optimizer=opt_name,
        )
        row = {
            "name": label, "experiment": "exp12",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, tr_acc, te_acc, epochs, elapsed))
        logging.info(f"  → saved")

    print(f"\n{'='*100}")
    print(f"{'Variant':<56} {'NonEmb':>8} {'Train':>8} {'Test':>8} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*100}")
    prev_best = {
        4_938: 0.0, 9_074: 0.0,
        34_698: 0.9437, 35_178: 0.9437,
        68_298: 0.9437, 68_874: 0.9437,
        134_538: 0.9689,
    }
    for label, nonemb, tr, te, ep, t in results:
        flag = " ★" if te > prev_best.get(nonemb, 0.96) else " "
        print(f"{label:<56} {nonemb:>8,} {tr:>7.4f} {te:>7.4f}{flag:2} {ep:>4} {t:>7.1f}s")
    print(f"{'='*100}")

    logging.info("Updating Pareto plot...")
    replot()
