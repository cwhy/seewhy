"""
Pareto tuning round 13: low-end sweep of the bilinear+MLP architecture.

Current low-end Pareto points:
  832 non-emb:   92.21%  bilinear+LN (dot-product head, no MLP)
  2,602 non-emb: 97.20%  bilinear+LN+mlp32
  9,802 non-emb: 97.82%  bilinear+LN+mlp128

The gap 832→2,602 is unexplored. Questions:
  1. What accuracy does bilinear+mlp8/16/24 achieve at 800–2K non-emb?
  2. Does K=2 (two queries) at the same tiny budget beat K=1?
  3. Does depth=2 help at very small MLP sizes?
  4. How does joint(d=32) bilinear+MLP compare to factored3 at tiny budgets?
     (joint has richer embedding, smaller d → smaller MLP input dim)

─── Param counts ──────────────────────────────────────────────────────────────
f3(4,4,4), d=64:
  K=1 depth=1:  d*(3+mlp) + 11*mlp + 10
    mlp=4:   64*7  + 44+10  =   448 + 54  =   502
    mlp=8:   64*11 + 88+10  =   704 + 98  =   802
    mlp=12:  64*15 + 132+10 =   960 + 142 = 1,102
    mlp=16:  64*19 + 176+10 = 1,216 + 186 = 1,402
    mlp=24:  64*27 + 264+10 = 1,728 + 274 = 2,002
    mlp=32:  64*35 + 352+10 = 2,240 + 362 = 2,602  ← current best
    mlp=48:  64*51 + 528+10 = 3,264 + 538 = 3,802
    mlp=64:  64*67 + 704+10 = 4,288 + 714 = 5,002

  K=2 depth=1: feat_dim=128; 128*(3+mlp) + 11*mlp + 10
    mlp=4:   128*7  + 54  =   896 + 54  =   950
    mlp=8:   128*11 + 98  = 1,408 + 98  = 1,506
    mlp=12:  128*15 + 142 = 1,920 + 142 = 2,062
    mlp=16:  128*19 + 186 = 2,432 + 186 = 2,618
    mlp=24:  128*27 + 274 = 3,456 + 274 = 3,730
    mlp=32:  128*35 + 362 = 4,480 + 362 = 4,842

  K=2 depth=2: K=2 depth=1 + extra LN (2*d=128 params)
    mlp=4:   950  + 128 = 1,078
    mlp=8:   1,506 + 128 = 1,634
    mlp=16:  2,618 + 128 = 2,746

joint(d=32), d=32:
  K=1 depth=1:  32*(3+mlp) + 11*mlp + 10
    mlp=8:   32*11 + 98  = 352 + 98  =   450
    mlp=16:  32*19 + 186 = 608 + 186 =   794
    mlp=32:  32*35 + 362 = 1,120 + 362 = 1,482
    mlp=64:  32*67 + 714 = 2,144 + 714 = 2,858
    mlp=128: 32*131+1418 = 4,192 +1418 = 5,610

  K=2 depth=1, joint: feat_dim=64; 64*(3+mlp) + 11*mlp + 10
    mlp=16:  64*19 + 186 = 1,216 + 186 = 1,402
    mlp=32:  64*35 + 362 = 2,240 + 362 = 2,602
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


# ── model builder ──────────────────────────────────────────────────────────────

def make_deep_bilinear(emb_type, emb_dims, mlp_dim, depth=1, n_queries=1,
                       n_val_bins=N_VAL_BINS_DEFAULT):
    """Bilinear+MLP with optional depth and multi-query. Same as exp17."""
    if emb_type == "factored3":
        d_r, d_c, d_v = emb_dims
        emb_dim  = d_r * d_c * d_v
        emb_keys = frozenset({"row_emb", "col_emb", "val_emb"})
    else:
        d        = emb_dims[0]
        emb_dim  = d
        emb_keys = frozenset({"joint_emb"})

    feat_dim = n_queries * emb_dim
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))

    if emb_type == "factored3":
        p = {
            "row_emb": jax.random.normal(next(kg).get(), (28, d_r))                 * 0.02,
            "col_emb": jax.random.normal(next(kg).get(), (28, d_c))                 * 0.02,
            "val_emb": jax.random.normal(next(kg).get(), (n_val_bins, d_v))         * 0.02,
        }
    else:
        p = {
            "joint_emb": jax.random.normal(next(kg).get(), (784, n_val_bins, d))    * 0.02,
        }

    p["H_answer"]      = jax.random.normal(next(kg).get(), (n_queries, emb_dim)) * 0.02
    p["ln_out_gamma"]  = jnp.ones(feat_dim)
    p["ln_out_beta"]   = jnp.zeros(feat_dim)
    p["W1"] = jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5
    p["b1"] = jnp.zeros(mlp_dim)
    p["W2"] = jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01
    p["b2"] = jnp.zeros(10)
    for i in range(depth):
        p[f"ln_hop{i}_gamma"] = jnp.ones(emb_dim)
        p[f"ln_hop{i}_beta"]  = jnp.zeros(emb_dim)

    def fwd(params, x):
        B = x.shape[0]
        if emb_type == "factored3":
            K    = params["val_emb"].shape[0]
            bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
            re   = params["row_emb"][ROWS]
            ce   = params["col_emb"][COLS]
            ve   = params["val_emb"][bins]
            pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
            emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, emb_dim)
        else:
            K    = params["joint_emb"].shape[1]
            bins = jnp.floor(x / 255.0 * (K - 1)).astype(jnp.int32)
            emb  = params["joint_emb"][T_IDX, bins]

        M    = jnp.einsum("bti,btj->bij", emb, emb)          # (B, d, d)
        outs = []
        for q in range(n_queries):
            h = params["H_answer"][q]
            h = jnp.broadcast_to(h, (B, emb_dim))
            for i in range(depth):
                h = jnp.einsum("bij,bj->bi", M, h)
                h = layer_norm(h, params[f"ln_hop{i}_gamma"], params[f"ln_hop{i}_beta"])
            outs.append(h)
        features = jnp.concatenate(outs, axis=-1)
        features = layer_norm(features, params["ln_out_gamma"], params["ln_out_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

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

    F3 = ("factored3", (4, 4, 4))   # emb_dim=64
    JT = ("joint",     (32,))       # emb_dim=32

    experiments = [
        # ── f3: sweep K=1 mlp at tiny sizes ──────────────────────────────────
        # Fill the 832→2,602 gap and establish how fast accuracy drops with width.
        ("f3(4,4,4) bilinear K=1 mlp=4 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=4), AUG_NOISE30, 30,
         "502 non-emb; extreme low end"),

        ("f3(4,4,4) bilinear K=1 mlp=8 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=8), AUG_NOISE30, 30,
         "802 non-emb; same budget as dot-head (92.21%) but now with MLP"),

        ("f3(4,4,4) bilinear K=1 mlp=16 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=16), AUG_NOISE30, 30,
         "1,402 non-emb"),

        ("f3(4,4,4) bilinear K=1 mlp=24 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=24), AUG_NOISE30, 30,
         "2,002 non-emb"),

        ("f3(4,4,4) bilinear K=1 mlp=48 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=48), AUG_NOISE30, 30,
         "3,802 non-emb; fill 2.6K→5K gap"),

        ("f3(4,4,4) bilinear K=1 mlp=64 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=64), AUG_NOISE30, 30,
         "5,002 non-emb; fill 2.6K→9.8K gap"),

        ("f3(4,4,4) bilinear K=1 mlp=96 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=96), AUG_NOISE30, 30,
         "7,362 non-emb; fill 5K→9.8K gap"),

        # ── f3: K=2 at tiny budgets — do 2 queries help? ─────────────────────
        # K=2 mlp=8: 1,506 non-emb  (vs K=1 mlp=16 at 1,402 — similar budget)
        # K=2 mlp=16: 2,618 non-emb (vs K=1 mlp=32 at 2,602 — same budget!)
        ("f3(4,4,4) bilinear K=2 mlp=8 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=8, n_queries=2), AUG_NOISE30, 30,
         "1,506 non-emb; K=2 at ~1.5K — 2 directions worth more than 1?"),

        ("f3(4,4,4) bilinear K=2 mlp=16 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=16, n_queries=2), AUG_NOISE30, 30,
         "2,618 non-emb; K=2 vs K=1 mlp=32 (2,602) — same budget"),

        ("f3(4,4,4) bilinear K=2 mlp=32 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=32, n_queries=2), AUG_NOISE30, 30,
         "4,842 non-emb; K=2 vs K=1 mlp=64 (5,002) — same budget"),

        # ── f3: depth=2 at tiny MLP ───────────────────────────────────────────
        # depth=2 K=2 mlp=8: 1,634 non-emb — near-free M² helps tiny K=2?
        ("f3(4,4,4) bilinear K=2 depth=2 mlp=8 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=8, depth=2, n_queries=2), AUG_NOISE30, 30,
         "1,634 non-emb; K=2 depth=2 at tiny budget"),

        ("f3(4,4,4) bilinear K=2 depth=2 mlp=16 +noise30",
         *make_deep_bilinear(*F3, mlp_dim=16, depth=2, n_queries=2), AUG_NOISE30, 30,
         "2,746 non-emb; K=2 depth=2 vs K=2 depth=1 mlp=16 (2,618)"),

        # ── joint(d=32): smaller d → cheaper MLP at same capacity ────────────
        # joint embedding is more expressive (per-position lookup) but d=32 → MLP
        # input is 32-dim, so W1 is 4× smaller than f3's 64-dim input.
        # At very small non-emb, this might be a better trade.
        ("joint(d=32) bilinear K=1 mlp=16 +noise30",
         *make_deep_bilinear(*JT, mlp_dim=16), AUG_NOISE30, 30,
         "794 non-emb; joint K=1 mlp=16; richer emb, smaller d, tiny MLP"),

        ("joint(d=32) bilinear K=1 mlp=32 +noise30",
         *make_deep_bilinear(*JT, mlp_dim=32), AUG_NOISE30, 30,
         "1,482 non-emb; joint K=1 mlp=32 vs f3 K=1 mlp=16 (1,402)"),

        ("joint(d=32) bilinear K=1 mlp=64 +noise30",
         *make_deep_bilinear(*JT, mlp_dim=64), AUG_NOISE30, 30,
         "2,858 non-emb; joint K=1 mlp=64 vs f3 K=1 mlp=32 (2,602)"),

        ("joint(d=32) bilinear K=2 mlp=32 +noise30",
         *make_deep_bilinear(*JT, mlp_dim=32, n_queries=2), AUG_NOISE30, 30,
         "2,602 non-emb; joint K=2 mlp=32; same budget as f3 K=1 mlp=32"),
    ]

    prev_best = {
        502:   ("f3 bilinear dot-head (extrapolated)", 0.9221),
        802:   ("f3 bilinear dot-head",                0.9221),
        2_602: ("f3 bilinear K=1 mlp=32",             0.9720),
        9_802: ("f3 bilinear K=1 mlp=128",            0.9782),
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
            f"  baseline: {base_name} = {base_acc:.4f}\n{'='*60}"
        )
        tr_acc, te_acc, elapsed, epochs = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            max_epochs=max_ep, aug=aug,
        )
        delta = te_acc - base_acc
        row = {
            "name": label, "experiment": "exp18",
            "n_nonemb_params": nonemb,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": epochs, "time_s": round(elapsed, 1), "notes": notes,
        }
        append_result(row)
        results.append((label, nonemb, emb, tr_acc, te_acc, epochs, elapsed,
                        base_acc, base_name))
        logging.info(f"  → {te_acc:.4f}  vs {base_name} {base_acc:.4f}: {delta:+.4f}")

    results.sort(key=lambda r: r[1])
    print(f"\n{'='*120}")
    print(f"{'Variant':<52} {'NonEmb':>7} {'Train':>8} {'Test':>8} {'Delta':>7} {'Ep':>4} {'Time':>8}")
    print(f"{'-'*120}")
    for label, nonemb, emb, tr, te, ep, t, base, bname in results:
        flag  = " ★" if te > base else "  "
        delta = f"{te-base:+.4f}"
        print(f"{label:<52} {nonemb:>7,} {tr:>7.4f} {te:>7.4f}{flag} {delta:>7} {ep:>4} {t:>7.1f}s")
    print(f"{'='*120}")
    print("\nCurrent low-end Pareto: 832→92.21%  2,602→97.20%  9,802→97.82%")

    logging.info("Updating Pareto plot...")
    replot()
