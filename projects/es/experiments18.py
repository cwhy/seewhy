"""
exp18 — Scale-fix initialization (domain-invariant).

Hypothesis: the 8-epoch plateau is caused by M = emb@emb.T being near-zero
noise at random init (scale 0.02 → M[i,j] ~ 1e-6), giving |fitness| ≈ 0.
If we initialize emb so that M ~ O(1) from epoch 0, the ES gradient signal
is non-zero immediately — no plateau warm-up needed.

Derivation (f3(4,4,4), d=64 features, 784 pixels):
  emb[b,t,i] = row_emb[r,j] * col_emb[c,k] * val_emb[bin,l]
  M[b,i,i]   = Σ_{t=0}^{783} emb[b,t,i]²  ≈  784 · s⁶
  Want M[b,i,i] ~ 1  →  s = 784^(-1/6) ≈ 0.33

This is domain-invariant: it uses only the fixed constants n_pixels=784 and
the embedding shapes, never any training-data statistics.

Variants:
  A: Baseline (scale=0.02, current default)
  B: Scale fix (scale=784^(-1/6) ≈ 0.33 for emb, 1/√d for H_answer)

Both use: Standard ES, σ=0.001, rank=8, n_pop=128, 25 epochs.

Outputs:
  exp18_accuracy.png   — train/test accuracy curves, both variants
  exp18_fitness.png    — |fitness| over epochs, both variants
  report18.md          — concise summary with figure URLs

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/es/experiments18.py
"""

import json, time, sys
from pathlib import Path
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
import logging
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_matplotlib_figure

N_VAL_BINS   = 32
BATCH_SIZE   = 128
SEED         = 42
MAX_EPOCHS   = 25
ADAM_LR_INIT = 1e-3
ADAM_LR_END  = 3e-5

JSONL       = Path(__file__).parent / "results_mnist.jsonl"
REPORT_PATH = Path(__file__).parent / "report18.md"

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28

# Scale-fix constant: want M[b,i,i] = Σ_{784 pixels} emb² ~ 1
# emb = A*B*C (triple product), E[emb²] = s^6  →  784·s^6 = 1  →  s = 784^(-1/6)
SCALE_FIX = 784 ** (-1.0 / 6.0)   # ≈ 0.330


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def random_noise_batch(imgs, rng, scale=30.0):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)


def make_f3_bilinear(d_r, d_c, d_v, mlp_dim, emb_scale=0.02):
    d  = d_r * d_c * d_v
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb":       jax.random.normal(next(kg).get(), (28,        d_r)) * emb_scale,
        "col_emb":       jax.random.normal(next(kg).get(), (28,        d_c)) * emb_scale,
        "val_emb":       jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * emb_scale,
        "H_answer":      jax.random.normal(next(kg).get(), (1,          d))   * (1.0 / d**0.5),
        "ln_out_gamma":  jnp.ones(d),
        "ln_out_beta":   jnp.zeros(d),
        "W1":            jax.random.normal(next(kg).get(), (mlp_dim, d)) / d**0.5,
        "b1":            jnp.zeros(mlp_dim),
        "W2":            jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2":            jnp.zeros(10),
        "ln_hop0_gamma": jnp.ones(d),
        "ln_hop0_beta":  jnp.zeros(d),
    }
    def fwd(params, x):
        B    = x.shape[0]
        bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS];  ce = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, d)
        M    = jnp.einsum("bti,btj->bij", emb, emb)
        h    = jnp.broadcast_to(params["H_answer"][0], (B, d))
        h    = jnp.einsum("bij,bj->bi", M, h)
        h    = layer_norm(h, params["ln_hop0_gamma"], params["ln_hop0_beta"])
        h    = layer_norm(h, params["ln_out_gamma"],  params["ln_out_beta"])
        h    = jax.nn.gelu(h @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]
    return fwd, p


def acc_fn(fwd, params, x, y):
    preds = jnp.concatenate([
        jnp.argmax(fwd(params, x[i:i + BATCH_SIZE]), 1)
        for i in range(0, len(x), BATCH_SIZE)])
    return float(jnp.mean(preds == y))


# ── ES step ────────────────────────────────────────────────────────────────

def make_es_step(fwd, sigma, n_pop, tx, rank=8):
    def make_perturbation(pop_key, params):
        eps = {}
        for idx, (k, v) in enumerate(params.items()):
            subkey = jax.random.fold_in(pop_key, idx)
            if v.ndim == 2:
                m, n = v.shape
                ab = jax.random.normal(subkey, (m + n, rank))
                a, b = ab[:m], ab[m:]
                eps[k] = (a @ b.T) * sigma
            else:
                eps[k] = jax.random.normal(subkey, v.shape) * sigma
        return eps

    def pair_estimate(pop_key, params, bx, by):
        eps   = make_perturbation(pop_key, params)
        p_pos = {k: v + eps[k] for k, v in params.items()}
        p_neg = {k: v - eps[k] for k, v in params.items()}
        ce    = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        return ce(p_pos) - ce(p_neg), eps

    @jax.jit
    def es_step(params, opt_state, bx, by, pop_keys):
        fitnesses, epses = jax.vmap(
            pair_estimate, in_axes=(0, None, None, None)
        )(pop_keys, params, bx, by)
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (n_pop * rank * sigma**2),
            epses,
        )
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, jnp.mean(jnp.abs(fitnesses))

    return es_step


# ── Training loop ──────────────────────────────────────────────────────────

def train_es(name, fwd, params, X_tr, y_tr, X_te, y_te,
             sigma=0.001, n_pop=128, rank=8, max_epochs=MAX_EPOCHS):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    es_step   = make_es_step(fwd, sigma, n_pop, tx, rank=rank)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    history = {"train": [], "test": [], "fitness": []}
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fit = []
        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, opt_state, fit = es_step(params, opt_state, bx, by, pop_keys)
            epoch_fit.append(float(fit))

        train_acc = acc_fn(fwd, params, X_tr, y_tr)
        test_acc  = acc_fn(fwd, params, X_te, y_te)
        mean_fit  = sum(epoch_fit) / len(epoch_fit)
        history["train"].append(train_acc)
        history["test"].append(test_acc)
        history["fitness"].append(mean_fit)
        logging.info(
            f"[{name}] ep {epoch:3d}  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  |fit|={mean_fit:.6f}  {time.perf_counter()-t0:.1f}s"
        )

    return train_acc, test_acc, time.perf_counter() - t0, history


# ── Plotting ───────────────────────────────────────────────────────────────

def make_plots(hist_base, hist_fix, te_base, te_fix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(MAX_EPOCHS))
    urls   = {}

    # Fig 1: Accuracy
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [v * 100 for v in hist_base["train"]],
            color="steelblue", linestyle="--", alpha=0.7, label="Baseline (scale=0.02) — train")
    ax.plot(epochs, [v * 100 for v in hist_base["test"]],
            color="steelblue", linewidth=2, label="Baseline (scale=0.02) — test")
    ax.plot(epochs, [v * 100 for v in hist_fix["train"]],
            color="darkorange", linestyle="--", alpha=0.7,
            label=f"Scale fix (scale≈{SCALE_FIX:.3f}) — train")
    ax.plot(epochs, [v * 100 for v in hist_fix["test"]],
            color="darkorange", linewidth=2,
            label=f"Scale fix (scale≈{SCALE_FIX:.3f}) — test")
    ax.axhline(te_base * 100, color="steelblue",  linestyle=":", alpha=0.5)
    ax.axhline(te_fix  * 100, color="darkorange", linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("exp18 — Scale-fix init: Accuracy (25 epochs)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["accuracy"] = save_matplotlib_figure("exp18_accuracy", fig)
    plt.close(fig)

    # Fig 2: |fitness|
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, hist_base["fitness"], color="steelblue",  label="Baseline (scale=0.02)")
    ax.plot(epochs, hist_fix["fitness"],  color="darkorange", label=f"Scale fix (scale≈{SCALE_FIX:.3f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("|fitness|")
    ax.set_title("exp18 — Mean |fitness| per epoch")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["fitness"] = save_matplotlib_figure("exp18_fitness", fig)
    plt.close(fig)

    return urls


# ── Report ─────────────────────────────────────────────────────────────────

def write_report(urls, tr_base, te_base, tr_fix, te_fix, t_base, t_fix,
                 plateau_base, plateau_fix):
    date_str = datetime.now().strftime("%Y-%m-%d")

    def plateau_str(ep):
        return f"ep {ep}" if ep is not None else "none (no escape in 25ep)"

    lines = [
        "# exp18 — Scale-fix Initialization Report",
        "",
        f"**Date:** {date_str}  **Epochs:** {MAX_EPOCHS}  **n_pop:** 128  "
        f"**σ:** 0.001  **rank:** 8",
        "",
        "## Hypothesis",
        "",
        "Initialize emb at scale `784^(-1/6) ≈ 0.330` so `M = emb@emb.T` has `O(1)` entries "
        "from epoch 0, eliminating the plateau caused by near-zero gradient signal.",
        "",
        "Scale derivation: `M[i,i] = Σ_{784} emb² ≈ 784·s⁶ = 1  →  s = 784^(-1/6)`",
        "",
        "## Results",
        "",
        "| Variant | Train | Test | Time | Plateau escape |",
        "|---|---|---|---|---|",
        f"| Baseline (scale=0.02)           | {tr_base*100:.2f}% | {te_base*100:.2f}% "
        f"| {t_base:.0f}s | {plateau_str(plateau_base)} |",
        f"| Scale fix (scale≈{SCALE_FIX:.3f})    | {tr_fix*100:.2f}%  | {te_fix*100:.2f}%  "
        f"| {t_fix:.0f}s  | {plateau_str(plateau_fix)} |",
        "",
        "## Accuracy curves",
        "",
        f"![accuracy]({urls['accuracy']})",
        "",
        "## |fitness| over time",
        "",
        f"![fitness]({urls['fitness']})",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    logging.info(f"Report written to {REPORT_PATH}")


def find_plateau_escape(history, threshold=0.20):
    """Return first epoch where test acc exceeds threshold, else None."""
    for i, v in enumerate(history["test"]):
        if v > threshold:
            return i
    return None


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info(f"Scale-fix constant: {SCALE_FIX:.6f}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # ── A: Baseline (scale=0.02) ──
    name_a = f"baseline   scale=0.02"
    logging.info(f"\n{'='*60}\n{name_a}\n{'='*60}")
    fwd, params_a = make_f3_bilinear(4, 4, 4, 256, emb_scale=0.02)
    tr_a, te_a, t_a, hist_a = train_es(
        name_a, fwd, params_a, X_train, y_train, X_test, y_test,
        sigma=0.001, n_pop=128, rank=8)
    logging.info(f"  → train={tr_a:.4f}  test={te_a:.4f}  {t_a:.0f}s")
    append_result({"experiment": "exp18", "variant": "baseline",
                   "emb_scale": 0.02,
                   "train_acc": round(tr_a, 6), "test_acc": round(te_a, 6),
                   "time_s": round(t_a, 1), "epochs": MAX_EPOCHS})

    # ── B: Scale fix ──
    name_b = f"scale-fix  scale={SCALE_FIX:.3f}"
    logging.info(f"\n{'='*60}\n{name_b}\n{'='*60}")
    fwd, params_b = make_f3_bilinear(4, 4, 4, 256, emb_scale=SCALE_FIX)
    tr_b, te_b, t_b, hist_b = train_es(
        name_b, fwd, params_b, X_train, y_train, X_test, y_test,
        sigma=0.001, n_pop=128, rank=8)
    logging.info(f"  → train={tr_b:.4f}  test={te_b:.4f}  {t_b:.0f}s")
    append_result({"experiment": "exp18", "variant": "scale-fix",
                   "emb_scale": round(SCALE_FIX, 6),
                   "train_acc": round(tr_b, 6), "test_acc": round(te_b, 6),
                   "time_s": round(t_b, 1), "epochs": MAX_EPOCHS})

    plateau_a = find_plateau_escape(hist_a)
    plateau_b = find_plateau_escape(hist_b)
    logging.info(f"Plateau escape — baseline: ep{plateau_a}  scale-fix: ep{plateau_b}")

    # ── Plots + report ──
    logging.info("Generating plots and uploading...")
    urls = make_plots(hist_a, hist_b, te_a, te_b)
    write_report(urls, tr_a, te_a, tr_b, te_b, t_a, t_b, plateau_a, plateau_b)

    print(f"\n{'='*75}")
    print(f"{'Variant':<45} {'Train':>7} {'Test':>7} {'Time':>7} {'Escape':>8}")
    print(f"{'-'*75}")
    p_a = f"ep{plateau_a}" if plateau_a is not None else "none"
    p_b = f"ep{plateau_b}" if plateau_b is not None else "none"
    print(f"{'baseline   scale=0.02':<45} {tr_a:>7.4f} {te_a:>7.4f} {t_a:>6.0f}s {p_a:>8}")
    print(f"{f'scale-fix  scale={SCALE_FIX:.3f}':<45} {tr_b:>7.4f} {te_b:>7.4f} {t_b:>6.0f}s {p_b:>8}")
    print(f"{'='*75}")
    print(f"\nReport: {REPORT_PATH}")
    print(f"Figures:")
    for k, u in urls.items():
        print(f"  {k}: {u}")
