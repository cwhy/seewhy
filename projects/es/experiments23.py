"""
exp23 — Single-sided ES: population and hyperparameter sweep.

Baseline (exp22-B): single-sided n_pop=256, sigma=0.001, rank 8→2 ep12 → 95.39%

Key observations from exp22-B:
  - Plateau escape at ep5 (vs ep7 for antithetic)
  - At rank switch ep12: already 91.43% (much higher than antithetic)
  → rank switch at ep8 might better exploit the earlier escape

Sweep:
  A: n_pop=256  sigma=0.001  rank_sw=12  (reference, 95.39%)
  B: n_pop=512  sigma=0.001  rank_sw=12
  C: n_pop=1024 sigma=0.001  rank_sw=12
  D: n_pop=256  sigma=0.001  rank_sw=8
  E: n_pop=512  sigma=0.001  rank_sw=8
  F: n_pop=256  sigma=0.0007 rank_sw=8

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/es/experiments23.py
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
MAX_EPOCHS   = 100
ADAM_LR_INIT = 1e-3
ADAM_LR_END  = 3e-5

JSONL       = Path(__file__).parent / "results_mnist.jsonl"
REPORT_PATH = Path(__file__).parent / "report23.md"

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28


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


def make_f3_bilinear(d_r, d_c, d_v, mlp_dim):
    d  = d_r * d_c * d_v
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb":       jax.random.normal(next(kg).get(), (28,        d_r)) * 0.02,
        "col_emb":       jax.random.normal(next(kg).get(), (28,        d_c)) * 0.02,
        "val_emb":       jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
        "H_answer":      jax.random.normal(next(kg).get(), (1,          d))  * (1.0 / d**0.5),
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


def make_ss_step(fwd, sigma, n_pop, tx, rank):
    """Single-sided ES step with mean baseline."""
    def sample_estimate(pop_key, params, bx, by):
        eps = {}
        for idx, (k, v) in enumerate(params.items()):
            subkey = jax.random.fold_in(pop_key, idx)
            if v.ndim == 2:
                m, n = v.shape
                ab = jax.random.normal(subkey, (m + n, rank))
                eps[k] = (ab[:m] @ ab[m:].T) * sigma
            else:
                eps[k] = jax.random.normal(subkey, v.shape) * sigma
        ce_i = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                fwd({k: v + eps[k] for k, v in params.items()}, bx), by))
        return ce_i, eps

    @jax.jit
    def step(params, opt_state, bx, by, pop_keys):
        ce_vals, epses = jax.vmap(
            sample_estimate, in_axes=(0, None, None, None)
        )(pop_keys, params, bx, by)
        fitnesses = ce_vals - jnp.mean(ce_vals)
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (n_pop * rank * sigma**2),
            epses)
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, jnp.mean(jnp.abs(fitnesses))

    return step


def train_ss(name, fwd, params, X_tr, y_tr, X_te, y_te,
             n_pop, sigma=0.001, rank_switch=12):
    num_batches = len(X_tr) // BATCH_SIZE
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT,
                                              MAX_EPOCHS * num_batches,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    step_hi   = make_ss_step(fwd, sigma, n_pop, tx, rank=8)
    step_lo   = make_ss_step(fwd, sigma, n_pop, tx, rank=2)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    history = {"train": [], "test": [], "fitness": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        step = step_hi if epoch < rank_switch else step_lo
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
            params, opt_state, fit = step(params, opt_state, bx, by, pop_keys)
            epoch_fit.append(float(fit))

        tr = acc_fn(fwd, params, X_tr, y_tr)
        te = acc_fn(fwd, params, X_te, y_te)
        mf = sum(epoch_fit) / len(epoch_fit)
        history["train"].append(tr); history["test"].append(te); history["fitness"].append(mf)
        rank_now = 8 if epoch < rank_switch else 2
        logging.info(f"[{name}] ep {epoch:3d}  train={tr:.4f}  test={te:.4f}"
                     f"  |fit|={mf:.6f}  rank={rank_now}  {time.perf_counter()-t0:.1f}s")

    return tr, te, time.perf_counter() - t0, history


# ── Plots ──────────────────────────────────────────────────────────────────

def make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(MAX_EPOCHS))
    urls   = {}
    colors = ["steelblue", "darkorange", "crimson", "green", "purple", "saddlebrown"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for (label, _, hist), color in zip(results, colors):
        ax.plot(epochs, [v*100 for v in hist["train"]], color=color, linestyle="--", alpha=0.3)
        ax.plot(epochs, [v*100 for v in hist["test"]],  color=color, linewidth=2, label=label)
    ax.axhline(95.0, color="gray", linestyle=":", alpha=0.6, label="95% target")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("exp23 — Single-sided ES: population & hyperparameter sweep (100 epochs)")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["accuracy"] = save_matplotlib_figure("exp23_accuracy", fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for (label, _, hist), color in zip(results, colors):
        ax.plot(epochs, hist["fitness"], color=color, linewidth=2, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("|fitness|")
    ax.set_title("exp23 — Mean |fitness| per epoch")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["fitness"] = save_matplotlib_figure("exp23_fitness", fig)
    plt.close(fig)

    return urls


def write_report(urls, rows):
    date_str = datetime.now().strftime("%Y-%m-%d")
    table = "\n".join(
        f"| {v} | {n} | {s} | {sw} | {tr*100:.2f}% | {te*100:.2f}% | {t:.0f}s |"
        for v, n, s, sw, tr, te, t in rows)
    lines = [
        "# exp23 — Single-sided ES Sweep Report",
        "",
        f"**Date:** {date_str}  **Epochs:** {MAX_EPOCHS}  **Architecture:** f3(4,4,4)",
        "",
        "Baseline from exp22: single-sided n_pop=256, σ=0.001, rank_sw=12 → **95.39%**",
        "",
        "## Results",
        "",
        "| Variant | n_pop | σ | rank_sw | Train | Test | Time |",
        "|---|---|---|---|---|---|---|",
        table,
        "",
        "## Accuracy curves",
        "",
        f"![accuracy]({urls['accuracy']})",
        "",
        "## |fitness| curves",
        "",
        f"![fitness]({urls['fitness']})",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    logging.info(f"Report written to {REPORT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────

SWEEP = [
    # (name,  n_pop, sigma,  rank_switch)
    ("A n=256  σ=0.001 sw=12", 256,  0.001,  12),   # reference (re-run for fair comparison)
    ("B n=512  σ=0.001 sw=12", 512,  0.001,  12),
    ("C n=1024 σ=0.001 sw=12", 1024, 0.001,  12),
    ("D n=256  σ=0.001 sw=8",  256,  0.001,   8),
    ("E n=512  σ=0.001 sw=8",  512,  0.001,   8),
    ("F n=256  σ=0.0007 sw=8", 256,  0.0007,  8),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    all_results, plot_data = [], []

    for name, n_pop, sigma, rank_sw in SWEEP:
        logging.info(f"\n{'='*60}\n{name}\n{'='*60}")
        fwd, params = make_f3_bilinear(4, 4, 4, 256)
        tr, te, t_s, hist = train_ss(
            name, fwd, params, X_train, y_train, X_test, y_test,
            n_pop=n_pop, sigma=sigma, rank_switch=rank_sw)
        logging.info(f"  → train={tr:.4f}  test={te:.4f}  {t_s:.0f}s")
        all_results.append((name, n_pop, sigma, rank_sw, tr, te, t_s))
        plot_data.append((name, n_pop, hist))
        append_result({"experiment": "exp23", "variant": name,
                       "n_pop": n_pop, "sigma": sigma, "rank_switch": rank_sw,
                       "train_acc": round(tr,6), "test_acc": round(te,6),
                       "time_s": round(t_s,1), "epochs": MAX_EPOCHS})

    logging.info("Generating plots and uploading...")
    urls = make_plots(plot_data)
    write_report(urls, all_results)

    print(f"\n{'='*75}")
    print(f"{'Variant':<28} {'n_pop':>6} {'sigma':>7} {'sw':>4} {'Test':>7} {'Time':>8}")
    print(f"{'-'*75}")
    for name, n_pop, sigma, rank_sw, tr, te, t_s in all_results:
        print(f"{name:<28} {n_pop:>6} {sigma:>7.4f} {rank_sw:>4} {te:>7.4f} {t_s:>7.0f}s")
    print(f"{'='*75}")
    print(f"\nReport: {REPORT_PATH}")
    for k, u in urls.items():
        print(f"  {k}: {u}")
