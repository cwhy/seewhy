"""
exp20 — Delightful ES (DG-ES): apply Delightful Policy Gradient gating to ES.

Paper: "Delightful Policy Gradient" (arxiv 2603.14608v1)
  Standard PG:   Δθ ∝ Σ_t  U_t · ∇ log π(A_t)
  Delightful PG: Δθ ∝ Σ_t  σ(U_t · ℓ_t / η) · U_t · ∇ log π(A_t)
  where ℓ_t = -log π(A_t)  (action surprisal)

Mapping to antithetic ES:
  U_i   = fitness_i = CE(params+eps_i) - CE(params-eps_i)
  ℓ_i   = 0.5 · Σ_k ||[A_k, B_k]||²_F   (neg log-prob of drawing the perturbation)
  ∇ log π · U_i / σ²  →  fitness_i · eps_i / (n_pop · rank · σ²)

Scale issue: raw ℓ_i ~ O(3000), fitness_i ~ O(0.01) → χ saturates.
Fix: use centered surprise  ℓ̃_i = ℓ_i - mean(ℓ)  (std ≈ 55 for rank=8)
     gate  w_i = sigmoid(fitness_i · ℓ̃_i / η)

The gate suppresses "blunders" (rare large eps that accidentally hurt) and
amplifies "breakthroughs" (rare large eps that help).

Variants (all use rank schedule 8→2 at ep12, 100 epochs):
  A: Standard ES            (control, best known non-GD config)
  B: DG-ES  η=1             (Delightful gating, paper default)
  C: DG-ES  η=0.1           (sharper gate)

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/es/experiments20.py
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
RANK_SWITCH_EP = 12   # switch rank 8→2 at this epoch (best from exp15)

JSONL       = Path(__file__).parent / "results_mnist.jsonl"
REPORT_PATH = Path(__file__).parent / "report20.md"

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


# ── ES steps ───────────────────────────────────────────────────────────────

def make_standard_es_step(fwd, sigma, n_pop, tx, rank):
    def make_perturbation(pop_key, params):
        eps = {}
        for idx, (k, v) in enumerate(params.items()):
            subkey = jax.random.fold_in(pop_key, idx)
            if v.ndim == 2:
                m, n = v.shape
                ab = jax.random.normal(subkey, (m + n, rank))
                eps[k] = (ab[:m] @ ab[m:].T) * sigma
            else:
                eps[k] = jax.random.normal(subkey, v.shape) * sigma
        return eps

    def pair_estimate(pop_key, params, bx, by):
        eps   = make_perturbation(pop_key, params)
        ce    = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        return ce({k: v + eps[k] for k, v in params.items()}) \
             - ce({k: v - eps[k] for k, v in params.items()}), eps

    @jax.jit
    def es_step(params, opt_state, bx, by, pop_keys):
        fitnesses, epses = jax.vmap(
            pair_estimate, in_axes=(0, None, None, None)
        )(pop_keys, params, bx, by)
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (n_pop * rank * sigma**2),
            epses)
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, jnp.mean(jnp.abs(fitnesses))

    return es_step


def make_dg_es_step(fwd, sigma, n_pop, tx, rank, eta=1.0):
    """
    Delightful ES: weight each pair by sigmoid(fitness_i * centered_surprise_i / eta).
    surprise_i = 0.5 * ||[A_k, B_k]||^2 for all param keys (neg log-prob of drawing eps).
    """
    def make_perturbation_with_surprise(pop_key, params):
        eps     = {}
        surprise = 0.0
        for idx, (k, v) in enumerate(params.items()):
            subkey = jax.random.fold_in(pop_key, idx)
            if v.ndim == 2:
                m, n = v.shape
                ab = jax.random.normal(subkey, (m + n, rank))
                eps[k]   = (ab[:m] @ ab[m:].T) * sigma
                surprise = surprise + 0.5 * jnp.sum(ab ** 2)   # -log p(ab)
            else:
                z        = jax.random.normal(subkey, v.shape)
                eps[k]   = z * sigma
                surprise = surprise + 0.5 * jnp.sum(z ** 2)    # -log p(z)
        return eps, surprise

    def pair_estimate(pop_key, params, bx, by):
        eps, surprise = make_perturbation_with_surprise(pop_key, params)
        ce = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        fitness = ce({k: v + eps[k] for k, v in params.items()}) \
                - ce({k: v - eps[k] for k, v in params.items()})
        return fitness, eps, surprise

    @jax.jit
    def es_step(params, opt_state, bx, by, pop_keys):
        fitnesses, epses, surprises = jax.vmap(
            pair_estimate, in_axes=(0, None, None, None)
        )(pop_keys, params, bx, by)

        # Center surprise across population (avoids sigmoid saturation)
        surprise_c = surprises - jnp.mean(surprises)

        # Delight = fitness × centered_surprise; gate = sigmoid(delight / η)
        delight = fitnesses * surprise_c
        gates   = jax.nn.sigmoid(delight / eta)

        weighted_fit = fitnesses * gates
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", weighted_fit, e) / (n_pop * rank * sigma**2),
            epses)
        updates, new_opt_state = tx.update(grad, opt_state, params)
        mean_abs_fit = jnp.mean(jnp.abs(fitnesses))
        mean_gate    = jnp.mean(gates)
        return optax.apply_updates(params, updates), new_opt_state, mean_abs_fit, mean_gate

    return es_step


# ── Training loops ─────────────────────────────────────────────────────────

def train_standard(name, fwd, params, X_tr, y_tr, X_te, y_te,
                   sigma=0.001, n_pop=128, rank_hi=8, rank_lo=2,
                   rank_switch=RANK_SWITCH_EP, max_epochs=MAX_EPOCHS):
    num_batches = len(X_tr) // BATCH_SIZE
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT,
                                              max_epochs * num_batches,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    # Build both rank steps once (avoid recompile on switch)
    step_hi = make_standard_es_step(fwd, sigma, n_pop, tx, rank_hi)
    step_lo = make_standard_es_step(fwd, sigma, n_pop, tx, rank_lo)

    opt_state = tx.init(params)
    history   = {"train": [], "test": [], "fitness": []}
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
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
        history["train"].append(tr); history["test"].append(te)
        history["fitness"].append(mf)
        rank_now = rank_hi if epoch < rank_switch else rank_lo
        logging.info(
            f"[{name}] ep {epoch:3d}  train={tr:.4f}  test={te:.4f}"
            f"  |fit|={mf:.6f}  rank={rank_now}  {time.perf_counter()-t0:.1f}s")

    return tr, te, time.perf_counter() - t0, history


def train_dg(name, fwd, params, X_tr, y_tr, X_te, y_te,
             sigma=0.001, n_pop=128, rank_hi=8, rank_lo=2,
             rank_switch=RANK_SWITCH_EP, eta=1.0, max_epochs=MAX_EPOCHS):
    num_batches = len(X_tr) // BATCH_SIZE
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT,
                                              max_epochs * num_batches,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    step_hi = make_dg_es_step(fwd, sigma, n_pop, tx, rank_hi, eta=eta)
    step_lo = make_dg_es_step(fwd, sigma, n_pop, tx, rank_lo, eta=eta)

    opt_state = tx.init(params)
    history   = {"train": [], "test": [], "fitness": [], "gate": []}
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        step = step_hi if epoch < rank_switch else step_lo
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fit, epoch_gate = [], []
        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, opt_state, fit, gate = step(params, opt_state, bx, by, pop_keys)
            epoch_fit.append(float(fit)); epoch_gate.append(float(gate))

        tr = acc_fn(fwd, params, X_tr, y_tr)
        te = acc_fn(fwd, params, X_te, y_te)
        mf = sum(epoch_fit) / len(epoch_fit)
        mg = sum(epoch_gate) / len(epoch_gate)
        history["train"].append(tr); history["test"].append(te)
        history["fitness"].append(mf); history["gate"].append(mg)
        rank_now = rank_hi if epoch < rank_switch else rank_lo
        logging.info(
            f"[{name}] ep {epoch:3d}  train={tr:.4f}  test={te:.4f}"
            f"  |fit|={mf:.6f}  gate={mg:.3f}  rank={rank_now}  {time.perf_counter()-t0:.1f}s")

    return tr, te, time.perf_counter() - t0, history


# ── Plots ──────────────────────────────────────────────────────────────────

def make_plots(results):
    """results: list of (label, color, history)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(MAX_EPOCHS))
    urls   = {}

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, hist in results:
        ax.plot(epochs, [v*100 for v in hist["train"]], color=color, linestyle="--", alpha=0.4)
        ax.plot(epochs, [v*100 for v in hist["test"]],  color=color, linewidth=2, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("exp20 — Delightful ES vs Standard ES (rank schedule 8→2, 100 epochs)")
    ax.axvline(RANK_SWITCH_EP, color="gray", linestyle=":", alpha=0.5, label="rank switch ep12")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["accuracy"] = save_matplotlib_figure("exp20_accuracy", fig)
    plt.close(fig)

    # |fitness| and gate
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, color, hist in results:
        axes[0].plot(epochs, hist["fitness"], color=color, linewidth=2, label=label)
        if "gate" in hist:
            axes[1].plot(epochs, hist["gate"], color=color, linewidth=2, label=label)
    axes[0].set_title("|fitness| per epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("mean gate per epoch (DG variants)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    fig.tight_layout()
    urls["diagnostics"] = save_matplotlib_figure("exp20_diagnostics", fig)
    plt.close(fig)

    return urls


# ── Report ─────────────────────────────────────────────────────────────────

def write_report(urls, rows):
    date_str = datetime.now().strftime("%Y-%m-%d")
    table_rows = "\n".join(
        f"| {v} | {i} | {tr*100:.2f}% | {te*100:.2f}% | {t:.0f}s |"
        for v, i, tr, te, t in rows)
    lines = [
        "# exp20 — Delightful ES Report",
        "",
        f"**Date:** {date_str}  **Epochs:** {MAX_EPOCHS}  **n_pop:** 128  "
        f"**σ:** 0.001  **rank schedule:** 8→2 at ep{RANK_SWITCH_EP}",
        "",
        "## Hypothesis",
        "",
        "Apply Delightful Policy Gradient (arxiv:2603.14608) gating to ES.",
        "Gate = σ(fitness_i × centered_surprise_i / η) where surprise = -log p(eps).",
        "Suppresses blunders (rare large eps with negative fitness).",
        "",
        "## Results",
        "",
        "| Variant | Config | Train | Test | Time |",
        "|---|---|---|---|---|",
        table_rows,
        "",
        "## Accuracy curves",
        "",
        f"![accuracy]({urls['accuracy']})",
        "",
        "## |fitness| and gate diagnostics",
        "",
        f"![diagnostics]({urls['diagnostics']})",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    logging.info(f"Report written to {REPORT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    all_results   = []
    plot_data     = []

    # ── A: Standard ES + rank schedule ──
    name = "A-standard  rank=8→2"
    logging.info(f"\n{'='*60}\n{name}\n{'='*60}")
    fwd, params = make_f3_bilinear(4, 4, 4, 256)
    tr, te, t_s, hist = train_standard(name, fwd, params, X_train, y_train, X_test, y_test)
    logging.info(f"  → train={tr:.4f}  test={te:.4f}  {t_s:.0f}s")
    all_results.append((name, "rank=8→2 ep12", tr, te, t_s))
    plot_data.append((name, "steelblue", hist))
    append_result({"experiment": "exp20", "variant": "standard",
                   "train_acc": round(tr,6), "test_acc": round(te,6),
                   "time_s": round(t_s,1), "epochs": MAX_EPOCHS})

    # ── B: DG-ES η=1 + rank schedule ──
    name = "B-DG-ES  η=1.0"
    logging.info(f"\n{'='*60}\n{name}\n{'='*60}")
    fwd, params = make_f3_bilinear(4, 4, 4, 256)
    tr, te, t_s, hist = train_dg(name, fwd, params, X_train, y_train, X_test, y_test, eta=1.0)
    logging.info(f"  → train={tr:.4f}  test={te:.4f}  {t_s:.0f}s")
    all_results.append((name, "DG η=1.0, rank=8→2", tr, te, t_s))
    plot_data.append((name, "darkorange", hist))
    append_result({"experiment": "exp20", "variant": "dg_eta1",
                   "train_acc": round(tr,6), "test_acc": round(te,6),
                   "time_s": round(t_s,1), "epochs": MAX_EPOCHS})

    # ── C: DG-ES η=0.1 + rank schedule ──
    name = "C-DG-ES  η=0.1"
    logging.info(f"\n{'='*60}\n{name}\n{'='*60}")
    fwd, params = make_f3_bilinear(4, 4, 4, 256)
    tr, te, t_s, hist = train_dg(name, fwd, params, X_train, y_train, X_test, y_test, eta=0.1)
    logging.info(f"  → train={tr:.4f}  test={te:.4f}  {t_s:.0f}s")
    all_results.append((name, "DG η=0.1, rank=8→2", tr, te, t_s))
    plot_data.append((name, "green", hist))
    append_result({"experiment": "exp20", "variant": "dg_eta01",
                   "train_acc": round(tr,6), "test_acc": round(te,6),
                   "time_s": round(t_s,1), "epochs": MAX_EPOCHS})

    # ── Plots + report ──
    logging.info("Generating plots and uploading...")
    urls = make_plots(plot_data)
    write_report(urls, all_results)

    print(f"\n{'='*65}")
    print(f"{'Variant':<25} {'Config':<22} {'Test':>7} {'Time':>7}")
    print(f"{'-'*65}")
    for v, cfg, tr, te, t_s in all_results:
        print(f"{v:<25} {cfg:<22} {te:>7.4f} {t_s:>6.0f}s")
    print(f"{'='*65}")
    print(f"\nReport: {REPORT_PATH}")
    for k, u in urls.items():
        print(f"  {k}: {u}")
