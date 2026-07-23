"""
exp21 — DG-ES v2: use model's own action probabilities as surprisal.

Fix from exp20: the perturbation norm (chi-squared) has no semantic content.
The correct surprisal is the model's cross-entropy on the batch:
  ℓ_i = (CE_+ + CE_-) / 2  =  how uncertain the model is under perturbation i

Mapping:
  advantage  U_i = -fitness_i = CE_- - CE_+  (positive = good direction)
  surprisal  ℓ_i = (CE_+ + CE_-) / 2         (model's own uncertainty)
  delight    χ_i = U_i × ℓ_i = -fitness_i × ce_avg_i
  gate       w_i = sigmoid(χ_i / η)

Scale: fitness ~ 0.01, CE ~ 1.0 → χ ~ 0.01. Need η ~ 0.01 for useful discrimination.

Variants (all use rank schedule 8→2 at ep12, 100 epochs):
  A: Standard ES            (control)
  B: DG-ES-v2  η=0.01       (properly scaled)
  C: DG-ES-v2  η=0.001      (sharper)

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/es/experiments21.py
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
RANK_SWITCH_EP = 12

JSONL       = Path(__file__).parent / "results_mnist.jsonl"
REPORT_PATH = Path(__file__).parent / "report21.md"

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

def _make_perturbation(pop_key, params, sigma, rank):
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


def make_standard_es_step(fwd, sigma, n_pop, tx, rank):
    def pair_estimate(pop_key, params, bx, by):
        eps = _make_perturbation(pop_key, params, sigma, rank)
        ce  = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        ce_pos = ce({k: v + eps[k] for k, v in params.items()})
        ce_neg = ce({k: v - eps[k] for k, v in params.items()})
        return ce_pos - ce_neg, eps

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


def make_dg_v2_step(fwd, sigma, n_pop, tx, rank, eta):
    """
    DG-ES v2: gate using the model's own cross-entropy as surprisal.
      surprisal ℓ_i = (CE_+ + CE_-) / 2   — model uncertainty on this pair
      advantage U_i = -fitness_i = CE_- - CE_+  (positive = pair helped)
      delight   χ_i = U_i × ℓ_i = -fitness_i × ce_avg_i
      gate      w_i = sigmoid(χ_i / η)
    Weighted gradient: Σ_i w_i × fitness_i × eps_i / (n_pop × rank × σ²)
    """
    def pair_estimate(pop_key, params, bx, by):
        eps = _make_perturbation(pop_key, params, sigma, rank)
        ce  = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        ce_pos = ce({k: v + eps[k] for k, v in params.items()})
        ce_neg = ce({k: v - eps[k] for k, v in params.items()})
        fitness  = ce_pos - ce_neg
        ce_avg   = (ce_pos + ce_neg) * 0.5
        return fitness, eps, ce_avg

    @jax.jit
    def es_step(params, opt_state, bx, by, pop_keys):
        fitnesses, epses, ce_avgs = jax.vmap(
            pair_estimate, in_axes=(0, None, None, None)
        )(pop_keys, params, bx, by)

        # delight = advantage × surprisal = -fitness × ce_avg
        # (advantage is negative fitness: positive when pair improved model)
        delight = -fitnesses * ce_avgs
        gates   = jax.nn.sigmoid(delight / eta)

        weighted_fit = fitnesses * gates
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", weighted_fit, e) / (n_pop * rank * sigma**2),
            epses)
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state,
                jnp.mean(jnp.abs(fitnesses)), jnp.mean(gates), jnp.mean(ce_avgs))

    return es_step


# ── Training loops ─────────────────────────────────────────────────────────

def train_standard(name, fwd, params, X_tr, y_tr, X_te, y_te,
                   sigma=0.001, n_pop=128, rank_hi=8, rank_lo=2):
    num_batches = len(X_tr) // BATCH_SIZE
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT,
                                              MAX_EPOCHS * num_batches,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    step_hi   = make_standard_es_step(fwd, sigma, n_pop, tx, rank_hi)
    step_lo   = make_standard_es_step(fwd, sigma, n_pop, tx, rank_lo)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    history   = {"train": [], "test": [], "fitness": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        step = step_hi if epoch < RANK_SWITCH_EP else step_lo
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng, epoch_fit = key, []
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
        rank_now = rank_hi if epoch < RANK_SWITCH_EP else rank_lo
        logging.info(f"[{name}] ep {epoch:3d}  train={tr:.4f}  test={te:.4f}"
                     f"  |fit|={mf:.6f}  rank={rank_now}  {time.perf_counter()-t0:.1f}s")

    return tr, te, time.perf_counter() - t0, history


def train_dg_v2(name, fwd, params, X_tr, y_tr, X_te, y_te,
                sigma=0.001, n_pop=128, rank_hi=8, rank_lo=2, eta=0.01):
    num_batches = len(X_tr) // BATCH_SIZE
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT,
                                              MAX_EPOCHS * num_batches,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    step_hi   = make_dg_v2_step(fwd, sigma, n_pop, tx, rank_hi, eta=eta)
    step_lo   = make_dg_v2_step(fwd, sigma, n_pop, tx, rank_lo, eta=eta)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    history   = {"train": [], "test": [], "fitness": [], "gate": [], "ce_avg": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        step = step_hi if epoch < RANK_SWITCH_EP else step_lo
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fit, epoch_gate, epoch_ce = [], [], []
        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, opt_state, fit, gate, ce_avg = step(params, opt_state, bx, by, pop_keys)
            epoch_fit.append(float(fit))
            epoch_gate.append(float(gate))
            epoch_ce.append(float(ce_avg))

        tr = acc_fn(fwd, params, X_tr, y_tr)
        te = acc_fn(fwd, params, X_te, y_te)
        mf = sum(epoch_fit) / len(epoch_fit)
        mg = sum(epoch_gate) / len(epoch_gate)
        mc = sum(epoch_ce)   / len(epoch_ce)
        history["train"].append(tr); history["test"].append(te)
        history["fitness"].append(mf); history["gate"].append(mg); history["ce_avg"].append(mc)
        rank_now = rank_hi if epoch < RANK_SWITCH_EP else rank_lo
        logging.info(f"[{name}] ep {epoch:3d}  train={tr:.4f}  test={te:.4f}"
                     f"  |fit|={mf:.6f}  gate={mg:.3f}  CE={mc:.3f}"
                     f"  rank={rank_now}  {time.perf_counter()-t0:.1f}s")

    return tr, te, time.perf_counter() - t0, history


# ── Plots ──────────────────────────────────────────────────────────────────

def make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(MAX_EPOCHS))
    urls   = {}

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, hist in results:
        ax.plot(epochs, [v*100 for v in hist["train"]], color=color, linestyle="--", alpha=0.4)
        ax.plot(epochs, [v*100 for v in hist["test"]],  color=color, linewidth=2, label=label)
    ax.axvline(RANK_SWITCH_EP, color="gray", linestyle=":", alpha=0.5, label=f"rank switch ep{RANK_SWITCH_EP}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("exp21 — DG-ES v2 (action probs as surprisal) vs Standard ES")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["accuracy"] = save_matplotlib_figure("exp21_accuracy", fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, color, hist in results:
        axes[0].plot(epochs, hist["fitness"], color=color, linewidth=2, label=label)
        if "gate" in hist:
            axes[1].plot(epochs, hist["gate"],   color=color, linewidth=2, label=label)
            axes[2].plot(epochs, hist["ce_avg"], color=color, linewidth=2, label=label)
    axes[0].set_title("|fitness|"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("mean gate (0.5=neutral)"); axes[1].axhline(0.5, color="gray", linestyle=":")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].set_title("mean CE_avg (model uncertainty)"); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    urls["diagnostics"] = save_matplotlib_figure("exp21_diagnostics", fig)
    plt.close(fig)

    return urls


def write_report(urls, rows):
    date_str = datetime.now().strftime("%Y-%m-%d")
    table_rows = "\n".join(
        f"| {v} | {cfg} | {tr*100:.2f}% | {te*100:.2f}% | {t:.0f}s |"
        for v, cfg, tr, te, t in rows)
    lines = [
        "# exp21 — DG-ES v2 Report",
        "",
        f"**Date:** {date_str}  **Epochs:** {MAX_EPOCHS}  **n_pop:** 128  "
        f"**σ:** 0.001  **rank schedule:** 8→2 at ep{RANK_SWITCH_EP}",
        "",
        "## Fix over exp20",
        "",
        "exp20 used perturbation norm as surprisal (chi-squared noise, no semantic content).",
        "v2 uses the model's own cross-entropy: `ℓ_i = (CE_+ + CE_-)/2`.",
        "Delight: `χ_i = -fitness_i × CE_avg_i` (sign: positive when pair helped AND model was uncertain).",
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
        "## Diagnostics",
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

    all_results, plot_data = [], []

    variants = [
        ("A-standard",      None,   "steelblue"),
        ("B-DG-v2  η=0.01", 0.01,   "darkorange"),
        ("C-DG-v2  η=0.001",0.001,  "green"),
    ]

    for vname, eta, color in variants:
        logging.info(f"\n{'='*60}\n{vname}\n{'='*60}")
        fwd, params = make_f3_bilinear(4, 4, 4, 256)

        if eta is None:
            tr, te, t_s, hist = train_standard(vname, fwd, params, X_train, y_train, X_test, y_test)
            cfg = "rank=8→2 ep12 (control)"
        else:
            tr, te, t_s, hist = train_dg_v2(vname, fwd, params, X_train, y_train, X_test, y_test, eta=eta)
            cfg = f"DG-v2 η={eta}, rank=8→2"

        logging.info(f"  → train={tr:.4f}  test={te:.4f}  {t_s:.0f}s")
        all_results.append((vname, cfg, tr, te, t_s))
        plot_data.append((vname, color, hist))
        append_result({"experiment": "exp21", "variant": vname,
                       "eta": eta, "train_acc": round(tr,6), "test_acc": round(te,6),
                       "time_s": round(t_s,1), "epochs": MAX_EPOCHS})

    logging.info("Generating plots and uploading...")
    urls = make_plots(plot_data)
    write_report(urls, all_results)

    print(f"\n{'='*65}")
    print(f"{'Variant':<25} {'Config':<25} {'Test':>7} {'Time':>7}")
    print(f"{'-'*65}")
    for v, cfg, tr, te, t_s in all_results:
        print(f"{v:<25} {cfg:<25} {te:>7.4f} {t_s:>6.0f}s")
    print(f"{'='*65}")
    for k, u in urls.items():
        print(f"  {k}: {u}")
